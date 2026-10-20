"""Check the HW7 objectives, split, checkpoints, and recorded test evaluation."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import FashionMNIST
from hw7 import FCAE, VAE, losses, evaluate, SEED


def main():
    root = Path(__file__).resolve().parent
    out = root / "hw7_outputs"
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # An independent closed-form case: BCE(logit=0)=log(2); KL(N(1,I)||N(0,I))=d/2.
    x = torch.full((3, 784), 0.5)
    rec, kl = losses(torch.zeros_like(x), x, torch.ones(3, 32), torch.zeros(3, 32))
    torch.testing.assert_close(rec, torch.full((3,), 784 * np.log(2), dtype=torch.float32))
    torch.testing.assert_close(kl, torch.full((3,), 16.0))
    _, kl_zero = losses(torch.zeros_like(x), x, torch.zeros(3, 32), torch.zeros(3, 32))
    assert torch.equal(kl_zero, torch.zeros_like(kl_zero))
    split = np.load(out / "split_indices.npz")
    assert len(split["train"]) == 55000 and len(split["validation"]) == 5000
    assert len(np.intersect1d(split["train"], split["validation"])) == 0
    assert np.array_equal(np.sort(np.concatenate([split["train"], split["validation"]])), np.arange(60000))
    metrics = json.loads((out / "metrics.json").read_text())
    test = FashionMNIST(root / "hw7_data", train=False, download=False)
    loader = DataLoader(TensorDataset(test.data.float().reshape(-1, 784) / 255, test.targets), batch_size=256)
    for name, cls in [("fc_ae", FCAE), ("vae", VAE)]:
        checkpoint = torch.load(out / f"{name}_best.pt", map_location=device, weights_only=True)
        model = cls(checkpoint["config"]["latent"]).to(device)
        model.load_state_dict(checkpoint["model"])
        measured = evaluate(model, loader, device)
        for key in ["mse", "mae", "bce_per_pixel", "objective", "kl"]:
            np.testing.assert_allclose(measured[key], metrics["results"][name][key], rtol=1e-6, atol=1e-7)
        assert all(torch.isfinite(p).all() for p in model.parameters())
        print(f"PASS {name}: checkpoint reload and full test metrics reproduced")
    # The sampling operation must propagate gradients into both VAE heads.
    torch.manual_seed(SEED)
    model = VAE()
    inputs = torch.rand(4, 784)
    logits, mu, logvar = model(inputs)
    rec, kl = losses(logits, inputs, mu, logvar)
    (rec + kl).mean().backward()
    for head in [model.mu, model.logvar]:
        assert head.weight.grad is not None
        assert torch.isfinite(head.weight.grad).all() and head.weight.grad.abs().sum() > 0
    print("PASS analytic BCE/KL, disjoint complete split, finite weights, and VAE gradients")


if __name__ == "__main__":
    main()
