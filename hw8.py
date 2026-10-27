"""Mini diffusion on MNIST. Timesteps are 1..100; index 0 is clean data."""
from pathlib import Path
import json
import time
import random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.datasets import MNIST
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEED = 6812
T = 100
OUT = Path('hw8_outputs')
OUT.mkdir(exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def seed_all(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)

def schedule(device=DEVICE):
    beta = torch.cat([torch.zeros(1), torch.linspace(0.0001, 0.02, T)]).to(device)
    alpha = 1 - beta
    abar = alpha.cumprod(0)
    return beta, alpha, abar

def extract(values, t):
    return values[t].view(-1, 1, 1, 1)

def q_sample(x0, t, noise, abar):
    a = extract(abar, t)
    return a.sqrt() * x0 + (1 - a).sqrt() * noise

class MiniCNN(nn.Module):
    """Eight convolutions, timestep embeddings, and two spatial resolutions."""
    def __init__(self):
        super().__init__()
        self.time = nn.Sequential(nn.Embedding(T + 1, 64), nn.Linear(64, 64), nn.SiLU())
        self.c1 = nn.Conv2d(1, 32, 3, padding=1)
        self.c2 = nn.Conv2d(32, 32, 3, padding=1)
        self.down = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.c3 = nn.Conv2d(64, 64, 3, padding=1)
        self.c4 = nn.Conv2d(64, 64, 3, padding=1)
        self.up = nn.Conv2d(64, 32, 3, padding=1)
        self.c5 = nn.Conv2d(64, 32, 3, padding=1)
        self.out = nn.Conv2d(32, 1, 3, padding=1)
        self.t32 = nn.Linear(64, 32)
        self.t64 = nn.Linear(64, 64)

    def forward(self, x, t):
        emb = self.time(t)
        e32 = self.t32(emb)[:, :, None, None]
        e64 = self.t64(emb)[:, :, None, None]
        h = F.silu(self.c1(x) + e32)
        skip = F.silu(self.c2(h))
        h = F.silu(self.down(skip) + e64)
        h = F.silu(self.c3(h))
        h = F.silu(self.c4(h) + h)
        h = F.interpolate(h, size=skip.shape[-2:], mode='nearest')
        h = F.silu(self.up(h) + e32)
        return self.out(F.silu(self.c5(torch.cat([h, skip], dim=1))))

def forward_plot(data):
    _, _, abar = schedule('cpu')
    x0 = data[0:1].float().unsqueeze(1) / 127.5 - 1
    noise = torch.randn_like(x0)
    steps = [0, 1, 10, 25, 50, 75, 100]
    fig, axes = plt.subplots(1, len(steps), figsize=(12, 2.1))
    for ax, t in zip(axes, steps):
        xt = q_sample(x0, torch.tensor([t]), noise, abar)
        ax.imshow(xt[0, 0], cmap='gray', vmin=-1, vmax=1)
        ax.set_title(f't = {t}')
        ax.axis('off')
    fig.suptitle('Forward diffusion (same noise draw at each timestep)')
    fig.tight_layout()
    fig.savefig(OUT / 'forward_diffusion.png', dpi=160)
    plt.close(fig)

def train(epochs=50):
    seed_all()
    dataset = MNIST('hw8_data', train=True, download=True)
    forward_plot(dataset.data)
    # Keep normalized images on GPU to avoid repeated CPU data conversion.
    data = dataset.data.unsqueeze(1).to(DEVICE, dtype=torch.float32) / 127.5 - 1
    model = MiniCNN().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    _, _, abar = schedule()
    history = []
    start_epoch = 0
    checkpoint = OUT / 'checkpoint.pt'
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state['model'])
        opt.load_state_dict(state['optimizer'])
        history = state['loss']
        start_epoch = len(history)
        torch.set_rng_state(state['rng'].cpu())
        if DEVICE.type == 'cuda':
            torch.cuda.set_rng_state(state['cuda_rng'].cpu())
    model.train()
    for epoch in range(start_epoch, epochs):
        start = time.perf_counter()
        total = 0.0
        order = torch.randperm(len(data), device=DEVICE)
        for ids in order.split(128):
            x0 = data[ids]
            t = torch.randint(1, T + 1, (len(ids),), device=DEVICE)
            eps = torch.randn_like(x0)
            prediction = model(q_sample(x0, t, eps, abar), t)
            loss = F.mse_loss(prediction, eps)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item() * len(ids)
        history.append(total / len(data))
        torch.save({'model': model.state_dict(), 'optimizer': opt.state_dict(),
                    'loss': history, 'rng': torch.get_rng_state(),
                    'cuda_rng': torch.cuda.get_rng_state() if DEVICE.type == 'cuda' else torch.get_rng_state()}, checkpoint)
        print(f'Epoch {epoch + 1:02d}/{epochs}: MSE={history[-1]:.6f}, {time.perf_counter()-start:.1f}s', flush=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, len(history) + 1), history)
    ax.set(xlabel='Epoch', ylabel='Mean noise-prediction MSE', title='Training on 60,000 MNIST images')
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT / 'training_loss.png', dpi=160)
    plt.close(fig)
    return model.eval(), history

@torch.no_grad()
def ddpm(model, initial):
    beta, alpha, abar = schedule(initial.device)
    x = initial.clone()
    for t in range(T, 0, -1):
        ts = torch.full((len(x),), t, device=x.device, dtype=torch.long)
        eps = model(x, ts)
        mean = (x - beta[t] / (1 - abar[t]).sqrt() * eps) / alpha[t].sqrt()
        variance = beta[t] * (1 - abar[t-1]) / (1 - abar[t])
        x = mean + variance.sqrt() * torch.randn_like(x) if t > 1 else mean
    return x

@torch.no_grad()
def ddim(model, initial, steps):
    _, _, abar = schedule(initial.device)
    # Exactly `steps` model evaluations; the last transition ends at clean t=0.
    times = torch.linspace(T, 1, steps).round().long().tolist()
    x = initial.clone()
    for t, prev in zip(times, times[1:] + [0]):
        ts = torch.full((len(x),), t, device=x.device, dtype=torch.long)
        eps = model(x, ts)
        x0 = (x - (1 - abar[t]).sqrt() * eps) / abar[t].sqrt()
        x = abar[prev].sqrt() * x0 + (1 - abar[prev]).sqrt() * eps
    return x

def compare(model, history):
    seed_all(SEED + 1)
    initial = torch.randn(32, 1, 28, 28, device=DEVICE)
    results, timings = {}, {}
    # Warm up CUDA before timing the full batches.
    with torch.no_grad():
        model(initial, torch.full((32,), T, device=DEVICE, dtype=torch.long))
    for name, sampler in [('DDPM (100 steps)', lambda: ddpm(model, initial)),
                          ('DDIM (50 steps)', lambda: ddim(model, initial, 50)),
                          ('DDIM (20 steps)', lambda: ddim(model, initial, 20))]:
        if DEVICE.type == 'cuda': torch.cuda.synchronize()
        start = time.perf_counter()
        samples = sampler()
        if DEVICE.type == 'cuda': torch.cuda.synchronize()
        timings[name] = time.perf_counter() - start
        results[name] = samples.cpu()
        assert torch.isfinite(samples).all()
    fig, axes = plt.subplots(6, 16, figsize=(16, 7))
    for group, (name, samples) in enumerate(results.items()):
        for i in range(32):
            ax = axes[2 * group + i // 16, i % 16]
            ax.imshow(samples[i, 0].clamp(-1, 1), cmap='gray', vmin=-1, vmax=1)
            ax.axis('off')
        axes[2 * group, 0].set_title(name, loc='left', fontsize=12, pad=10)
    fig.suptitle('Same starting Gaussian noise for all three samplers', fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / 'sampling_comparison.png', dpi=180)
    plt.close(fig)
    torch.save({'initial': initial.cpu(), 'samples': results}, OUT / 'samples.pt')
    metrics = {'seed': SEED, 'epochs': len(history), 'batch_size': 128,
               'train_images': 60000, 'loss': history, 'sampling_seconds_32_images': timings,
               'alpha_bar_T': schedule('cpu')[2][-1].item(),
               'parameters': sum(p.numel() for p in model.parameters()),
               'device': torch.cuda.get_device_name() if DEVICE.type == 'cuda' else 'CPU',
               'torch_version': str(torch.__version__)}
    (OUT / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: v for k, v in metrics.items() if k != 'loss'}, indent=2), flush=True)
    return metrics

if __name__ == '__main__':
    model, history = train()
    compare(model, history)
