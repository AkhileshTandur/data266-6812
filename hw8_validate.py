"""Check indexing and sampler equations independently of learned quality."""
import torch
from torch import nn
from hw8 import schedule, q_sample, MiniCNN, ddpm, ddim, SEED, DEVICE

torch.manual_seed(SEED)
beta, alpha, abar = schedule()
assert beta.shape == (101,) and abar[0] == 1
assert torch.allclose(beta[1:], torch.linspace(.0001, .02, 100, device=DEVICE))
x0 = torch.randn(3, 1, 28, 28, device=DEVICE)
eps = torch.randn_like(x0)
assert torch.equal(q_sample(x0, torch.zeros(3, dtype=torch.long, device=DEVICE), eps, abar), x0)
t = torch.tensor([1, 50, 100], device=DEVICE)
xt = q_sample(x0, t, eps, abar)
recovered = (xt - (1 - abar[t, None, None, None]).sqrt() * eps) / abar[t, None, None, None].sqrt()
assert torch.allclose(recovered, x0, atol=1e-6)
net = MiniCNN().to(DEVICE)
assert sum(isinstance(m, nn.Conv2d) for m in net.modules()) >= 4
assert net(xt, t).shape == xt.shape
net(xt, t).square().mean().backward()
assert net.time[0].weight.grad.abs().sum() > 0

class Oracle(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []
    def forward(self, x, t):
        self.calls.append(int(t[0]))
        a = abar[t, None, None, None]
        return (x - a.sqrt() * x0) / (1 - a).sqrt()

initial = torch.randn_like(x0)
for steps in (20, 50):
    oracle = Oracle()
    result = ddim(oracle, initial, steps)
    assert len(oracle.calls) == steps and len(set(oracle.calls)) == steps
    assert oracle.calls[0] == 100 and oracle.calls[-1] == 1
    assert torch.allclose(result, x0, atol=2e-5)
    assert torch.equal(result, ddim(Oracle(), initial, steps))
oracle = Oracle()
assert torch.allclose(ddpm(oracle, initial), x0, atol=2e-5)
assert oracle.calls == list(range(100, 0, -1))
print('PASS: schedule, forward reconstruction, output shape, timestep gradients, oracle reverse recovery, step counts, and DDIM determinism.')
