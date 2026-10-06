"""Synthetic checks only; no STL-10 download or experiment results."""
import json
import os
import tempfile
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import numpy as np
import torch
from PIL import Image
import hw6 as h


def main():
    torch.set_num_threads(2)
    h.seed_all(42)
    h.DEVICE = torch.device('cpu')
    a, b = torch.randn(4, 8, requires_grad=True), torch.randn(4, 8, requires_grad=True)
    z = torch.nn.functional.normalize(torch.cat([a, b]), dim=1)
    terms = []
    for i in range(8):
        scores = torch.stack([torch.dot(z[i], z[j])/0.2 for j in range(8) if j != i])
        terms.append(-torch.dot(z[i], z[(i+4)%8])/0.2 + torch.logsumexp(scores, 0))
    loss = h.nt_xent(a, b)
    torch.testing.assert_close(loss, torch.stack(terms).mean())
    loss.backward()
    assert torch.isfinite(a.grad).all() and torch.isfinite(b.grad).all()
    basis = torch.eye(4)
    assert h.nt_xent(basis, basis) < h.nt_xent(basis, basis.roll(1, 0))
    scores, ids = h.nearest_neighbors(torch.tensor([[1., 0.], [2., 0.], [0., 1.], [-1., 0.]]), 0, 2)
    assert ids.tolist() == [1, 2]
    torch.testing.assert_close(scores, torch.tensor([1., 0.]))
    _, eval_tf, contrast_tf = h.build_transforms()
    image = Image.fromarray(np.random.default_rng(0).integers(0, 256, (96, 96, 3), dtype=np.uint8))
    base = [(image, 0)] * 4
    v1, v2 = h.TwoViews(base, range(4), contrast_tf)[0]
    assert v1.shape == v2.shape == (3, 96, 96) and not torch.equal(v1, v2)
    rotated, target = h.RotationDataset(base, eval_tf)[0]
    torch.testing.assert_close(rotated, torch.rot90(eval_tf(image), target, (1, 2)))
    with tempfile.TemporaryDirectory() as directory:
        h.OUT = Path(directory)
        h.CONFIG['linear_epochs'] = 2
        encoder = h.make_encoder()
        batches = h.loader(h.TensorDataset(torch.randn(4, 3, 32, 32), torch.tensor([0, 1, 2, 3])))
        before = h.state_digest(encoder)
        head = h.linear_probe(encoder, batches, 'check')
        assert before == h.state_digest(encoder)
        accuracy, features, labels, predictions = h.test_result(encoder, head, batches)
        assert features.shape == (4, 512) and 0 <= accuracy <= 1
        assert all(not p.requires_grad for p in encoder.parameters())
        h.plt.show = lambda: None
        labels = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2])
        names = ['Supervised', 'Rotation SSL', 'SimCLR']
        results = {name: {'test_accuracy': 0.25} for name in names}
        features = {name: torch.randn(9, 512) for name in names}
        for stage in ['supervised', 'rotation', 'simclr']:
            (h.OUT / f'{stage}_history.json').write_text('[{"epoch": 1, "loss": 1.0}]')
        previous_cwd = Path.cwd()
        try:
            os.chdir(directory)
            h.report(results, features, labels, [(image, int(y)) for y in labels], ['a', 'b', 'c'])
            with h.capture_run('synthetic logger check'):
                print('test console message')
            try:
                with h.capture_run('intentional exception check'):
                    raise ValueError('synthetic logging test')
            except ValueError:
                pass
            logged = Path('RUN_LOG.txt').read_text(encoding='utf-8')
            assert 'test console message' in logged and 'Traceback' in logged
            assert 'ValueError: synthetic logging test' in logged and '+00:00' in logged
        finally:
            os.chdir(previous_cwd)
        assert (h.OUT / 'nearest_neighbors.png').exists()
        assert len((h.OUT / 'neighbors.csv').read_text().splitlines()) == 46
    notebook = json.loads(Path('HW6_STL10_Representation_Learning.ipynb').read_text(encoding='utf-8'))
    assert notebook['nbformat'] == 4
    first = ''.join(notebook['cells'][0]['source'])
    assert all(f'{k} = {v}' in first for k, v in h.PERSONAL.items())
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            compile(''.join(cell['source']), f'cell_{i}', 'exec')
            assert cell['outputs'] == []
    print('PASS: loss reference/gradients, cosine retrieval, transforms, frozen encoder/BN,')
    print('linear probe, report export, notebook syntax. Synthetic checks, not STL-10 results.')


if __name__ == '__main__':
    main()
