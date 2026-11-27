"""Regression checks for opt-in device-resident batching."""
import torch
from idsmultiplicity.training import _gpu_batches, _gpu_predict


def test_gpu_batcher_deterministic_and_complete():
    x = torch.arange(23, dtype=torch.float32).view(-1, 1)
    y = torch.arange(23, dtype=torch.long)
    batches = list(_gpu_batches(x, y, size=6, shuffle=True, seed=123))
    seen = torch.cat([b[1] for b in batches])
    assert sorted(seen.tolist()) == list(range(23))
    assert all(torch.equal(bx.flatten().long(), by) for bx, by in batches)
    repeat = torch.cat([by for bx, by in _gpu_batches(x, y, size=6, shuffle=True, seed=123)])
    assert torch.equal(seen, repeat)
    changed = torch.cat([by for bx, by in _gpu_batches(x, y, size=6, shuffle=True, seed=124)])
    assert not torch.equal(seen, changed)


def test_gpu_batcher_in_order_and_unlabeled():
    x = torch.arange(11, dtype=torch.float32).view(-1, 1)
    y = torch.arange(11, dtype=torch.long)
    assert torch.equal(torch.cat([b[0] for b in _gpu_batches(x, y, size=4)]), x)
    assert torch.equal(torch.cat([b[1] for b in _gpu_batches(x, y, size=4)]), y)
    assert torch.equal(torch.cat([b[0] for b in _gpu_batches(x, size=4)]), x)


def test_gpu_predict_matches_direct_softmax():
    model = torch.nn.Linear(2, 2).eval()
    x = torch.randn(11, 2)
    pred, probs = _gpu_predict(model, x, batch_size=4)
    expected = torch.softmax(model(x), dim=1).detach().numpy()
    assert probs.shape == (11, 2)
    assert (pred == expected.argmax(axis=1)).all()
    assert (abs(probs - expected) < 1e-6).all()
