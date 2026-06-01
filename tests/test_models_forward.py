import pytest


torch = pytest.importorskip("torch")


def test_small_cnn_forward():
    from edge_casting.models.registry import build_model

    model = build_model("small_cnn")
    y = model(torch.randn(2, 3, 96, 96))
    assert tuple(y.shape) == (2, 2)


def test_mobilenet_v3_forward_no_pretrained():
    from edge_casting.models.registry import build_model

    model = build_model("mobilenet_v3_small", pretrained=False)
    y = model(torch.randn(1, 3, 160, 160))
    assert tuple(y.shape) == (1, 2)
