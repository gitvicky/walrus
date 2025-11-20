import torch


def test_import():
    import walrus  # noqa: F401
    from walrus.models.shared_utils.mlps import MLP

    model = MLP(3)
    model(torch.randn(1, 3))
