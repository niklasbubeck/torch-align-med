import torch

from torch_align_med import calculate_alignment_metrics, make_grid_coords


def test_full_pipeline_runs():
    a = torch.randn(2, 16, 32)
    b = torch.randn(2, 16, 32)
    coords = make_grid_coords((4, 4))

    out = calculate_alignment_metrics(
        a, b,
        coords=coords,
        patch_cosine=True,
        linear_cka=True,
        l_mdms=True,
        lds=True,
        cds=True,
        rmsc=True,
    )

    assert "patch_cosine" in out
    assert "linear_cka" in out
    assert "l_mdms" in out
    assert "lds_input1" in out
    assert "lds_input2" in out
    assert "cds_input1" in out
    assert "rmsc_input1" in out
    assert all(isinstance(v, float) for v in out.values())


def test_no_flags_returns_empty():
    a = torch.randn(2, 16, 32)
    b = torch.randn(2, 16, 32)
    assert calculate_alignment_metrics(a, b) == {}


def test_metric_kwargs_threaded_through():
    a = torch.randn(2, 16, 32)
    b = torch.randn(2, 16, 32)
    coords = make_grid_coords((4, 4))
    out = calculate_alignment_metrics(
        a, b,
        coords=coords,
        lds=True,
        metric_kwargs={"lds": {"init": {"r_near": 2, "r_far": 4}}},
    )
    assert "lds_input1" in out
