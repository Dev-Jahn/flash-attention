import pytest
import torch
from flashattn_2d import flash_attn_2d_func
from flashattn_2d.reference import spatial_flash_attn_2d_v2


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("d", [64, 128])
@pytest.mark.parametrize("height,width", [(24, 24), (27, 27), (24, 27), (27, 24)])
@pytest.mark.parametrize("w", [-1, 1, 2, 3])
def test_flash_attn_2d_output(height, width, d, w, dtype):
    device = "cuda"
    torch.random.manual_seed(0)

    batch_size = 4
    nheads = 8
    seqlen = height * width

    window_size = (w, w, w, w)

    q = torch.randn(
        batch_size, seqlen, nheads, d, device=device, dtype=dtype, requires_grad=True
    )
    k = torch.randn(
        batch_size, seqlen, nheads, d, device=device, dtype=dtype, requires_grad=True
    )
    v = torch.randn(
        batch_size, seqlen, nheads, d, device=device, dtype=dtype, requires_grad=True
    )

    out_ref = spatial_flash_attn_2d_v2(q, k, v, height, width, window_size)

    out_fa, _ = flash_attn_2d_func(
        q, k, v, height=height, width=width, causal=False, window_size=window_size
    )

    print(f"Output max diff: {(out_fa - out_ref).abs().max().item()}")
    print(f"Output mean diff: {(out_fa - out_ref).abs().mean().item()}")

    assert (out_fa - out_ref).abs().max().item() <= 2e-2
    assert (out_fa - out_ref).abs().mean().item() <= 2e-3
