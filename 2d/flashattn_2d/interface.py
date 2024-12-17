from typing import Optional, Union

# isort: off
# We need to import the CUDA kernels after importing torch
import torch
import torch.nn as nn

import flashattn_2d_hopper_cuda  # type: ignore

# isort: on


def maybe_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


def _flash_attn_2d_forward(
    q,
    k,
    v,
    height,
    width,
    softmax_scale,
    window_size,
    descale_q=None,
    descale_k=None,
    descale_v=None,
    gqa_parallel=False,
):
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    out, q, k, v, out_padded, softmax_lse, S_dmask = flashattn_2d_hopper_cuda.fwd(
        q,
        k,
        v,
        None,
        softmax_scale,
        descale_q,
        descale_k,
        descale_v,
        False,  # is_causal
        window_size[0],
        window_size[1],
        window_size[2],
        window_size[3],
        height,
        width,
        gqa_parallel,
    )
    return out, q, k, v, out_padded, softmax_lse, S_dmask


def _flash_attn_2d_backward(
    dout,
    q,
    k,
    v,
    out,
    height,
    width,
    softmax_lse,
    dq,
    dk,
    dv,
    softmax_scale,
    window_size,
    deterministic=False,
):
    # dq, dk, dv are allocated by us so they should already be contiguous
    dout, q, k, v, out = [maybe_contiguous(x) for x in (dout, q, k, v, out)]
    dq, dk, dv, softmax_d, *rest = flashattn_2d_hopper_cuda.bwd(
        dout,
        q,
        k,
        v,
        out,
        softmax_lse,
        dq,
        dk,
        dv,
        softmax_scale,
        False,  # is_causal
        window_size[0],
        window_size[1],
        window_size[2],
        window_size[3],
        height,
        width,
        deterministic,
    )
    return dq, dk, dv, softmax_d


class FlashAttn2DFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        height,
        width,
        softmax_scale,
        window_size,
        deterministic=False,
        descale_q=None,
        descale_k=None,
        descale_v=None,
        gqa_parallel=False,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        out, q, k, v, out_padded, softmax_lse, S_dmask = _flash_attn_2d_forward(
            q,
            k,
            v,
            height,
            width,
            softmax_scale,
            window_size,
            descale_q=descale_q,
            descale_k=descale_k,
            descale_v=descale_v,
            gqa_parallel=gqa_parallel,
        )
        ctx.save_for_backward(q, k, v, out_padded, softmax_lse)
        ctx.height, ctx.width = height, width
        ctx.softmax_scale = softmax_scale
        ctx.window_size = window_size
        ctx.deterministic = deterministic
        ctx.gqa_parallel = gqa_parallel
        return out, softmax_lse

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse = ctx.saved_tensors
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        _flash_attn_2d_backward(
            dout,
            q,
            k,
            v,
            out,
            ctx.height,
            ctx.width,
            softmax_lse,
            dq,
            dk,
            dv,
            ctx.softmax_scale,
            ctx.window_size,
            ctx.deterministic,
        )
        dq = dq[..., : dout.shape[-1]]  # We could have padded the head dimension
        dk = dk[..., : dout.shape[-1]]
        dv = dv[..., : dout.shape[-1]]
        return dq, dk, dv, None, None, None, None, None, None, None, None, None


def flash_attn_2d_func(
    q,
    k,
    v,
    height,
    width,
    softmax_scale=None,
    window_size=(-1, -1, -1, -1),
    deterministic=False,
    descale_q=None,
    descale_k=None,
    descale_v=None,
    gqa_parallel=False,
):
    """
    Arguments:
        *others are same*
        height: int. The number of tokens along the second dimension
        width: int. The number of tokens along the first dimension
        window_size: (left, right, top, bottom).
    """
    return FlashAttn2DFunc.apply(
        q,
        k,
        v,
        height,
        width,
        softmax_scale,
        window_size,
        deterministic,
        descale_q,
        descale_k,
        descale_v,
        gqa_parallel,
    )
