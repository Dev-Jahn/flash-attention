import torch.nn.functional as F
from flash_attn_interface import flash_attn_func


def spatial_flash_attn_2d_v2(q, k, v, height, width, w):
    # q,k,v shape: (B, N, H, D) where N = T * height * width
    # Reshape to 5D
    B, N, H, D = q.shape
    _, _, Hkv, _ = k.shape
    assert (
        N % (height * width) == 0
    ), f"N ({N}) must be divisible by hT = N // (height * width)"
    T = N // (height * width)

    # Reshape to 5D (B, T, H*W, H, D)
    q = q.view(B, T, height * width, H, D)
    k = k.view(B, T, height * width, Hkv, D)
    v = v.view(B, T, height * width, Hkv, D)

    # global attention
    if w == -1:
        attn_out = flash_attn_func(
            q.view(B * T, height * width, H, D),
            k.view(B * T, height * width, Hkv, D),
            v.view(B * T, height * width, Hkv, D),
            softmax_scale=None,
            causal=False,
            window_size=(-1, -1),
            deterministic=False,
        )[0]
        return attn_out.view(B, N, H, D)

    # Reshape for spatial processing
    q = q.view(B * T, height, width, H * D)
    k = k.reshape(B * T, height, width, Hkv * D)
    v = v.reshape(B * T, height, width, Hkv * D)

    window_len = 2 * w + 1
    num_attend = window_len**2

    # Pad spatial dimensions -> extract windows -> gather and reshape
    mode = "constant"
    pad_value = 0
    q = (
        F.pad(q, (0, 0, w, w, w, w), mode=mode, value=pad_value)
        .unfold(1, height, 1)
        .unfold(2, width, 1)
        .permute(0, 4, 5, 1, 2, 3)
        .reshape(B, T * height * width * num_attend, H, D)
    )
    k = (
        F.pad(k, (0, 0, w, w, w, w), mode=mode, value=pad_value)
        .unfold(1, height, 1)
        .unfold(2, width, 1)
        .permute(0, 4, 5, 1, 2, 3)
        .reshape(B, T * height * width * num_attend, Hkv, D)
    )
    v = (
        F.pad(v, (0, 0, w, w, w, w), mode=mode, value=pad_value)
        .unfold(1, height, 1)
        .unfold(2, width, 1)
        .permute(0, 4, 5, 1, 2, 3)
        .reshape(B, T * height * width * num_attend, Hkv, D)
    )

    # Apply flash attention with window size
    attn_out = flash_attn_func(
        q,
        k,
        v,
        softmax_scale=None,
        causal=False,
        window_size=(num_attend // 2, num_attend // 2),  # (1,1) means window 3 in 1d
        deterministic=False,
    )[0]
    # slice centerpiece
    attn_out = attn_out.view(B, T * height * width, num_attend, -1)[
        :, :, num_attend // 2, :
    ]
    return attn_out.reshape(B, N, H, D)
