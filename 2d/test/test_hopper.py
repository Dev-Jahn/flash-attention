import torch
from flashattn_2d.reference import spatial_flash_attn_2d_v2


if __name__ == "__main__":
    S = 5
    SS = S * S
    window_size = 3  # attend with 3x3 adjacent tokens in spatial dim
    num_attend = window_size**2
    w = 1

    B = 1
    T = 1
    seq_len = S**2 * T
    H = 4
    Hkv = 1
    D = 64

    dtype = torch.bfloat16
    device = torch.device("cuda:0")

    q = torch.rand(B, seq_len, H, D, device=device, dtype=dtype)
    k = torch.rand(B, seq_len, Hkv, D, device=device, dtype=dtype)
    v = torch.rand(B, seq_len, Hkv, D, device=device, dtype=dtype)

    out_v2 = spatial_flash_attn_2d_v2(
        q.view(B * T, S**2, -1, D),
        k.view(B * T, S**2, -1, D),
        v.view(B * T, S**2, -1, D),
        w=-1,
    )
    print(out_v2)
