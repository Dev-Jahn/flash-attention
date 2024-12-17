import torch
from flashattn_2d import flash_attn_2d_func


if __name__ == "__main__":
    torch.manual_seed(42)
    # S = 24
    # SS = S * S
    # window_size = 3  # attend with 3x3 adjacent tokens in spatial dim
    # num_attend = window_size**2
    # w = 1

    # B = 4
    # T = 32
    # seq_len = S**2 * T
    # H = 32
    # Hkv = 8
    # D = 64

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

    out = flash_attn_2d_func(
        q.view(B * T, SS, -1, D),
        k.view(B * T, SS, -1, D),
        v.view(B * T, SS, -1, D),
        S,
        S,
        window_size=(1, 1, 1, 1),
    )[0].view(B, T, SS, H, D)
    print(out)
