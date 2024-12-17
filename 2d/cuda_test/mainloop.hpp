#pragma once

#include "cute/tensor.hpp"

#include "kernel_traits.h"
#include "seq_len.h"
#include <cutlass/array.h>
#include <cutlass/cutlass.h>

using Ktraits = Flash_fwd_kernel_traits<64, 128, 128, 12, 2, false, 1, cutlass::bfloat16_t, false, 4>;
using Element = typename Ktraits::Element;
using TileShape_MNK = typename Ktraits::TileShape_MNK;
using ClusterShape = typename Ktraits::ClusterShape_MNK;
using Seqlen_traits_Q = flash::SeqLenTraits<false, false, true>;
using Seqlen_traits = flash::SeqLenTraits<false, false, false>;


struct Arguments2D {
    int height;
    int width;
    int window_size_top;
    int window_size_bottom;
};

struct MainloopParams {
    int seqlen_q;
    int seqlen_k;
    int window_size_left;
    int window_size_right;
    cutlass::FastDivmod num_splits_divmod;
    Arguments2D args_2d;
};

static constexpr bool Is_split = true;
static constexpr bool Is_local = true;

CUTLASS_HOST_DEVICE
void get_n_block_min_max(
    MainloopParams const& mainloop_params,
    int m_block,
    int n_split_idx,
    int& n_block_min,
    int& n_block_max
) {
    static constexpr int kBlockN = get<1>(TileShape_MNK{});
    static constexpr int kBlockM_div_H = get<0>(TileShape_MNK{}) / Ktraits::kBlockH;
    int const seqlen_k = mainloop_params.seqlen_k;
    n_block_max = cute::ceil_div(seqlen_k, kBlockN);

    if constexpr (Is_split) {
        int const n_blocks_per_split = mainloop_params.num_splits_divmod.divide(
            n_block_max + int(mainloop_params.num_splits_divmod) - 1);
        n_block_min = n_split_idx * n_blocks_per_split;
        n_block_max =
            std::min(n_block_max, (n_split_idx + 1) * n_blocks_per_split);
    }
    if constexpr (Is_local) {
        // Convert 1D indices to 2D coordinates
        const int height = mainloop_params.args_2d.height;
        const int width = mainloop_params.args_2d.width;

        // Get query block's 2D position
        const int q_block_idx = m_block * kBlockM_div_H;
        const int q_row = q_block_idx / width;
        const int q_col = q_block_idx % width;

        // Calculate valid key block range based on 2D window
        const int min_valid_row =
            std::max(0, q_row - mainloop_params.args_2d.window_size_top);
        const int max_valid_row = std::min(
            height - 1, q_row + mainloop_params.args_2d.window_size_bottom);
        const int min_valid_col =
            std::max(0, q_col - mainloop_params.window_size_left);
        const int max_valid_col =
            std::min(width - 1, q_col + mainloop_params.window_size_right);

        // Convert 2D ranges back to block indices
        const int min_valid_idx =
            (min_valid_row * width + min_valid_col) / kBlockN;
        const int max_valid_idx =
            ((max_valid_row * width + max_valid_col) / kBlockN) + 1;

        n_block_min = std::max(n_block_min, min_valid_idx);
        n_block_max = std::min(n_block_max, max_valid_idx);
    }
}
