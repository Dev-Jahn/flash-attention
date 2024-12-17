#include "mainloop.hpp"
#include <iostream>

int main() {
    MainloopParams params = {
        .seqlen_q = 25,
        .seqlen_k = 25,
        .window_size_left = 1,
        .window_size_right = 1,
        .num_splits_divmod = cutlass::FastDivmod(1),
        .args_2d = {
            .height = 5,
            .width = 5,
            .window_size_top = 1,
            .window_size_bottom = 1,
        },
    };
    int m_block = 0;
    int n_split_idx = 0;
    int n_block_min = 0;
    int n_block_max = 0;

    // 함수 호출
    get_n_block_min_max(params, m_block, n_split_idx, n_block_min, n_block_max);

    // 결과 출력
    std::cout << "n_block_min: " << n_block_min << std::endl;
    std::cout << "n_block_max: " << n_block_max << std::endl;

    return 0;
}