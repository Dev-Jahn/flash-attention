cmake_minimum_required(VERSION 3.18)
project(cuda_test CUDA CXX)

# CUDA 컴파일러 설정
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 90a)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_90a,code=sm_90a --expt-relaxed-constexpr")

set(CUTLASS_DIR "${CMAKE_SOURCE_DIR}/../../csrc/cutlass")
include_directories(${CUTLASS_DIR}/include)
include_directories("/usr/local/cuda/include")

# 실행 파일 생성
add_executable(cuda_test
    test.cu
)

target_link_libraries(cuda_test
    cuda
    cudart
)
