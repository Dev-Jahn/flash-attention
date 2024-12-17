import os
from pathlib import Path
from setuptools import find_packages, setup
from setuptools.command.bdist_wheel import bdist_wheel

import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


# if os.system("which ccache") == 0:
#     os.environ["CXX"] = "ccache"
#     os.environ["CUDA_NVCC_EXECUTABLE"] = "ccache"

"""
debug_mode
Only instantiate the templates for,

- bfloat16,
- head dim 64
- Is_causal = false
- Is_local = true
- gqa 4

Set all debug flags for host and device code.
"""
debug_mode = os.environ.get("FLASH_DEBUG", "0") == "1"


class bdist_wheel_abi3(bdist_wheel):
    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = False
        self.plat_name_supplied = True


torch_path = os.path.dirname(torch.__file__)
curdir = os.path.dirname(os.path.abspath(__file__))
repo_dir = Path(curdir).parent
cutlass_dir = repo_dir / "csrc" / "cutlass"
if debug_mode:
    sources = [
        "flash_api.cpp",
        "flash_fwd_hdim64_bf16_gqa4_sm90.cu",
        "flash_bwd_hdim64_bf16_sm90.cu",
    ]
else:
    sources = [
        "flash_api.cpp",
        "flash_fwd_hdim64_fp16_sm90.cu",
        "flash_fwd_hdim64_bf16_sm90.cu",
        "flash_fwd_hdim128_fp16_sm90.cu",
        "flash_fwd_hdim128_bf16_sm90.cu",
        "flash_fwd_hdim256_fp16_sm90.cu",
        "flash_fwd_hdim256_bf16_sm90.cu",
        "flash_bwd_hdim64_fp16_sm90.cu",
        "flash_bwd_hdim96_fp16_sm90.cu",
        "flash_bwd_hdim128_fp16_sm90.cu",
        "flash_bwd_hdim64_bf16_sm90.cu",
        "flash_bwd_hdim96_bf16_sm90.cu",
        "flash_bwd_hdim128_bf16_sm90.cu",
        "flash_fwd_hdim64_e4m3_sm90.cu",
        "flash_fwd_hdim128_e4m3_sm90.cu",
        "flash_fwd_hdim256_e4m3_sm90.cu",
        "flash_fwd_hdim64_fp16_gqa2_sm90.cu",
        "flash_fwd_hdim64_fp16_gqa4_sm90.cu",
        "flash_fwd_hdim64_fp16_gqa8_sm90.cu",
        "flash_fwd_hdim64_fp16_gqa16_sm90.cu",
        "flash_fwd_hdim64_fp16_gqa32_sm90.cu",
        "flash_fwd_hdim128_fp16_gqa2_sm90.cu",
        "flash_fwd_hdim128_fp16_gqa4_sm90.cu",
        "flash_fwd_hdim128_fp16_gqa8_sm90.cu",
        "flash_fwd_hdim128_fp16_gqa16_sm90.cu",
        "flash_fwd_hdim128_fp16_gqa32_sm90.cu",
        "flash_fwd_hdim256_fp16_gqa2_sm90.cu",
        "flash_fwd_hdim256_fp16_gqa4_sm90.cu",
        "flash_fwd_hdim256_fp16_gqa8_sm90.cu",
        "flash_fwd_hdim256_fp16_gqa16_sm90.cu",
        "flash_fwd_hdim256_fp16_gqa32_sm90.cu",
        "flash_fwd_hdim64_bf16_gqa2_sm90.cu",
        "flash_fwd_hdim64_bf16_gqa4_sm90.cu",
        "flash_fwd_hdim64_bf16_gqa8_sm90.cu",
        "flash_fwd_hdim64_bf16_gqa16_sm90.cu",
        "flash_fwd_hdim64_bf16_gqa32_sm90.cu",
        "flash_fwd_hdim128_bf16_gqa2_sm90.cu",
        "flash_fwd_hdim128_bf16_gqa4_sm90.cu",
        "flash_fwd_hdim128_bf16_gqa8_sm90.cu",
        "flash_fwd_hdim128_bf16_gqa16_sm90.cu",
        "flash_fwd_hdim128_bf16_gqa32_sm90.cu",
        "flash_fwd_hdim256_bf16_gqa2_sm90.cu",
        "flash_fwd_hdim256_bf16_gqa4_sm90.cu",
        "flash_fwd_hdim256_bf16_gqa8_sm90.cu",
        "flash_fwd_hdim256_bf16_gqa16_sm90.cu",
        "flash_fwd_hdim256_bf16_gqa32_sm90.cu",
        "flash_fwd_hdim64_e4m3_gqa2_sm90.cu",
        "flash_fwd_hdim64_e4m3_gqa4_sm90.cu",
        "flash_fwd_hdim64_e4m3_gqa8_sm90.cu",
        "flash_fwd_hdim64_e4m3_gqa16_sm90.cu",
        "flash_fwd_hdim64_e4m3_gqa32_sm90.cu",
        "flash_fwd_hdim128_e4m3_gqa2_sm90.cu",
        "flash_fwd_hdim128_e4m3_gqa4_sm90.cu",
        "flash_fwd_hdim128_e4m3_gqa8_sm90.cu",
        "flash_fwd_hdim128_e4m3_gqa16_sm90.cu",
        "flash_fwd_hdim128_e4m3_gqa32_sm90.cu",
        "flash_fwd_hdim256_e4m3_gqa2_sm90.cu",
        "flash_fwd_hdim256_e4m3_gqa4_sm90.cu",
        "flash_fwd_hdim256_e4m3_gqa8_sm90.cu",
        "flash_fwd_hdim256_e4m3_gqa16_sm90.cu",
        "flash_fwd_hdim256_e4m3_gqa32_sm90.cu",
    ]


# 기본 cxx 플래그 설정
cxx_flags = ["-std=c++17", "-DFLASHATTENTION_ENABLE_2D"]

# 기본 nvcc 플래그 설정
nvcc_flags = [
    "-std=c++17",
    # "-U__CUDA_NO_HALF_OPERATORS__",
    # "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT16_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT162_OPERATORS__",
    "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "--use_fast_math",
    # "--ptxas-options=--verbose,--register-usage-level=10,--warn-on-local-memory-usage",  # printing out number of registers
    # "-lineinfo",
    "-DCUTLASS_DEBUG_TRACE_LEVEL=0",
    "-DNDEBUG",
    "-DFLASHATTENTION_ENABLE_2D",
    "-gencode",
    "arch=compute_90a,code=sm_90a",
    "--threads",
    "16",
    "-g",  # Host code debug
    "-O0",  # Host code opt level
    "-G",  # Device code debug
    "-DFLASH_DEBUG",
]

# 디버그 모드일 경우 플래그 추가
if debug_mode:
    cxx_flags.extend(["-g", "-O0", "-DFLASH_DEBUG"])
    nvcc_flags.extend(
        [
            "-g",  # Host code debug
            "-O0",  # Host code opt level
            "-G",  # Device code debug
            "-DFLASH_DEBUG",
            # "-lineinfo",
            # "--ptxas-options=-O1",
        ]
    )
else:
    cxx_flags.extend(["-O3"])
    nvcc_flags.extend(["-O3"])

setup(
    name="flashattn_2d",
    py_modules=["flashattn_2d.interface"],
    ext_modules=[
        CUDAExtension(
            name="flashattn_2d_hopper_cuda",
            sources=[os.path.join("src", source) for source in sources],
            extra_compile_args={
                "cxx": cxx_flags,
                "nvcc": nvcc_flags,
            },
            include_dirs=[
                cutlass_dir / "include",
            ],
            libraries=["cuda"],
            runtime_library_dirs=[os.path.join(os.path.dirname(torch_path), "lib")],
        )
    ],
    cmdclass={"build_ext": BuildExtension, "bdist_wheel": bdist_wheel_abi3},
)
