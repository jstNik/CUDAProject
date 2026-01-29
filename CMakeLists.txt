cmake_minimum_required(VERSION 3.29)
project(CUDAProject CUDA)

set(CMAKE_CUDA_STANDARD 26)

add_executable(CUDAProject main.cu
        utils.h
        naive.cu
        smem.cu
        lin_smem.cu
        pad_smem.cu
        initialize.cu
)

set_target_properties(CUDAProject PROPERTIES
                      CUDA_SEPARABLE_COMPILATION ON)
