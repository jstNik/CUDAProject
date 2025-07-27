cmake_minimum_required(VERSION 3.29)
project(CUDAProject CUDA)

set(CMAKE_CUDA_STANDARD 26)

add_executable(CUDAProject main.cu
        naive.cu
        utils.h
        initialize.cu
        pad.cu
        transpose_padding.cu)

set_target_properties(CUDAProject PROPERTIES
                      CUDA_SEPARABLE_COMPILATION ON)
