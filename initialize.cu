#include "curand_kernel.h"
#include "utils.h"

__global__ void initialize(unsigned char * const board, const int seed, const unsigned int width, const unsigned int height) {

    const unsigned int gx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int gy = blockIdx.y * blockDim.y + threadIdx.y;

    if (gx > width - 1 || gy >= height - 1) return;

    const unsigned int linear_index = gx + gy * width;
    curandState state;
    curand_init(seed, linear_index, 0, &state);
    board[linear_index] = curand_uniform(&state) > 0.5;
}

__host__ void launch_initialize(unsigned char * const result, const int seed, const unsigned int width, const unsigned int height) {

    const unsigned int size = width * height * sizeof(unsigned char);
    unsigned char * d_init;
    CHECK(cudaMalloc(&d_init, size));
    initialize<<<dim3((width + 31) / 32, (height + 31) / 32), dim3(32, 32)>>>(d_init, seed, width, height);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(result, d_init, size, cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_init));

}