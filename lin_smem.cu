#pragma once

#include "windows.h"
#include "utils.h"
#define TILE_DIM 32

__global__ void lin_smem(const unsigned char * const r_board, unsigned char * const w_board, const unsigned int tile_dim, const unsigned int width, const unsigned int height) {
    // Indici matrice
    const unsigned int gx = threadIdx.x + blockDim.x * blockIdx.x;
    const unsigned int gy = threadIdx.y + tile_dim * blockIdx.y;

    __shared__ unsigned char smem[TILE_DIM + 2][TILE_DIM + 2];

    // Se fuori dal limite, si ritorna

    constexpr unsigned int p_tile_dim = (TILE_DIM + 2) * (TILE_DIM + 2);
    const unsigned int steps = (p_tile_dim + blockDim.x * blockDim.y - 1) / (blockDim.x * blockDim.y) * blockDim.y;

    for (unsigned int i = 0; i < steps; i += blockDim.y) {
        const unsigned int l_idx = threadIdx.x + blockDim.x * (threadIdx.y + i);

        const unsigned int w = width < TILE_DIM ? width + 2 : TILE_DIM + 2;

        const unsigned int ly = l_idx / w;
        const unsigned int lx = l_idx - ly * w;
        const unsigned int gx_to_load = (blockIdx.x * blockDim.x + lx - 1 + width) % width;
        const unsigned int gy_to_load = (blockIdx.y * TILE_DIM + ly - 1 + height) % height;
        const unsigned int g_to_load = gx_to_load + gy_to_load * width;

        if (ly < TILE_DIM + 2 && lx < TILE_DIM + 2 && g_to_load < width * height) {
            smem[ly][lx] = r_board[g_to_load];
        }
    }

    __syncthreads();

    for (unsigned int i = 0; i < TILE_DIM; i += blockDim.y) {

        const unsigned int thread_y = threadIdx.y + i;

        if (thread_y < TILE_DIM + 2 && gx < width && gy < height) {
            const unsigned char n1 = smem[thread_y][threadIdx.x];
            const unsigned char n2 = smem[thread_y][threadIdx.x + 1];
            const unsigned char n3 = smem[thread_y][threadIdx.x + 2];

            const unsigned char n4 = smem[thread_y + 1][threadIdx.x];
            const unsigned char is_alive = smem[thread_y + 1][threadIdx.x + 1];
            const unsigned char n5 = smem[thread_y + 1][threadIdx.x + 2];

            const unsigned char n6 = smem[thread_y + 2][threadIdx.x];
            const unsigned char n7 = smem[thread_y + 2][threadIdx.x + 1];
            const unsigned char n8 = smem[thread_y + 2][threadIdx.x + 2];

            const unsigned int count_alive_cells = n1 + n2 + n3 + n4 + n5 + n6 + n7 + n8;

            // Si applicano le regole del gioco
            w_board[gx + (gy + i) * width] = is_alive == 1 && count_alive_cells == 2 || count_alive_cells == 3;
        }
    }

}

__host__ double* launch_lin_smem(
    const unsigned char * initial_board,
    unsigned char * result,
    const unsigned int width,
    const unsigned int height,
    const unsigned int generations
) {

    LARGE_INTEGER freq, start, end;
    QueryPerformanceFrequency(&freq);

    const unsigned int size = width * height * sizeof(unsigned char);
    unsigned char *board_a, *board_b;
    CHECK(cudaMalloc(&board_a, size));
    CHECK(cudaMalloc(&board_b, size));
    CHECK(cudaMemcpy(board_a, initial_board, size, cudaMemcpyHostToDevice));

    // const dim3 grid_size((width + tile_dim - 1) / tile_dim, (height + tile_dim - 1) / tile_dim);

    dim3 block_size = dim3(32, 8);
    dim3 grid_size = dim3((width + TILE_DIM - 1) / TILE_DIM, (height + TILE_DIM - 1) / TILE_DIM);

    double* time = (double *) malloc(generations * sizeof(double));

    for (int i = 0; i < generations; i++) {

        QueryPerformanceCounter(&start);

        lin_smem<<<grid_size, block_size>>>(board_a, board_b, TILE_DIM, width, height);
        CHECK(cudaDeviceSynchronize());
        swap_pointer((void**) &board_a, (void**) &board_b);

        QueryPerformanceCounter(&end);
        time[i] = (double) (end.QuadPart - start.QuadPart) * 1000 / (double) freq.QuadPart;

    }

    if (generations % 2 == 1) swap_pointer((void**) initial_board, (void**) result);

    CHECK(cudaMemcpy(result, board_a, size, cudaMemcpyDeviceToHost));
    CHECK(cudaFree(board_a));
    CHECK(cudaFree(board_b));

    return time;
}