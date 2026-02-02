#pragma once

#include "windows.h"
#include "utils.h"
#define TILE_DIM 32

__global__ void smem(const unsigned char * const r_board, unsigned char * const w_board, const unsigned int width, const unsigned int height) {
    // Indici matrice
    const unsigned int gx = threadIdx.x + blockDim.x * blockIdx.x;
    const unsigned int gy = threadIdx.y + TILE_DIM * blockIdx.y;

    __shared__ unsigned char smem[34][34];

    // Se fuori dal limite, si ritorna
    if (gx >= width || gy >= height) return;

    unsigned int steps = (TILE_DIM + 2 + blockDim.y - 1) / blockDim.y * blockDim.y;

    for (unsigned int i = 0; i < steps; i += blockDim.y) {

        const unsigned int y = gy + i;
        if (y >= height) break;

        const unsigned int thread_y = threadIdx.y + i;
        if (thread_y >= TILE_DIM) break;

        const unsigned int toroid_x_plus_1 = (gx + 1) % width;
        const unsigned int toroid_x_minus_1 = (gx - 1 + width) % width;
        const unsigned int toroid_y_plus_1 = (y + 1) % height * width;
        const unsigned int toroid_y_minus_1 = (y - 1 + height) % height * width;

        if (threadIdx.x == 0 && thread_y == 0)
            smem[0][0] = r_board[toroid_x_minus_1 + toroid_y_minus_1];
        if (threadIdx.x == 0 && (thread_y == TILE_DIM - 1 || y == height - 1))
            smem[33][0] = r_board[toroid_x_minus_1 + toroid_y_plus_1];
        if (threadIdx.x == blockDim.x - 1 && thread_y == 0)
            smem[0][33] = r_board[toroid_x_plus_1 + toroid_y_minus_1];
        if (threadIdx.x == blockDim.x - 1 && (thread_y == TILE_DIM - 1 || y == height - 1))
            smem[33][33] = r_board[toroid_x_plus_1 + toroid_y_plus_1];

        if (threadIdx.x == 0)
            smem[thread_y + 1][0] = r_board[toroid_x_minus_1 + y * width];
        if (threadIdx.x == blockDim.x - 1)
            smem[thread_y + 1][33] = r_board[toroid_x_plus_1 + y * width];
        if (thread_y == 0)
            smem[0][threadIdx.x + 1] = r_board[gx + toroid_y_minus_1];
        if (thread_y == TILE_DIM - 1 || y == height - 1)
            smem[33][threadIdx.x + 1] = r_board[gx + toroid_y_plus_1];

        smem[thread_y + 1][threadIdx.x + 1] = r_board[gx + y * width];
    }

    __syncthreads();

    steps = (TILE_DIM + blockDim.y - 1) / blockDim.y * blockDim.y;

    for (unsigned int i = 0; i < steps; i += blockDim.y) {

        const unsigned int thread_y = threadIdx.y + i;
        if (thread_y >= TILE_DIM) break;
        const unsigned int y = gy + i;

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
        w_board[gx + y * width] = is_alive == 1 && count_alive_cells == 2 || count_alive_cells == 3;
    }

}

__host__ double* launch_smem(
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

        smem<<<grid_size, block_size>>>(board_a, board_b, width, height);
        CHECK(cudaDeviceSynchronize());
        swap_pointer((void**) &board_a, (void**) &board_b);

        QueryPerformanceCounter(&end);
        time[i] = (double) (end.QuadPart - start.QuadPart) * 1000 / (double) freq.QuadPart;
    }

    if (generations % 2 == 1) swap_pointer((void**) initial_board, (void**) result);

    CHECK(cudaMemcpy(result, board_a, size, cudaMemcpyDeviceToHost));
    CHECK(cudaFree(board_a));
    CHECK(cudaFree(board_b));
    QueryPerformanceCounter(&end);
    return time;
}