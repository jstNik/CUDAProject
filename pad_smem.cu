#pragma once

#include "windows.h"
#include "utils.h"
#define TILE_DIM 32

__global__ void padding(
    unsigned char * const board,
    const unsigned int width,
    const unsigned int height
) {

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        board[0] = board[width * (height - 1) - 2];
        board[width - 1] = board[width * (height - 2) + 1];
        board[width * (height - 1)] = board[2 * width - 2];
        board[width * height - 1] = board[width + 1];
    }

    const unsigned int steps = (TILE_DIM + blockDim.x - 1) / blockDim.x * blockDim.x;

    for (unsigned int i = 0; i < steps; i += blockDim.x) {
        const unsigned int tx = threadIdx.x + i + blockIdx.x * TILE_DIM;

        if (tx > 0 && tx < width - 1) {
            board[tx] = board[tx + width * (height - 2)];
            board[width * (height - 1) + tx] = board[tx + width];
        }

        if (tx > 0 && tx < height - 1) {
            board[tx * width] = board[(tx + 1) * width - 2];
            board[(tx + 1) * width - 1] = board[tx * width + 1];
        }
    }
}

__global__ void pad_smem(
    unsigned char *const r_board,
    unsigned char * w_board,
    const unsigned int width,
    const unsigned int height
) {

    __shared__ unsigned char smem[TILE_DIM + 2][TILE_DIM + 2];

    // Se fuori dal limite, si ritorna
    // if (gx > width - 2 || gy > height - 2) return;

    constexpr unsigned int p_tile_dim = (TILE_DIM + 2) * (TILE_DIM + 2);

    const unsigned int steps = (p_tile_dim + blockDim.x * blockDim.y - 1) / (blockDim.x * blockDim.y) * blockDim.y;

    for (unsigned int i = 0; i < steps; i += blockDim.y) {

        const unsigned int lx = threadIdx.x + blockDim.x * blockIdx.x;
        const unsigned int ly = threadIdx.y + i + TILE_DIM * blockIdx.y;
        const unsigned int l_idx = lx + ly * width;


        const unsigned int sy = threadIdx.y + i;
        const unsigned int sx = threadIdx.x;

        if (sx < width && sy < height && sy < TILE_DIM + 2 && sx < TILE_DIM + 2) {
            smem[sy][sx] = r_board[l_idx];
        }
    }

    __syncthreads();

    for (unsigned int i = 0; i < TILE_DIM; i += blockDim.y) {

        const unsigned int thread_y = threadIdx.y + i;
        const unsigned int thread_x = threadIdx.x;
        const unsigned int x = thread_x + 1 + blockDim.x * blockIdx.x;
        const unsigned int y = thread_y + 1 + TILE_DIM * blockIdx.y;

        if (x >= width - 1 || y >= height - 1) break;

        const unsigned int lg = x + y * width;


        const unsigned char n1 = smem[thread_y][thread_x];
        const unsigned char n2 = smem[thread_y][thread_x + 1];
        const unsigned char n3 = smem[thread_y][thread_x + 2];

        const unsigned char n4 = smem[thread_y + 1][thread_x];
        const unsigned char is_alive = smem[thread_y + 1][thread_x + 1];
        const unsigned char n5 = smem[thread_y + 1][thread_x + 2];

        const unsigned char n6 = smem[thread_y + 2][thread_x];
        const unsigned char n7 = smem[thread_y + 2][thread_x + 1];
        const unsigned char n8 = smem[thread_y + 2][thread_x + 2];

        const unsigned int count_alive_cells = n1 + n2 + n3 + n4 + n5 + n6 + n7 + n8;

        // Si applicano le regole del gioco
        w_board[lg] = is_alive == 1 && count_alive_cells > 1 && count_alive_cells < 4 ||
            is_alive == 0 && count_alive_cells == 3;
    }

}

__host__ double* launch_pad_smem(
    const unsigned char *const initial_board,
    unsigned char *const result,
    const unsigned int width,
    const unsigned int height,
    const unsigned int generations
) {

    LARGE_INTEGER freq, start, end;
    QueryPerformanceFrequency(&freq);

    const unsigned int size = (width + 2) * (height + 2) * sizeof(unsigned char);
    unsigned char *board_a, *board_b;
    CHECK(cudaMalloc(&board_a, size));
    CHECK(cudaMalloc(&board_b, size));
    CHECK(cudaMemcpy2D(
        &board_a[width + 3],
        (width + 2) * sizeof(unsigned char),
        initial_board,
        width * sizeof(unsigned char),
        width * sizeof(unsigned char),
        height,
        cudaMemcpyHostToDevice
    ));

    dim3 block_size = dim3(32, 8);
    dim3 grid_size = dim3((width + TILE_DIM - 1) / TILE_DIM, (height + TILE_DIM - 1) / TILE_DIM);
    dim3 p_block_size = dim3(256);
    dim3 p_grid_size = dim3((max(width, height) + p_block_size.x - 1) / p_block_size.x);


    double* time = (double *) malloc(generations * sizeof(double));

    for (int i = 0; i < generations; i++) {

        QueryPerformanceCounter(&start);

        padding<<<p_grid_size, p_block_size>>>(
            board_a,
            width + 2,
            height + 2
        );
        CHECK(cudaDeviceSynchronize());
        pad_smem<<<grid_size, block_size>>>(
            board_a,
            board_b,
            width + 2,
            height + 2
        );
        CHECK(cudaDeviceSynchronize());
        swap_pointer((void**) &board_a, (void**) &board_b);

        QueryPerformanceCounter(&end);
        time[i] = (double) (end.QuadPart - start.QuadPart) * 1000 / (double) freq.QuadPart;
    }

    if (generations % 2 == 1) swap_pointer((void**) initial_board, (void**) result);

    CHECK(cudaMemcpy2D(
        result,
        width * sizeof(unsigned char),
        &board_a[width + 3],
        (width + 2) * sizeof(unsigned char),
        width * sizeof(unsigned char),
        height,
        cudaMemcpyDeviceToHost
        ));
    CHECK(cudaFree(board_a));
    CHECK(cudaFree(board_b));
    return time;
}