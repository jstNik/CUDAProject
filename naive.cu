#pragma once

#include "windows.h"
#include "utils.h"

__global__ void naive_0(const unsigned char * const r_board, unsigned char * const w_board, const unsigned int tile_height, const unsigned int width, const unsigned int height) {
    // Indici matrice
    const unsigned int gx = threadIdx.x + blockDim.x * blockIdx.x;
    const unsigned int gy = threadIdx.y + blockDim.y * blockIdx.y;

    // Se fuori dal limite, si ritorna
    if (gx > width - 1 || gy > height - 1) return;

    for (unsigned int i = 0; i < tile_height; i += blockDim.y) {

        const unsigned int y = gy + tile_height * i;
        if (y > height - 1) break;

        const unsigned char n1 = r_board[(gx - 1 + width) % width + (y - 1 + height) % height * width];
        const unsigned char n2 = r_board[gx + (y - 1 + height) % height * width];
        const unsigned char n3 = r_board[(gx + 1) % width + (y - 1 + height) % height * width];

        const unsigned char n4 = r_board[(gx - 1 + width) % width + y * width];
        const unsigned char is_alive = r_board[gx + y * width];
        const unsigned char n5 = r_board[(gx + 1) % width + y * width];

        const unsigned char n6 = r_board[(gx - 1 + width) % width + (y + 1) % height * width];
        const unsigned char n7 = r_board[gx + (y + 1) % height * width];
        const unsigned char n8 = r_board[(gx + 1) % width + (y + 1) % height * width];

        const unsigned int count_alive_cells = n1 + n2 + n3 + n4 + n5 + n6 + n7 + n8;

        // Si applicano le regole del gioco
        w_board[gx + gy * width] = is_alive == 1 && count_alive_cells > 1 && count_alive_cells < 4 ||
                               is_alive == 0 && count_alive_cells == 3;
    }

}

__host__ double* launch_naive(
    const unsigned char *const initial_board,
    unsigned char *const result,
    const dim3 block_size,
    const unsigned int tile_dim,
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

    const dim3 grid_size((width + tile_dim - 1) / tile_dim, (height + tile_dim - 1) / tile_dim);
    const unsigned int tile_height = (block_size.y + tile_dim - 1) / tile_dim;

    double* time = (double *) malloc(generations * sizeof(double));

    for (int i = 0; i < generations; i++) {

        QueryPerformanceCounter(&start);

        naive_0<<<grid_size, block_size>>>(board_a, board_b, tile_height, width, height);
        CHECK(cudaDeviceSynchronize());
        swap_pointer((void**) &board_a, (void**) &board_b);

        QueryPerformanceCounter(&end);
        time[i] = (double) (end.QuadPart - start.QuadPart) * 1000 / (double) freq.QuadPart;
    }

    CHECK(cudaMemcpy(result, board_a, size, cudaMemcpyDeviceToHost));
    CHECK(cudaFree(board_a));
    CHECK(cudaFree(board_b));
    QueryPerformanceCounter(&end);
    return time;
}