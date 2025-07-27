#pragma once

#include "windows.h"
#include "utils.h"

__global__ void naive_0(const unsigned char * const r_board, unsigned char * const w_board, const unsigned int width, const unsigned int height) {
    // Indici matrice
    const unsigned int gx = threadIdx.x + blockDim.x * blockIdx.x;
    const unsigned int gy = threadIdx.y + blockDim.y * blockIdx.y;

    // Se fuori dal limite, si ritorna
    if (gx > width - 1 || gy > height - 1) return;

    const unsigned char n1 = r_board[(gx - 1 + width) % width + (gy - 1 + height) % height * width];
    const unsigned char n2 = r_board[gx + (gy - 1 + height) % height * width];
    const unsigned char n3 = r_board[(gx + 1) % width + (gy - 1 + height) % height * width];

    const unsigned char n4 = r_board[(gx - 1 + width) % width + gy * width];
    const unsigned char is_alive = r_board[gx + gy * width];
    const unsigned char n5 = r_board[(gx + 1) % width + gy * width];

    const unsigned char n6 = r_board[(gx - 1 + width) % width + (gy + 1) % height * width];
    const unsigned char n7 = r_board[gx + (gy + 1) % height * width];
    const unsigned char n8 = r_board[(gx + 1) % width + (gy + 1) % height * width];

    const unsigned int count_alive_cells = n1 + n2 + n3 + n4 + n5 + n6 + n7 + n8;

    // Si applicano le regole del gioco
    w_board[gx + gy * width] = is_alive == 1 && count_alive_cells > 1 && count_alive_cells < 4 ||
                           is_alive == 0 && count_alive_cells == 3;

}

__host__ double launch_naive(const unsigned char * const initial_board, unsigned char * const result, const unsigned int width, const unsigned int height, const unsigned int generations) {

    LARGE_INTEGER freq, start, end;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&start);

    const unsigned int size = width * height * sizeof(unsigned char);
    unsigned char *board_a, *board_b;
    CHECK(cudaMalloc(&board_a, size));
    CHECK(cudaMalloc(&board_b, size));
    CHECK(cudaMemcpy(board_a, initial_board, size, cudaMemcpyHostToDevice));

    for (int i = 0; i < generations; i++) {
        naive_0<<<dim3((width + 31) / 32, (height + 31) / 32), dim3(32, 32)>>>(board_a, board_b, width, height);
        CHECK(cudaDeviceSynchronize());
        swap_pointer((void**) &board_a, (void**) &board_b);
    }

    CHECK(cudaMemcpy(result, board_a, size, cudaMemcpyDeviceToHost));
    CHECK(cudaFree(board_a));
    CHECK(cudaFree(board_b));
    QueryPerformanceCounter(&end);
    return (double) (end.QuadPart - start.QuadPart) * 1000 / (double) freq.QuadPart;
}