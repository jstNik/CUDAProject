#pragma once

#include "windows.h"
#include "utils.h"

__device__ void update_padding(
    unsigned char *const board,
    const unsigned int x,
    const unsigned int y,
    const unsigned int width,
    const unsigned int height,
    const unsigned char value
) {

    if (x == 1 && y == 1) {
        board[width - 1 + (height - 1) * width] = value;
    } else if (x == width - 2 && y == 1) {
        board[(height - 1) * width] = value;
    } else if (x == 1 && y == height - 2) {
        board[width - 1] = value;
    } else if (x == width - 2 && y == height - 2) {
        board[0] = value;
    }

    if (y == 1) {
        board[x + (height - 1) * width] = value;
    } else if (y == height - 2) {
        board[x] = value;
    }

    if (x == 1) {
        board[width - 1 + y * height] = value;
    } else if (x == width - 2) {
        board[y * height] = value;
    }


}

__global__ void add_padding_1(const unsigned char *const r_board, unsigned char * const w_board, const unsigned int tile_height, const unsigned int width, const unsigned int height) {

    const unsigned int gx = threadIdx.x + blockDim.x * blockIdx.x;
    const unsigned int gy = threadIdx.y + blockDim.y * blockIdx.y;

    if (gx > width - 1 || gy > height - 1) return;

    for (unsigned int i = 0; i < tile_height; i += blockDim.y) {
        const unsigned int y = gy + i;
        if (y > height - 1) break;
        const unsigned char cell_value = r_board[gx + y * width];

        w_board[gx + 1 + (y + 1) * (width + 2)] = cell_value;

        update_padding(w_board, gx + 1, y + 1, width + 2, height + 2, cell_value);
    }
}

__global__ void remove_padding_1(const unsigned char *const r_board, unsigned char * const w_board, const unsigned int tile_height, const unsigned int width, const unsigned int height) {

    const unsigned int gx = threadIdx.x + blockDim.x * blockIdx.x;
    const unsigned int gy = threadIdx.y + blockDim.y * blockIdx.y;

    if (gx > width - 1 || gy > height - 1) return;

    for (unsigned int i = 0; i < tile_height; i += blockDim.y) {
        const unsigned int y = gy + i;
        if (y > height - 1) break;
        w_board[gx + gy * width] = r_board[gx + 1 + (gy + 1) * (width + 2)];
    }
}

__global__ void compute_gen_1(const unsigned char *const r_board, unsigned char * const w_board, const unsigned int tile_height, const unsigned int width, const unsigned int height) {

    // Indici matrice
    const unsigned int gx = threadIdx.x + blockDim.x * blockIdx.x + 1;
    const unsigned int gy = threadIdx.y + blockDim.y * blockIdx.y + 1;

    // Se fuori dal limite, si ritorna
    if (gx < 1 || gy < 1 || gx > width - 2 || gy > height - 2) return;

    for (unsigned int i = 0; i < tile_height; i += blockDim.y) {
        const unsigned int y = gy + i;
        if (y > height - 2) break;

        const unsigned char n1 = r_board[gx - 1 + (y - 1) * width];
        const unsigned char n2 = r_board[gx + (y - 1) * width];
        const unsigned char n3 = r_board[gx + 1 + (y - 1) * width];

        const unsigned char n4 = r_board[gx - 1 + y * width];
        const int is_alive = r_board[gx + y * width];
        const unsigned char n5 = r_board[gx + 1 + y * width];

        const unsigned char n6 = r_board[gx - 1 + (y + 1) * width];
        const unsigned char n7 = r_board[gx + (y + 1) * width];
        const unsigned char n8 = r_board[gx + 1 + (y + 1) * width];

        const unsigned int count_alive_cells = n1 + n2 + n3 + n4 + n5 + n6 + n7 + n8;

        // Si applicano le regole del gioco
        const unsigned char result = is_alive == 1 && count_alive_cells > 1 && count_alive_cells < 4 ||
                               is_alive == 0 && count_alive_cells == 3;

        w_board[gx + y * width] = result;

        update_padding(w_board, gx, y, width, height, result);
    }
}

__host__ double launch_padding(
    const unsigned char * const initial_board,
    unsigned char * const result,
    const dim3 block_size,
    const unsigned int tile_dim,
    const unsigned int width,
    const unsigned int height,
    const unsigned int generations
) {

    LARGE_INTEGER freq, start, end;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&start);

    const unsigned int pad_size = (width + 2) * (height + 2) * sizeof(unsigned char);
    unsigned char *board_a, *board_b, *normal;
    CHECK(cudaMalloc(&board_a, pad_size));
    CHECK(cudaMalloc(&board_b, pad_size));
    CHECK(cudaMalloc(&normal, width * height * sizeof(unsigned char)));
    CHECK(cudaMemcpy(normal, initial_board, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice));

    const dim3 grid_size((width + tile_dim - 1) / tile_dim, (height + tile_dim - 1) / tile_dim);

    add_padding_1<<<grid_size, block_size>>>(normal, board_a, tile_dim, width, height);
    CHECK(cudaDeviceSynchronize());

    for (int g = 0; g < generations; g++) {
        compute_gen_1<<<grid_size, block_size>>>(board_a, board_b, tile_dim, width + 2, height + 2);
        CHECK(cudaDeviceSynchronize());
        swap_pointer((void**) &board_a, (void**) &board_b);
    }

    remove_padding_1<<<grid_size, block_size>>>(board_a, normal, tile_dim, width, height);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(result, normal, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost));

    CHECK(cudaFree(board_a));
    CHECK(cudaFree(board_b));
    CHECK(cudaFree(normal));

    QueryPerformanceCounter(&end);
    return (double) (end.QuadPart - start.QuadPart) * 1000 / (double) freq.QuadPart;
}