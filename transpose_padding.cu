#pragma once

#include "utils.h"
#include "windows.h"

__global__ void add_padding_3(
    const unsigned char *const r_board,
    unsigned char * const w_board,
    const unsigned int tile_dim,
    const unsigned int width,
    const unsigned int height
    ) {

    const unsigned int gx = threadIdx.x + tile_dim * blockIdx.x;
    const unsigned int gy = threadIdx.y + tile_dim * blockIdx.y;

    if (gx > width - 1 || gy > height - 1) return;

    for (unsigned int k = 0; k < tile_dim; k += blockDim.y) {
        const unsigned int y = gy + k;
        if (y > height - 1) break;
            w_board[gx + 1 + (y + 1) * (width + 2)] = r_board[gx + y * width];
    }
}

__global__ void remove_padding_3(
    const unsigned char *const r_board,
    unsigned char * const w_board,
    const unsigned int tile_dim,
    const unsigned int width,
    const unsigned int height
    ) {

    const unsigned int gx = threadIdx.x + blockDim.x * blockIdx.x;
    const unsigned int gy = threadIdx.y + blockDim.y * blockIdx.y;

    if (gx > width - 1 || gy > height - 1) return;

    for (unsigned int k = 0; k < tile_dim; k += blockDim.y) {
        const unsigned int y = gy + k;
        if (y > height - 1) break;
        w_board[gx + y * width] = r_board[gx + 1 + (y + 1) * (width + 2)];
    }
}

__global__ void update_padding_3(unsigned char * const board, const unsigned width, const unsigned height){

    const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if(idx > width - 1) return;

    const unsigned char idx_first = idx == 0;
    const unsigned char idx_last = idx == width - 1;
    const unsigned char idx_middle = !idx_first && !idx_last;

    board[idx] = board[
        idx_first * (width * (height - 1) - 2) +
        idx_middle * (width * (height - 2) + idx) +
        idx_last * (width * (height - 2) + 1)
    ];

    board[width * (height - 1) + idx] = board[
        idx_first * (2 * width - 2) +
        idx_middle * (width + idx) +
        idx_last * (width + 1)
    ];

}

__global__ void transpose_3(
    const unsigned char *const r_board,
    unsigned char *const w_board,
    const unsigned int tile_dim,
    const unsigned int width,
    const unsigned int height) {

    __shared__ unsigned char buffer[32][33];

    unsigned int gx = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int gy = threadIdx.y + blockDim.y * blockIdx.y;

    if (gx < width && gy < height) {
        for (unsigned int k = 0; k < tile_dim; k += blockDim.y) {
            const unsigned int y = gy + k;
            if (y > height - 1) break;
            buffer[threadIdx.y + k][threadIdx.x] = r_board[gx + y * width];
        }
    }

    __syncthreads();

    gx = threadIdx.x + blockDim.y * blockIdx.y;
    gy = threadIdx.y + blockDim.x * blockIdx.x;

    if (gx < height && gy < width) {
        for (unsigned int k = 0; k < tile_dim; k += blockDim.y) {
            const unsigned int y = gy + k;
            if (y > width - 1) break;
            w_board[gx + y * height] = buffer[threadIdx.x][threadIdx.y + k];
        }
    }

}

__global__ void compute_gen_3(
    const unsigned char *const r_board,
    unsigned char * const w_board,
    const unsigned int tile_dim,
    const unsigned int width,
    const unsigned int height
    ) {

    // Indici matrice
    const unsigned int gx = threadIdx.x + blockDim.x * blockIdx.x + 1;
    const unsigned int gy = threadIdx.y + blockDim.y * blockIdx.y + 1;

    // Se fuori dal limite, si ritorna
    if (gx < 1 || gy < 1 || gx > width - 2 || gy > height - 2) return;

    for (unsigned int k = 0; k < tile_dim; k += blockDim.y) {

        const unsigned int y = gy + k;
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
        w_board[gx + y * width] = is_alive == 1 && count_alive_cells > 1 && count_alive_cells < 4 ||
                               is_alive == 0 && count_alive_cells == 3;
    }
}

__host__ double launch_transpose_padding(
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

    const unsigned int pad_size = (width + 2) * (height + 2);
    unsigned char *board_a, *board_b, *normal;
    CHECK(cudaMalloc(&board_a, pad_size * sizeof(unsigned char)));
    CHECK(cudaMalloc(&board_b, pad_size * sizeof(unsigned char)));
    CHECK(cudaMalloc(&normal, width * height * sizeof(unsigned char)));
    CHECK(cudaMemcpy(normal, initial_board, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice));

    dim3 grid_size((width + tile_dim - 1) / tile_dim, (height  + tile_dim - 1) / tile_dim);
    dim3 padded_grid_size((width + tile_dim + 1) / tile_dim, (height + tile_dim + 1) / tile_dim);

    add_padding_3<<<grid_size, block_size>>>(normal, board_a, tile_dim, width, height);
    CHECK(cudaDeviceSynchronize());

    for (int i = 0; i < generations; i++) {

        update_padding_3<<<(width + 1025) / 1024, 1024>>>(board_a, width + 2, height + 2);
        CHECK(cudaDeviceSynchronize());

        transpose_3<<<padded_grid_size, block_size>>>(board_a, board_b, tile_dim, width + 2, height + 2);
        CHECK(cudaDeviceSynchronize());

        update_padding_3<<<(height + 1025) / 1024, 1024>>>(board_b, height + 2, width + 2);
        CHECK(cudaDeviceSynchronize());

        transpose_3<<<dim3(padded_grid_size.y, padded_grid_size.x), block_size>>>(
            board_b, board_a, tile_dim, height + 2, width + 2
        );
        CHECK(cudaDeviceSynchronize());

        compute_gen_3<<<padded_grid_size, block_size>>>(board_a, board_b, tile_dim, width + 2, height + 2);
        CHECK(cudaDeviceSynchronize());
        swap_pointer((void**) &board_a, (void**) &board_b);
    }

    remove_padding_3<<<grid_size, block_size>>>(board_a, normal, tile_dim, width, height);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(result, normal, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost));

    CHECK(cudaFree(board_a));
    CHECK(cudaFree(board_b));
    CHECK(cudaFree(normal));

    QueryPerformanceCounter(&end);
    return (double) (end.QuadPart - start.QuadPart) * 1000 / (double) freq.QuadPart;
}