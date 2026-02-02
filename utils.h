#pragma once

#include <float.h>
#include <windows.h>

#include "curand_kernel.h"

#include "stdio.h"

#define CHECK(call) { \
const cudaError_t error = call; \
if (error != cudaSuccess) { \
printf("Error: %s:%d, ", __FILE__, __LINE__); \
printf("code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
exit(1); \
} \
}

__host__ __forceinline__ void print(const unsigned char * const board, const unsigned int width, const unsigned int height) {
    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; i++) {
            if (board[i + j * width] == 1)
                printf("*");
            else
                printf("#");
        }
        printf("\n");
    }
    printf("\n\n\n");
}

inline int compare(void const* a, void const* b) {

    const double _a = *(double const *) a;
    const double _b = *(double const *) b;

    if (_a < _b) return -1;
    if (_a > _b) return 1;
    return 0;
}

__host__ __forceinline__ double median(double* array, int size) {
    qsort(array, size, sizeof(double), compare);

    if (size % 2 == 0) {
        return (array[size / 2 - 1] + array[size / 2]) / 2.0;
    }
    return array[size / 2];
}

__host__ __forceinline__ void swap_pointer(void **ptr1, void **ptr2) {
    void *tmp = *ptr1;
    *ptr1 = *ptr2;
    *ptr2 = tmp;
}

__host__ __forceinline__ void set_default_pattern(unsigned char * const board, const unsigned int width, const unsigned int height) {

    const unsigned int size = width * height * sizeof(unsigned char);

    memset(board, 0, size);
    board[2] = 1;
    board[2 + width] = 1;
    board[2 + 2 * width] = 1;
    board[1 + 2 * width] = 1;
    board[width] = 1;

}

__host__ __forceinline__ double* launch_sequential(const unsigned char * r_board, unsigned char *w_board, const unsigned int width, const unsigned int height, const unsigned int generations) {

    LARGE_INTEGER freq, start, end;
    QueryPerformanceFrequency(&freq);
    double* time = (double *) malloc(generations * sizeof(double));

    for (unsigned int g = 0; g < generations; g++) {

        QueryPerformanceCounter(&start);

        for (unsigned int j = 0; j < height; j++) {
            for (unsigned int i = 0; i < width; i++) {
                unsigned int n = 0;
                for (int l = -1; l < 2; l++) {
                    for (int k = -1; k < 2; k++) {
                        if (k == 0 && l == 0) continue;
                        const unsigned int x = (i + k + width) % width;
                        const unsigned int y = (j + l + height) % height;
                        n += r_board[x + y * width];
                    }
                }
                const unsigned char is_alive = r_board[i + j * width];
                w_board[i + j * width] = is_alive == 1 && n > 1 && n < 4 ||
                               is_alive == 0 && n == 3;
            }
        }
        swap_pointer((void**) r_board, (void**) w_board);

        QueryPerformanceCounter(&end);
        time[g] = (double) (end.QuadPart - start.QuadPart) * 1000 / (double) freq.QuadPart;
    }
    if (generations % 2 == 1) swap_pointer((void**) r_board, (void**) w_board);
    return time;
}