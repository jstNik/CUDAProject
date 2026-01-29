#pragma once

#include <float.h>

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