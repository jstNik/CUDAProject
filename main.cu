#include "utils.h"

#define W 64
#define H 64
#define GEN 100

void launch_initialize(unsigned char *const result, const int seed, const unsigned int width,
                       const unsigned int height);

double *launch_naive(const unsigned char *const initial_board, unsigned char *const result, const dim3 block_size,
                     const unsigned int tile_dim, const unsigned int width, const unsigned int height,
                     const unsigned int generations);

double *launch_smem(const unsigned char *const initial_board, unsigned char *const result,
                    const unsigned int width, const unsigned int height,
                    const unsigned int generations);

double *launch_lin_smem(const unsigned char *const initial_board, unsigned char *const result,
                        const unsigned int width, const unsigned int height,
                        const unsigned int generations);

double* launch_pad_smem(
    const unsigned char *const initial_board,
    unsigned char *const result,
    const unsigned int width,
    const unsigned int height,
    const unsigned int generations
);


int main() {

    const unsigned int size = W * H * sizeof(unsigned char);
    unsigned char * const initial_board = (unsigned char*) malloc(size);
    unsigned char * const result = (unsigned char*) malloc(size);
    unsigned char * const naive_result = (unsigned char*) malloc(size);
    set_default_pattern(initial_board, W, H);
    // print(initial_board, W, H);

    double *elapsed = launch_naive(initial_board, naive_result, dim3(32, 32), 32, W, H, GEN);
    printf("%.3f\n", median(elapsed, GEN));
    free(elapsed);
    // print(result, W, H);

    elapsed = launch_naive(initial_board, result, dim3(32, 8), 32, W, H, GEN);
    printf("%.3f - %d\n", median(elapsed, GEN), memcmp(result, naive_result, size) == 0);
    free(elapsed);
    // print(result, W, H);

    elapsed = launch_smem(initial_board, result, W, H, GEN);
    printf("%.3f - %d\n", median(elapsed, GEN), memcmp(result, naive_result, size) == 0);
    free(elapsed);

    elapsed = launch_lin_smem(initial_board, result, W, H, GEN);
    printf("%.3f - %d\n", median(elapsed, GEN), memcmp(result, naive_result, size) == 0);
    free(elapsed);

    elapsed = launch_pad_smem(initial_board, result, W, H, GEN);
    printf("%.3f - %d\n", median(elapsed, GEN), memcmp(result, naive_result, size) == 0);
    free(elapsed);

    free(initial_board);
    free(result);
    free(naive_result);
    return 0;
}
