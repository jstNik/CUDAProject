#include "utils.h"

#define W 16382
#define H 16382
#define GEN 1

void launch_initialize(unsigned char * const result, const int seed, const unsigned int width, const unsigned int height);
double launch_naive(const unsigned char * const initial_board, unsigned char * const result, const unsigned int width, const unsigned int height, const unsigned int generations);
double launch_padding(const unsigned char * const initial_board, unsigned char * const result, const dim3 block_size, const unsigned int tile_dim, const unsigned int width, const unsigned int height, const unsigned int generations);
double launch_transpose_padding(const unsigned char * const initial_board, unsigned char * const result, const dim3 block_size, const unsigned int tile_dim, const unsigned int width, const unsigned int height, const unsigned int generations);


int main() {

    const unsigned int size = W * H * sizeof(unsigned char);
    unsigned char * const initial_board = (unsigned char*) malloc(size);
    unsigned char * const result = (unsigned char*) malloc(size);

    launch_initialize(initial_board, time(NULL), W, H);
    // set_default_pattern(initial_board, W, H);
    // print(initial_board, W, H);

    double elapsed = launch_naive(initial_board, result, W, H, 1);
    printf("%.3f\n", elapsed);
    // print(result, W, H);

    elapsed = launch_padding(initial_board, result, dim3(32, 32), 32, W, H, 1);
    printf("%.3f\n", elapsed);
    // print(result, W, H);

    elapsed = launch_padding(initial_board, result, dim3(32, 8), 32, W, H, 1);
    printf("%.3f\n", elapsed);

    elapsed = launch_transpose_padding(initial_board, result, dim3(32, 32), 32, W, H, 1);
    printf("%.3f\n", elapsed);
    // print(result, W, H);

    elapsed = launch_transpose_padding(initial_board, result, dim3(32, 8), 32, W, H, 1);
    printf("%.3f\n", elapsed);

    free(initial_board);
    free(result);
    return 0;
}
