#include <curand.h>
#include <curand_kernel.h>
#include <windows.h>
#include <filesystem>
#define W 10000
#define H 10000
#define GEN 100

#define CHECK(call) { \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        printf("Error: %s:%d, ", __FILE__, __LINE__); \
        printf("code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1); \
    } \
}


__host__ void print(const unsigned char *board) {
    for (int j = 0; j < H; j++) {
        for (int i = 0; i < W; i++) {
            if (board[i + j * W] == 1)
                printf("*");
            else
                printf("#");
        }
        printf("\n");
    }
    printf("\n\n\n");
}

__host__ void getSurfaceObject(cudaSurfaceObject_t *surfObj, const cudaArray_t *array) {
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = *array;
    CHECK(cudaCreateSurfaceObject(surfObj, &resDesc));
}

__host__ void swap_pointer(void **ptr1, void **ptr2) {
    void *tmp = *ptr1;
    *ptr1 = *ptr2;
    *ptr2 = tmp;
}

__host__ void play_cpu(unsigned char *board, const int generations) {

    unsigned char* next_gen = (unsigned char*) malloc(W * H * sizeof(unsigned char));

    for (int g = 0; g < generations; g++) {
        for (int j = 0; j < H; j++) {
            for (int i = 0; i < W; i++) {
                const bool is_alive = board[i + j * W];
                unsigned int neighbors = 0;
                unsigned int y = ((j - 1) % H + H) % H * W;
                neighbors += board[((i - 1) % W + W) % W + y];
                neighbors += board[i + y];
                neighbors += board[(i + 1) % W + y];

                y = j * W;
                neighbors += board[((i - 1) % W + W) % W + y];
                neighbors += board[(i + 1) % W + y];

                y = (j + 1) % H * W;
                neighbors += board[((i - 1) % W + W) % W + y];
                neighbors += board[i + y];
                neighbors += board[(i + 1) % W + y];

                next_gen[i + j * W] = is_alive == 1 && neighbors > 1 && neighbors < 4 || is_alive == 0 && neighbors == 3;
            }
        }
        unsigned char *tmp = board;
        board = next_gen;
        next_gen = tmp;
    }
}


__global__ void initialize(unsigned char *board, const int seed) {
    const unsigned int gx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int gy = blockIdx.y * blockDim.y + threadIdx.y;

    if (gx >= W || gy >= H) return;

    const unsigned int linear_index = gx + gy * W;
    curandState state;
    curand_init(seed, linear_index, 0, &state);
    board[linear_index] = curand_uniform(&state) > 0.5;
}

__global__ void play_1_1(const unsigned char *r_board, unsigned char *w_board) {
    // Indici matrice
    const int gx = (int) threadIdx.x + blockDim.x * blockIdx.x;
    const int gy = (int) threadIdx.y + blockDim.y * blockIdx.y;

    // Se fuori dal limite, si ritorna
    if (gx >= W || gy >= H) return;

    const int is_alive = r_board[gx + gy * W];
    int count_alive_cells = 0;

    count_alive_cells += r_board[(gx - 1 + W) % W + (gy - 1 + H) % H * W];
    count_alive_cells += r_board[gx + (gy - 1 + H) % H * W];
    count_alive_cells += r_board[(gx + 1) % W + (gy - 1 + H) % H * W];

    count_alive_cells += r_board[(gx - 1 + W) % W + gy * W];
    count_alive_cells += r_board[(gx + 1) % W + gy * W];

    count_alive_cells += r_board[(gx - 1 + W) % W + (gy + 1) % H * W];
    count_alive_cells += r_board[gx + (gy + 1) % H * W];
    count_alive_cells += r_board[(gx + 1) % W + (gy + 1) % H * W];

    // Si applicano le regole del gioco
    w_board[gx + gy * W] = is_alive == 1 && count_alive_cells > 1 && count_alive_cells < 4 ||
                         is_alive == 0 && count_alive_cells == 3;

}

__global__ void play_1_2(const unsigned char *r_board, unsigned char *w_board) {
    // Indici matrice
    const int gx = (int) threadIdx.x + blockDim.x * blockIdx.x;
    const int gy = (int) threadIdx.y + blockDim.y * blockIdx.y;

    // Se fuori dal limite, si ritorna
    if (gx >= W || gy >= H) return;

    const int is_alive = r_board[gx + gy * W];

    const unsigned char n1 = r_board[(gx - 1 + W) % W + (gy - 1 + H) % H * W];
    const unsigned char n2 = r_board[gx + (gy - 1 + H) % H * W];
    const unsigned char n3 = r_board[(gx + 1) % W + (gy - 1 + H) % H * W];

    const unsigned char n4 = r_board[(gx - 1 + W) % W + gy * W];
    const unsigned char n5 = r_board[(gx + 1) % W + gy * W];

    const unsigned char n6 = r_board[(gx - 1 + W) % W + (gy + 1) % H * W];
    const unsigned char n7 = r_board[gx + (gy + 1) % H * W];
    const unsigned char n8 = r_board[(gx + 1) % W + (gy + 1) % H * W];

    const unsigned int count_alive_cells = n1 + n2 + n3 + n4 + n5 + n6 + n7 + n8;


    // Si applicano le regole del gioco
    w_board[gx + gy * W] = is_alive == 1 && count_alive_cells > 1 && count_alive_cells < 4 ||
                         is_alive == 0 && count_alive_cells == 3;


}

__global__ void play_1_3(const unsigned char *r_board, unsigned char *w_board) {
    // Indici matrice
    const int gx = (int) threadIdx.x + blockDim.x * blockIdx.x;
    const int gy = (int) threadIdx.y + blockDim.y * blockIdx.y;

    // Se fuori dal limite, si ritorna
    if (gx >= W || gy >= H) return;
    const int is_alive = r_board[gx + gy * W];

    const unsigned int count_alive_cells =
        r_board[(gx - 1 + W) % W + (gy - 1 + H) % H * W] +
        r_board[gx % W + (gy - 1 + H) % H * W] +
        r_board[(gx + 1) % W + (gy - 1 + H) % H * W] +
        r_board[(gx - 1 + W) % W + gy * W] +
        r_board[(gx + 1) % W + gy * W] +
        r_board[(gx - 1 + W) % W + (gy + 1) % H * W] +
        r_board[gx + (gy + 1) % H * W] +
        r_board[(gx + 1) % W + (gy + 1) % H * W];

    // Si applicano le regole del gioco
    w_board[gx + gy * W] = is_alive == 1 && count_alive_cells > 1 && count_alive_cells < 4 ||
                         is_alive == 0 && count_alive_cells == 3;

}


__global__ void play_2_1(const unsigned char *r_board, unsigned char *w_board) {

    extern __shared__ unsigned char shared[];

    // Indici matriciali
    int gx = (int) blockIdx.x * (blockDim.x - 2) + threadIdx.x - 1;
    int gy = (int) blockIdx.y * (blockDim.y - 2) + threadIdx.y - 1;

    // Controllo posizione thread
    if (gx < -1 || gx > W || gy < -1 || gy > H) return;

    // Indici toroidali
    gx = (gx + W) % W;
    gy = (gy + H) % H;
    const int g_idx = gx + gy * W;

    // Indice memoria condivisa
    const unsigned int l_idx = threadIdx.y * blockDim.x + threadIdx.x;
        // Se controlla che il thread è nei limiti per evitare accessi illegali in memoria
    shared[l_idx] = r_board[g_idx];

    __syncthreads();

    // Si aggiornano i limiti
    if (threadIdx.x > 0 && threadIdx.x < blockDim.x - 1 && threadIdx.y > 0 && threadIdx.y < blockDim.y - 1) {
        const int is_alive = shared[l_idx];
        unsigned int count_alive_cells = 0;
        // Indice celle vicine
        unsigned int idx = (threadIdx.y - 1) * blockDim.x + threadIdx.x - 1;
        count_alive_cells += shared[idx];
        count_alive_cells += shared[idx + 1];
        count_alive_cells += shared[idx + 2];

        idx = threadIdx.y * blockDim.x + threadIdx.x - 1;
        count_alive_cells += shared[idx];
        count_alive_cells += shared[idx + 2];

        idx = (threadIdx.y + 1) * blockDim.x + threadIdx.x - 1;
        count_alive_cells += shared[idx];
        count_alive_cells += shared[idx + 1];
        count_alive_cells += shared[idx + 2];

        // Si applicano le regole del gioco
        w_board[g_idx] = is_alive == 1 && count_alive_cells > 1 && count_alive_cells < 4 ||
                       is_alive == 0 && count_alive_cells == 3;
    }

}

__global__ void play_2_2(const unsigned char *r_board, unsigned char *w_board) {

    extern __shared__ unsigned char shared[];

    // Indici matriciali
    int gx = (int) blockIdx.x * (blockDim.x - 2) + threadIdx.x - 1;
    int gy = (int) blockIdx.y * (blockDim.y - 2) + threadIdx.y - 1;

    // Controllo posizione thread
    if (gx < -1 || gx > W || gy < -1 || gy > H) return;

    // Indici toroidali
    gx = (gx % W + W) % W;
    gy = (gy % H + H) % H;

    // Indice matrice
    const int g_idx = gx + gy * W;

    // Indice memoria condivisa
    const unsigned int l_idx = threadIdx.y * blockDim.x + threadIdx.x;

    // Se controlla che il thread è nei limiti per evitare accessi illegali in memoria
    shared[l_idx] = r_board[g_idx];

    __syncthreads();

    // Si aggiornano i limiti
    if (threadIdx.x > 0 && threadIdx.x < blockDim.x - 1 && threadIdx.y > 0 && threadIdx.y < blockDim.y - 1) {
        const int is_alive = shared[l_idx];

        // Indice celle vicine
        const bool n1 = shared[(threadIdx.y - 1) * blockDim.x + threadIdx.x - 1];
        const bool n2 = shared[(threadIdx.y - 1) * blockDim.x + threadIdx.x];
        const bool n3 = shared[(threadIdx.y - 1) * blockDim.x + threadIdx.x + 1];

        const bool n4 = shared[threadIdx.y * blockDim.x + threadIdx.x - 1];
        const bool n5 = shared[threadIdx.y * blockDim.x + threadIdx.x + 1];

        const bool n6 = shared[(threadIdx.y + 1) * blockDim.x + threadIdx.x - 1];
        const bool n7 = shared[(threadIdx.y + 1) * blockDim.x + threadIdx.x];
        const bool n8 = shared[(threadIdx.y + 1) * blockDim.x + threadIdx.x + 1];

        const int count_alive_cells = n1 + n2 + n3 + n4 + n5 + n6 + n7 + n8;

        // Si applicano le regole del gioco
        w_board[g_idx] = is_alive == 1 && count_alive_cells > 1 && count_alive_cells < 4 ||
                       is_alive == 0 && count_alive_cells == 3;
    }

}

__global__ void play_3_1(const cudaSurfaceObject_t r_obj, const cudaSurfaceObject_t w_obj) {
    // Indici matrice
    const int gx = (int) threadIdx.x + blockDim.x * blockIdx.x;
    const int gy = (int) threadIdx.y + blockDim.y * blockIdx.y;

    // Se fuori dal limite, si ritorna
    if (gx > W - 1 || gy > H - 1) return;

    int count_alive_cells = 0;

    // Si caricano i dati dalla memoria globale attraverso l'uso della texture memory.
    // Purtroppo il tipo bool non è supportato è quindi necessario usare un altro tipo che abbia dimensioni
    // uguali, cioè un byte. In questo caso abbiamo usato il char.

    const int is_alive = surf2Dread<unsigned char>(r_obj, gx, gy);
    count_alive_cells += surf2Dread<unsigned char>(r_obj, (gx - 1 + W) % W, (gy - 1 + H) % H);
    count_alive_cells += surf2Dread<unsigned char>(r_obj, gx, (gy - 1 + H) % H);
    count_alive_cells += surf2Dread<unsigned char>(r_obj, (gx + 1) % W, (gy - 1 + H) % H);

    count_alive_cells += surf2Dread<unsigned char>(r_obj, (gx - 1 + W) % W, gy);
    count_alive_cells += surf2Dread<unsigned char>(r_obj, (gx + 1) % W, gy);

    count_alive_cells += surf2Dread<unsigned char>(r_obj, (gx - 1 + W) % W, (gy + 1) % H);
    count_alive_cells += surf2Dread<unsigned char>(r_obj, gx, (gy + 1) % H);
    count_alive_cells += surf2Dread<unsigned char>(r_obj, (gx + 1) % W, (gy + 1) % H);

    // Si applicano le regole del gioco
    surf2Dwrite<unsigned char>(
        is_alive == 1 && count_alive_cells > 1 && count_alive_cells < 4 || is_alive == 0 && count_alive_cells == 3,
        w_obj, gx, gy
    );
}

__global__ void play_3_2(const cudaSurfaceObject_t r_obj, const cudaSurfaceObject_t w_obj) {
    // Indici matrice
    const int gx = (int) threadIdx.x + blockDim.x * blockIdx.x;
    const int gy = (int) threadIdx.y + blockDim.y * blockIdx.y;

    // Se fuori dal limite, si ritorna
    if (gx > W - 1 || gy > H - 1) return;


    // Si caricano i dati dalla memoria globale attraverso l'uso della texture memory.
    const int is_alive = surf2Dread<unsigned char>(r_obj, gx, gy);
    const unsigned char n1 = surf2Dread<unsigned char>(r_obj, (gx - 1 + W) % W, (gy - 1 + H) % H);
    const unsigned char n2 = surf2Dread<unsigned char>(r_obj, gx, (gy - 1 + H) % H);
    const unsigned char n3 = surf2Dread<unsigned char>(r_obj, (gx + 1) % W, (gy - 1 + H) % H);

    const unsigned char n4 = surf2Dread<unsigned char>(r_obj, (gx - 1 + W) % W, gy);
    const unsigned char n5 = surf2Dread<unsigned char>(r_obj, (gx + 1) % W, gy);

    const unsigned char n6 = surf2Dread<unsigned char>(r_obj, (gx - 1 + W) % W, (gy + 1) % H);
    const unsigned char n7 = surf2Dread<unsigned char>(r_obj, gx, (gy + 1) % H);
    const unsigned char n8 = surf2Dread<unsigned char>(r_obj, (gx + 1) % W, (gy + 1) % H);

    const unsigned int count_alive_cells = n1 + n2 + n3 + n4 + n5 + n6 + n7 + n8;

    // Si applicano le regole del gioco

    surf2Dwrite<unsigned char>(
        is_alive == 1 && count_alive_cells > 1 && count_alive_cells < 4 || is_alive == 0 && count_alive_cells == 3,
        w_obj, gx, gy
    );

}

int main() {
    cudaDeviceProp deviceProp = {};
    cudaGetDeviceProperties(&deviceProp, 0);

    int max_threads_per_block = deviceProp.maxThreadsPerBlock;
    int thread_y = (int) sqrt(max_threads_per_block);
    int thread_x = (max_threads_per_block + thread_y - 1) / thread_y;
    if (thread_x > W) thread_x = W;
    if (thread_y > H) thread_y = H;

    int block_x = (W + thread_x - 1) / thread_x;
    int block_y = (H + thread_y - 1) / thread_y;
    dim3 grid_size(block_x, block_y);
    dim3 block_size(thread_x, thread_y);

    int num_cells = W * H;
    unsigned char *d_board_1, *d_board_2;
    unsigned char *h_board = (unsigned char *) malloc(num_cells * sizeof(unsigned char));
    unsigned char *initial_board = (unsigned char *) malloc(num_cells * sizeof(unsigned char));

    LARGE_INTEGER start, end, freq;
    double elapsed_time;
    QueryPerformanceFrequency(&freq);

    CHECK(cudaMalloc(&d_board_1, num_cells * sizeof(unsigned char)));
    CHECK(cudaMalloc(&d_board_2, num_cells * sizeof(unsigned char)));

    srand(time(NULL));
    initialize<<<grid_size, block_size>>>(d_board_1, rand());
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(initial_board, d_board_1, num_cells * sizeof(unsigned char), cudaMemcpyDeviceToHost));

    int pad_thread_x = thread_x;
    int pad_thread_y = thread_y;
    if ((pad_thread_x + 2) * (pad_thread_y + 2) > max_threads_per_block) {
        pad_thread_x -= 2;
        pad_thread_y -= 2;
    }

    block_x = (W + pad_thread_x - 1) / pad_thread_x;
    block_y = (H + pad_thread_y - 1) / pad_thread_y;
    dim3 pad_grid_size(block_x, block_y);
    dim3 pad_block_size(pad_thread_x + 2, pad_thread_y + 2);

    memcpy(h_board, initial_board, num_cells * sizeof(unsigned char));
    QueryPerformanceCounter(&start);
    play_cpu(h_board, 1);
    QueryPerformanceCounter(&end);
    elapsed_time = (double) (end.QuadPart - start.QuadPart) / (double) freq.QuadPart * 100;
    printf("CPU 1.1:\t%.10f\n", elapsed_time);
    free(h_board);


    QueryPerformanceCounter(&start);
    for (int i = 0; i < GEN; i++){
        play_1_1<<<grid_size, block_size>>>(d_board_1, d_board_2);
        CHECK(cudaDeviceSynchronize());
        swap_pointer((void**) &d_board_1, (void**) &d_board_2);
    }
    QueryPerformanceCounter(&end);
    elapsed_time = (double) (end.QuadPart - start.QuadPart) / (double) freq.QuadPart;
    printf("GPU 1.1:\t%.10f\n", elapsed_time);
    

    CHECK(cudaMemcpy(d_board_1, initial_board, num_cells * sizeof(unsigned char), cudaMemcpyHostToDevice));
    QueryPerformanceCounter(&start);
    for (int i = 0; i < GEN; i++) {
        play_1_2<<<grid_size, block_size>>>(d_board_1, d_board_2);
        CHECK(cudaDeviceSynchronize());
        swap_pointer((void**) &d_board_1, (void**) &d_board_2);
    }
    QueryPerformanceCounter(&end);
    elapsed_time = (double) (end.QuadPart - start.QuadPart) / (double) freq.QuadPart;
    printf("GPU 1.2:\t%.10f\n", elapsed_time);
    


    CHECK(cudaMemcpy(d_board_1, initial_board, num_cells * sizeof(unsigned char), cudaMemcpyHostToDevice));
    QueryPerformanceCounter(&start);
    for (int i = 0; i < GEN; i++) {
        play_1_3<<<grid_size, block_size>>>(d_board_1, d_board_2);
        CHECK(cudaDeviceSynchronize());
        swap_pointer((void**) &d_board_1, (void**) &d_board_2);
    }
    QueryPerformanceCounter(&end);
    elapsed_time = (double) (end.QuadPart - start.QuadPart) / (double) freq.QuadPart;
    printf("GPU 1.3:\t%.10f\n", elapsed_time);
    


    CHECK(cudaMemcpy(d_board_1, initial_board, num_cells * sizeof(unsigned char), cudaMemcpyHostToDevice));
    QueryPerformanceCounter(&start);
    for (int i = 0; i < GEN; i++) {
        play_2_1<<<pad_grid_size, pad_block_size, (pad_thread_x + 2) * (pad_thread_y + 2) * sizeof(unsigned char)>>>(d_board_1, d_board_2);
        CHECK(cudaDeviceSynchronize());
        swap_pointer((void**) &d_board_1, (void**) &d_board_2);
    }
    QueryPerformanceCounter(&end);
    elapsed_time = (double) (end.QuadPart - start.QuadPart) / (double) freq.QuadPart;
    printf("GPU 4.1:\t%.10f\n", elapsed_time);
    


    CHECK(cudaMemcpy(d_board_1, initial_board, num_cells * sizeof(unsigned char), cudaMemcpyHostToDevice));
    QueryPerformanceCounter(&start);
    for (int i = 0; i < GEN; i++) {
        play_2_2<<<pad_grid_size, pad_block_size, (pad_thread_x + 2) * (pad_thread_y + 2) * sizeof(unsigned char)>>>(d_board_1, d_board_2);
        CHECK(cudaDeviceSynchronize());
        swap_pointer((void**) &d_board_1, (void**) &d_board_2);
    }
    QueryPerformanceCounter(&end);
    elapsed_time = (double) (end.QuadPart - start.QuadPart) / (double) freq.QuadPart;
    printf("GPU 4.2:\t%.10f\n", elapsed_time);
    CHECK(cudaFree(d_board_1));
    CHECK(cudaFree(d_board_2));
    

    cudaArray_t d_array_1, d_array_2;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<unsigned char>();
    cudaMallocArray(&d_array_1, &channelDesc, W, H, cudaArraySurfaceLoadStore);
    cudaMallocArray(&d_array_2, &channelDesc, W, H, cudaArraySurfaceLoadStore);

    cudaSurfaceObject_t surfObj_1, surfObj_2;
    getSurfaceObject(&surfObj_1, &d_array_1);
    getSurfaceObject(&surfObj_2, &d_array_2);

    CHECK(cudaMemcpy2DToArray(
        d_array_1, 0, 0,
        initial_board, W * sizeof(unsigned char), W, H,
        cudaMemcpyHostToDevice
    ));
    QueryPerformanceCounter(&start);
    for (int i = 0; i < GEN; i++) {
        play_3_1<<<grid_size, block_size>>>(surfObj_1, surfObj_2);
        CHECK(cudaDeviceSynchronize());
        swap_pointer((void**) &surfObj_1, (void**) &surfObj_2);
        swap_pointer((void**) &d_array_1, (void**) &d_array_2);
    }
    QueryPerformanceCounter(&end);
    elapsed_time = (double) (end.QuadPart - start.QuadPart) / (double) freq.QuadPart;
    printf("GPU 5.1:\t%.10f\n", elapsed_time);
    

    
    CHECK(cudaMemcpy2DToArray(
        d_array_1, 0, 0,
        initial_board, W * sizeof(unsigned char), W, H,
        cudaMemcpyHostToDevice
    ));
    QueryPerformanceCounter(&start);
    for (int i = 0; i < GEN; i++) {
        play_3_2<<<grid_size, block_size>>>(surfObj_1, surfObj_2);
        CHECK(cudaDeviceSynchronize());
        swap_pointer((void**) &surfObj_1, (void**) &surfObj_2);
        swap_pointer((void**) &d_array_1, (void**) &d_array_2);
    }
    QueryPerformanceCounter(&end);
    elapsed_time = (double) (end.QuadPart - start.QuadPart) / (double) freq.QuadPart;
    printf("GPU 5.2:\t%.10f\n", elapsed_time);
    



    CHECK(cudaMemcpy2DToArray(
        d_array_1, 0, 0,
        initial_board, W * sizeof(unsigned char), W, H,
        cudaMemcpyHostToDevice
    ));
    unsigned int h_thread_x = (block_size.x + 1) / 2;
    unsigned int h_thread_y = (block_size.y + 1) / 2;
    unsigned int h_block_x = (W + h_thread_x - 1) / h_thread_x;
    unsigned int h_block_y = (H + h_thread_y - 1) / h_thread_y;
    dim3 h_grid_size(h_block_x, h_block_y);
    dim3 h_block_size(h_thread_x, h_thread_y);
    QueryPerformanceCounter(&start);
    for (int i = 0; i < GEN; i++) {
        play_3_2<<<h_grid_size, h_block_size>>>(surfObj_1, surfObj_2);
        CHECK(cudaDeviceSynchronize());
        swap_pointer((void**) &surfObj_1, (void**) &surfObj_2);
        swap_pointer((void**) &d_array_1, (void**) &d_array_2);
    }
    QueryPerformanceCounter(&end);
    elapsed_time = (double) (end.QuadPart - start.QuadPart) / (double) freq.QuadPart;
    printf("GPU 5.3:\t%.10f\n", elapsed_time);
    

    CHECK(cudaDestroySurfaceObject(surfObj_1));
    CHECK(cudaDestroySurfaceObject(surfObj_2));
    CHECK(cudaFreeArray(d_array_1));
    CHECK(cudaFreeArray(d_array_2));
    free(initial_board);
    return 0;
}
