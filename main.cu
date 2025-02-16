#include <iostream>
#include <curand.h>
#include <curand_kernel.h>
#include <cmath>
#include <filesystem>
#define W 10000
#define H 10000
#define GEN 100

__host__ void print(bool* board) {
    for (int j = 0; j < H; j++) {
        for (int i = 0; i < W;  i++) {
            if (board[i + j * W])
                printf("*");
            else
                printf("#");
        }
        printf("\n");
    }
    printf("\n\n\n");
}

__global__ void initialize(bool* board, const int seed) {
    const int gx = blockIdx.x * blockDim.x + threadIdx.x;
    const int gy = blockIdx.y * blockDim.y + threadIdx.y;

    if (gx >= W || gy >= H) return;

    const int linear_index = gx + gy * W;
    curandState state;
    curand_init(seed, linear_index, 0, &state);
    board[linear_index] = curand_uniform(&state) > 0.5;
}


__global__ void play_1(bool *board, const int generations) {

    // Dimensioni blocchi senza padding
    const int blockDimx = blockDim.x - 2;
    const int blockDimy = blockDim.y - 2;

    const int block_dim = blockDim.x * blockDim.y;

    // Indici matrice
    int gx = (int) blockIdx.x * blockDimx + (int) threadIdx.x - 1;
    int gy = (int) blockIdx.y * blockDimy + (int) threadIdx.y - 1;

    extern __shared__ bool shared[];

    // Controllo posizione thread
    bool is_outside = gx < -1 || gx > W || gy < -1 || gy > H;

    for (int i = 0; i < generations; i++) {

        // Indici toroidali
        const int tor_x = !is_outside * ((gx + W) % W) + is_outside * W;
        const int tor_y = !is_outside * ((gy + H) % H) + is_outside * (H - 1);
        const int tor_idx = tor_x + tor_y * W;

        // Indice memoria condivisa
        const int l_idx = !is_outside * (threadIdx.y * blockDim.x + threadIdx.x) + is_outside * block_dim;

        shared[l_idx] = board[tor_idx];

        __syncthreads();

        is_outside = threadIdx.x < 1 || threadIdx.y < 1 || threadIdx.x > blockDimx || threadIdx.y > blockDimy
            || gx < 0 || gy < 0 || gx > W - 1 || gy > H - 1;

        // Indici matrice
        gx = !is_outside * gx + is_outside * W;
        gy = !is_outside * gy + is_outside * (H - 1);
        const int g_idx = gx + gy * W;

        int count_alive_cells = 0;
        const bool is_alive = shared[l_idx];

        // Indice celle vicine
        int n_idx = (threadIdx.y - 1) * blockDim.x + threadIdx.x - 1;

        count_alive_cells += !is_outside * shared[!is_outside * n_idx + is_outside * block_dim];
        count_alive_cells += !is_outside * shared[!is_outside * (n_idx + 1) + is_outside * block_dim];
        count_alive_cells += !is_outside * shared[!is_outside * (n_idx + 2) + is_outside * block_dim];

        n_idx = threadIdx.y * blockDim.x + threadIdx.x - 1;

        count_alive_cells += !is_outside * shared[!is_outside * n_idx + is_outside * block_dim];
        count_alive_cells += !is_outside * shared[!is_outside * (n_idx + 2) + is_outside * block_dim];

        n_idx = (threadIdx.y + 1) * blockDim.x + threadIdx.x - 1;

        count_alive_cells += !is_outside * shared[!is_outside * n_idx + is_outside * block_dim];
        count_alive_cells += !is_outside * shared[!is_outside * (n_idx + 1) + is_outside * block_dim];
        count_alive_cells += !is_outside * shared[!is_outside * (n_idx + 2) + is_outside * block_dim];

        // Si applicano le regole del gioco
        board[g_idx] = is_alive && (count_alive_cells > 1 && count_alive_cells < 4) ||
            !is_alive && count_alive_cells == 3;

        __syncthreads();
    }

}

__global__ void play_2(bool *board, const int generations) {

    extern __shared__ bool shared[];

    // Dimensioni blocco senza padding
    const int blockDimx = blockDim.x - 2;
    const int blockDimy = blockDim.y - 2;

    // Indici matriciali
    int gx = (int) blockIdx.x * blockDimx + (int) threadIdx.x - 1;
    int gy = (int) blockIdx.y * blockDimy + (int) threadIdx.y - 1;

    // Controllo posizione thread
    bool is_outside = gx < -1 || gx > W || gy < -1 || gy > H;

    for (int i = 0; i < generations; i++) {

        // Indici toroidali
        const int tor_x = (gx % W + W) % W;
        const int tor_y = (gy % H + H) % H;
        const int tor_idx = tor_x + tor_y * W;

        // Indice memoria condivisa
        const int l_idx = threadIdx.y * blockDim.x + threadIdx.x;

        // Se controlla che il thread è nei limiti per evitare accessi illegali in memoria
        if (!is_outside)
            shared[l_idx] = board[tor_idx];

        __syncthreads();

        // Si aggiornano i limiti
        is_outside = threadIdx.x < 1 || threadIdx.y < 1 || threadIdx.x > blockDimx || threadIdx.y > blockDimy
            || gx < 0 || gy < 0 || gx > W - 1 || gy > H - 1;

        if (!is_outside) {

            // Indice matrice
            const int g_idx = gx + gy * W;

            int count_alive_cells = 0;
            const bool is_alive = shared[l_idx];

            // Indice celle vicine
            int n_idx = (threadIdx.y - 1) * blockDim.x + threadIdx.x - 1;
            count_alive_cells += shared[n_idx];
            count_alive_cells += shared[n_idx + 1];
            count_alive_cells += shared[n_idx + 2];

            n_idx = threadIdx.y * blockDim.x + threadIdx.x - 1;
            count_alive_cells += shared[n_idx];
            count_alive_cells += shared[n_idx + 2];

            n_idx = (threadIdx.y + 1) * blockDim.x + threadIdx.x - 1;
            count_alive_cells += shared[n_idx];
            count_alive_cells += shared[n_idx + 1];
            count_alive_cells += shared[n_idx + 2];

            // Si applicano le regole del gioco
            board[g_idx] = is_alive && (count_alive_cells > 1 && count_alive_cells < 4) ||
                !is_alive && count_alive_cells == 3;
        }
        __syncthreads();
    }
}

__global__ void play_3(bool *board, const int generations) {

    extern __shared__ bool shared[];

    // Indici matrice
    const int gx = threadIdx.x + blockDim.x * blockIdx.x;
    const int gy = threadIdx.y + blockDim.y * blockIdx.y;

    // Se fuori dai limiti ritorna
    if (gx >= W || gy >= H) return;


    const int block_dim = blockDim.x * blockDim.y;
    const int padded_block_dim = (blockDim.x + 2) * (blockDim.y + 2);

    const int t_idx = threadIdx.x + threadIdx.y * blockDim.x;
    const int cells_to_cache = (padded_block_dim + block_dim - 1) / block_dim;

    // Numero di celle che ogni thread deve salvare nella memoria condivisa
    const int start = t_idx * cells_to_cache;
    const int end = start + cells_to_cache;

    for (int g = 0; g < generations; g++) {
        for (int i = start; i < end; i++) {

            // Si calcolano gli indici della matrice
            int y = i / (blockDim.x + 2);
            int x = i - y * (blockDim.x + 2);
            y = ((y - 1 + (int) (blockDim.y * blockIdx.y)) % H + H) % H;
            x = ((x - 1 + (int) (blockDim.x * blockIdx.x)) % W + W) % W;

            shared[i] = board[x + y * W];
        }

        __syncthreads();

        int count_alive_cells = 0;
        const int is_alive = shared[threadIdx.x + 1 + (threadIdx.y + 1) * (blockDim.x + 2)];

        // Indici celle vicine
        int n_idx = threadIdx.x + threadIdx.y * (blockDim.x + 2);
        count_alive_cells += shared[n_idx];
        count_alive_cells += shared[n_idx + 1];
        count_alive_cells += shared[n_idx + 2];

        n_idx = threadIdx.x + (threadIdx.y + 1) * (blockDim.x + 2);
        count_alive_cells += shared[n_idx];
        count_alive_cells += shared[n_idx + 2];

        n_idx = threadIdx.x + (threadIdx.y + 2) * (blockDim.x + 2);
        count_alive_cells += shared[n_idx];
        count_alive_cells += shared[n_idx + 1];
        count_alive_cells += shared[n_idx + 2];

        // Si applicano le regole del gioco
        board[gx + gy * W] = is_alive && (count_alive_cells > 1 && count_alive_cells < 4) ||
            !is_alive && count_alive_cells == 3;

        __syncthreads();
    }
}

__global__ void play_4(bool* board, int generations) {

    // Indici matrice
    const int gx = threadIdx.x + blockDim.x * blockIdx.x;
    const int gy = threadIdx.y + blockDim.y * blockIdx.y;

    // Se fuori dal limite, si ritorna
    if (gx >= W || gy >= H) return;

    for (int i = 0; i < generations; i++) {
        const int is_alive = board[gx + gy * W];
        int count_alive_cells = 0;

        count_alive_cells += board[((gx - 1) % W + W) % W + ((gy - 1) % H + H) % H * W];
        count_alive_cells += board[(gx % W + W) % W + ((gy - 1) % H + H) % H * W];
        count_alive_cells += board[((gx + 1) % W + W) % W + ((gy - 1) % H + H) % H * W];

        count_alive_cells += board[((gx - 1) % W + W) % W + (gy % H + H) % H * W];
        count_alive_cells += board[((gx + 1) % W + W) % W + (gy % H + H) % H * W];

        count_alive_cells += board[((gx - 1) % W + W) % W + ((gy + 1) % H + H) % H * W];
        count_alive_cells += board[(gx % W + W) % W + ((gy + 1) % H + H) % H * W];
        count_alive_cells += board[((gx + 1) % W + W) % W + ((gy + 1) % H + H) % H * W];

        __syncthreads();

        // Si applicano le regole del gioco
        board[gx + gy * W] = is_alive && (count_alive_cells > 1 && count_alive_cells < 4) ||
                !is_alive && count_alive_cells == 3;

        __syncthreads();
    }
}

__global__ void play_5(const cudaTextureObject_t tex_obj, bool* board, const int generations) {

    // Indici matrice
    const int gx = threadIdx.x + blockDim.x * blockIdx.x;
    const int gy = threadIdx.y + blockDim.y * blockIdx.y;

    // Se fuori dal limite, si ritorna
    if (gx >= W || gy >= H) return;

    for (int i = 0; i < generations; i++) {
        int count_alive_cells = 0;

        // Si caricano i dati dalla memoria globale attraverso l'uso della texture memory.
        // Purtroppo il tipo bool non è supportato è quindi necessario usare un altro tipo che abbia dimensioni
        // uguali, cioè un byte. In questo caso abbiamo usato il char.
        const bool is_alive = tex1Dfetch<char>(tex_obj, gx + gy * W);
        count_alive_cells += tex1Dfetch<char>(tex_obj, ((gx - 1) % W + W) % W + ((gy - 1) % H + H) % H * W);
        count_alive_cells += tex1Dfetch<char>(tex_obj, (gx % W + W) % W + ((gy - 1) % H + H) % H * W);
        count_alive_cells += tex1Dfetch<char>(tex_obj, ((gx + 1) % W + W) % W + ((gy - 1) % H + H) % H * W);

        count_alive_cells += tex1Dfetch<char>(tex_obj, ((gx - 1) % W + W) % W + (gy % H + H) % H * W);
        count_alive_cells += tex1Dfetch<char>(tex_obj, ((gx + 1) % W + W) % W + (gy % H + H) % H * W);

        count_alive_cells += tex1Dfetch<char>(tex_obj, ((gx - 1) % W + W) % W + ((gy + 1) % H + H) % H * W);
        count_alive_cells += tex1Dfetch<char>(tex_obj, (gx % W + W) % W + ((gy + 1) % H + H) % H * W);
        count_alive_cells += tex1Dfetch<char>(tex_obj, ((gx + 1) % W + W) % W + ((gy + 1) % H + H) % H * W);

        __syncthreads();

        // Si applicano le regole del gioco
        board[gx + gy * W] = is_alive && (count_alive_cells > 1 && count_alive_cells < 4) ||
                        !is_alive && count_alive_cells == 3;

        __syncthreads();
    }
}

__global__ void play_6(bool* board, const int generations) {

    extern __shared__ bool shared[];
    const int padded_x_size = blockDim.x + 2;
    const int padded_y_size = blockDim.y + 2;

    // Indici matrice
    const int gx = blockIdx.x * blockDim.x + threadIdx.x;
    const int gy = blockIdx.y * blockDim.y + threadIdx.y;

    // se fuori dai limiti, si ritorna
    if (gx >= W || gy >= H) return;

    // Indice blocco
    const int lx = threadIdx.x + 1;
    const int ly = threadIdx.y + 1;

    for (int gen = 0; gen < generations; gen++) {

        // Caricamento memoria globale nella memoria condivisa
        shared[lx + ly * padded_x_size] = board[gx + gy * W];

        // Gestione bordi tramite if. Alcuni thread lavorano più di altri
        if (threadIdx.x == 0) {
            shared[ly * padded_x_size] = board[(gx - 1 + W) % W + gy * W];
        }
        if (threadIdx.x == blockDim.x - 1) {
            shared[(ly + 1) * padded_x_size - 1] = board[(gx + 1) % W + gy * W];
        }
        if (threadIdx.y == 0) {
            shared[lx] = board[gx + (gy - 1 + H) % H * W];
        }
        if (threadIdx.y == blockDim.y - 1) {
            shared[lx + ly * (padded_y_size - 1)] = board[gx + (gy + 1) % H * W];
        }

        __syncthreads();

        const bool is_alive = shared[lx + ly * padded_x_size];

        // Calcola vicini dalla shared memory
        int count_alive_cells = 0;
        count_alive_cells += shared[lx - 1 + (ly - 1) * padded_x_size];
        count_alive_cells += shared[lx + (ly - 1) * padded_x_size];
        count_alive_cells += shared[lx + 1 + (ly - 1) * padded_x_size];
        count_alive_cells += shared[lx - 1 + ly * padded_x_size];
        count_alive_cells += shared[lx + 1 + ly * padded_x_size];
        count_alive_cells += shared[lx - 1 + (ly + 1) * padded_x_size];
        count_alive_cells += shared[lx + (ly + 1) * padded_x_size];
        count_alive_cells += shared[lx + 1 + (ly + 1) * padded_x_size];

        // Si applicano le regole del gioco
        board[gx + gy * W] = is_alive && (count_alive_cells > 1 && count_alive_cells < 4) ||
                        !is_alive && count_alive_cells == 3;

        __syncthreads();
    }
}


int main() {

    cudaDeviceProp deviceProp = {};
    cudaGetDeviceProperties(&deviceProp, 0);

    int max_threads_per_block = deviceProp.maxThreadsPerBlock;
    int block_y = (int) sqrt(max_threads_per_block);
    int block_x = (max_threads_per_block + block_y - 1) / block_y;
    if (block_x > W) block_x = W;
    if (block_y > H) block_y = H;

    int grid_x = (W + block_x - 1) / block_x;
    int grid_y = (H + block_y - 1) / block_y;
    dim3 grid_size(grid_x, grid_y);
    dim3 block_size(block_x, block_y);

    int num_cells = W * H;
    bool* d_board;
    bool* h_board = (bool*) malloc(num_cells * sizeof(bool));
    bool* initial_board = (bool*) malloc(num_cells * sizeof(bool));
    // memset(initial_board, 0, num_cells * sizeof(bool));
    // initial_board[1 + 0 * W] = true;
    // initial_board[2 + 1 * W] = true;
    // initial_board[0 + 2 * W] = true;
    // initial_board[1 + 2 * W] = true;
    // initial_board[2 + 2 * W] = true;

    cudaMalloc(&d_board, num_cells * sizeof(bool));
    srand(time(NULL));
    initialize<<<grid_size, block_size>>>(d_board, rand());
    cudaDeviceSynchronize();
    cudaMemcpy(initial_board, d_board, num_cells * sizeof(bool), cudaMemcpyDeviceToHost);
    cudaFree(d_board);

    int pad_block_x = block_x;
    int pad_block_y = block_y;
    if ((pad_block_x + 2) * (pad_block_y + 2) > max_threads_per_block) {
        pad_block_x -= 2;
        pad_block_y -= 2;
    }

    grid_x = (W + pad_block_x - 1) / pad_block_x;
    grid_y = (H + pad_block_y - 1) / pad_block_y;
    dim3 pad_grid_size (grid_x, grid_y);
    dim3 pad_block_size(pad_block_x + 2, pad_block_y + 2);


    cudaMalloc(&d_board, (num_cells + 1) * sizeof(bool));
    cudaMemcpy(d_board, initial_board, num_cells * sizeof(bool), cudaMemcpyHostToDevice);
    play_1<<<pad_grid_size, pad_block_size, ((pad_block_x + 2) * (pad_block_y + 2) + 1) * sizeof(bool)>>>(d_board, GEN);
    cudaDeviceSynchronize();
    cudaFree(d_board);

    cudaMalloc(&d_board, num_cells * sizeof(bool));
    cudaMemcpy(d_board, initial_board, num_cells * sizeof(bool), cudaMemcpyHostToDevice);
    play_2<<<pad_grid_size, pad_block_size, (pad_block_x + 2) * (pad_block_y + 2) * sizeof(bool)>>>(d_board, GEN);
    cudaDeviceSynchronize();
    cudaFree(d_board);

    cudaMalloc(&d_board, num_cells * sizeof(bool));
    cudaMemcpy(d_board, initial_board, num_cells * sizeof(bool), cudaMemcpyHostToDevice);
    int cells = ((block_x + 2) * (block_y + 2) + block_x * block_size.y - 1) / (block_x * block_y);
    play_3<<<grid_size, block_size, cells * block_x * block_y * sizeof(bool)>>>(d_board, GEN);
    cudaDeviceSynchronize();
    cudaFree(d_board);

    cudaMalloc(&d_board, num_cells * sizeof(bool));
    cudaMemcpy(d_board, initial_board, num_cells * sizeof(bool), cudaMemcpyHostToDevice);
    play_4<<<grid_size, block_size>>>(d_board, GEN);
    cudaDeviceSynchronize();
    cudaFree(d_board);

    cudaMalloc(&d_board, num_cells * sizeof(bool));
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = d_board;
    resDesc.res.linear.sizeInBytes = num_cells * sizeof(bool);
    resDesc.res.linear.desc = cudaCreateChannelDesc<char>();
    cudaTextureDesc texDesc = {};
    texDesc.readMode = cudaReadModeElementType;
    cudaTextureObject_t texObj = {};
    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);
    cudaMemcpy(d_board, initial_board, num_cells * sizeof(bool), cudaMemcpyHostToDevice);
    play_5<<<grid_size, block_size>>>(texObj, d_board, GEN);
    cudaDeviceSynchronize();
    cudaFree(d_board);
    cudaDestroyTextureObject(texObj);

    cudaMalloc(&d_board, num_cells * sizeof(bool));
    cudaMemcpy(d_board, initial_board, num_cells * sizeof(bool), cudaMemcpyHostToDevice);
    play_6<<<grid_size, block_size, (block_x + 2) * (block_y + 2) * sizeof(bool)>>>(d_board, GEN);
    cudaDeviceSynchronize();
    cudaFree(d_board);
    
    free(h_board);
    free(initial_board);
    return 0;
}