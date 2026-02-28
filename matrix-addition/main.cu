#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

__global__ void matrix_add_kernel(const float *A, const float *B, float *C, int num_rows, int num_cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < num_rows && col < num_cols) {
        int index = row * num_cols + col;
        C[index] = A[index] + B[index];
    }
}

void matrix_add_cpu(const float *A, const float *B, float *C, int num_rows, int num_cols) {
    for (int i = 0; i < num_rows * num_cols; i++) {
        C[i] = A[i] + B[i];
    }
}

void init_matrix(float *matrix, int num_rows, int num_cols) {
    for (int i = 0; i < num_rows * num_cols; i++) {
        matrix[i] = rand() / (float)RAND_MAX;
    }
}

bool compare_matrices(const float *C_gpu, const float *C_cpu, int num_elements, float tolerance = 1e-5) {
    for (int i = 0; i < num_elements; i++) {
        if (fabs(C_gpu[i] - C_cpu[i]) > tolerance) {
            printf("Mismatch at index %d: GPU = %f, CPU = %f\n", i, C_gpu[i], C_cpu[i]);
            return false;
        }
    }
    return true;
}

int main() {
    int num_rows = 999;
    int num_cols = 499;
    int num_elements = num_rows * num_cols;
    size_t matrix_size = num_elements * sizeof(float);

    float *h_A = (float *)malloc(matrix_size);
    float *h_B = (float *)malloc(matrix_size);
    float *h_C_gpu = (float *)malloc(matrix_size);
    float *h_C_cpu = (float *)malloc(matrix_size);
    if (!h_A || !h_B || !h_C_gpu || !h_C_cpu) {
        printf("Host memory allocation failed\n");
        return -1;
    }

    srand(42);
    init_matrix(h_A, num_rows, num_cols);
    init_matrix(h_B, num_rows, num_cols);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, matrix_size);
    cudaMalloc(&d_B, matrix_size);
    cudaMalloc(&d_C, matrix_size);
    if (!d_A || !d_B || !d_C) {
        printf("Device memory allocation failed\n");
        return -1;
    }

    cudaMemcpy(d_A, h_A, matrix_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, matrix_size, cudaMemcpyHostToDevice);

    dim3 threads_per_block(16, 16);
    dim3 blocks_per_grid(
        (num_cols + threads_per_block.x - 1) / threads_per_block.x,
        (num_rows + threads_per_block.y - 1) / threads_per_block.y
    );

    printf("Kernel configuration:\n");
    printf("  Grid: %d x %d blocks\n", blocks_per_grid.x, blocks_per_grid.y);
    printf("  Block: %d x %d threads\n", threads_per_block.x, threads_per_block.y);
    printf("  Total threads: %d\n", blocks_per_grid.x * blocks_per_grid.y * threads_per_block.x * threads_per_block.y);
    printf("  Matrix size: %d x %d = %d elements\n\n", num_rows, num_cols, num_elements);

    matrix_add_kernel<<<blocks_per_grid, threads_per_block>>>(d_A, d_B, d_C, num_rows, num_cols);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    cudaDeviceSynchronize();

    cudaMemcpy(h_C_gpu, d_C, matrix_size, cudaMemcpyDeviceToHost);

    matrix_add_cpu(h_A, h_B, h_C_cpu, num_rows, num_cols);

    printf("Verifying results...\n");
    bool is_correct = compare_matrices(h_C_gpu, h_C_cpu, num_elements);
    
    if (is_correct) {
        printf("Verification passed! GPU and CPU results match.\n");
        
        printf("\nFirst 5 results (GPU vs CPU):\n");
        for (int i = 0; i < min(5, num_elements); i++) {
            printf("  Index %d: GPU = %f, CPU = %f\n", i, h_C_gpu[i], h_C_cpu[i]);
        }
    } else {
        printf("Verification failed! GPU and CPU results differ.\n");
    }

    free(h_A);
    free(h_B);
    free(h_C_gpu);
    free(h_C_cpu);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
