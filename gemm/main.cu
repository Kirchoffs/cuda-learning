#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

__global__ void gemm_kernel(const float *A, const float *B, float *C, int num_rows, int num_cols, int shared_dim) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < num_rows && col < num_cols) {
        float sum = 0;

        for (int k = 0; k < shared_dim; k++) {
            sum += A[row * shared_dim + k] * B[k * num_cols + col];
        }
        
        C[row * num_cols + col] = sum;
    }
}

void gemm_cpu(const float *A, const float *B, float *C, int num_rows, int num_cols, int shared_dim) {
    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < num_cols; j++) {
            float sum = 0;
            for (int k = 0; k < shared_dim; k++) {
                sum += A[i * shared_dim + k] * B[k * num_cols + j];
            }
            C[i * num_cols + j] = sum;
        }
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

void init_matrix(float *matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = rand() / (float)RAND_MAX;
    }
}

int main() {
    int num_rows = 100;
    int num_cols = 80;
    int shared_dim = 120;
    
    int num_elements_C = num_rows * num_cols;
    size_t size_A = num_rows * shared_dim * sizeof(float);
    size_t size_B = shared_dim * num_cols * sizeof(float);
    size_t size_C = num_elements_C * sizeof(float);
    
    float *h_A = (float *)malloc(size_A);
    float *h_B = (float *)malloc(size_B);
    float *h_C_gpu = (float *)malloc(size_C);
    float *h_C_cpu = (float *)malloc(size_C);
    
    if (!h_A || !h_B || !h_C_gpu || !h_C_cpu) {
        printf("Host memory allocation failed\n");
        return -1;
    }
    
    srand(42);
    init_matrix(h_A, num_rows, shared_dim);
    init_matrix(h_B, shared_dim, num_cols);
    
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);
    
    if (!d_A || !d_B || !d_C) {
        printf("Device memory allocation failed\n");
        return -1;
    }

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
    
    dim3 threads_per_block(16, 16);
    dim3 blocks_per_grid(
        (num_cols + threads_per_block.x - 1) / threads_per_block.x,
        (num_rows + threads_per_block.y - 1) / threads_per_block.y
    );
    
    printf("Matrix Multiplication Configuration:\n");
    printf("  A: %d x %d\n", num_rows, shared_dim);
    printf("  B: %d x %d\n", shared_dim, num_cols);
    printf("  C: %d x %d\n", num_rows, num_cols);
    printf("  Grid: %d x %d blocks\n", blocks_per_grid.x, blocks_per_grid.y);
    printf("  Block: %d x %d threads\n", threads_per_block.x, threads_per_block.y);
    printf("  Total threads launched: %d\n\n", 
           blocks_per_grid.x * blocks_per_grid.y * threads_per_block.x * threads_per_block.y);
    
    gemm_kernel<<<blocks_per_grid, threads_per_block>>>(d_A, d_B, d_C, num_rows, num_cols, shared_dim);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_C_gpu, d_C, size_C, cudaMemcpyDeviceToHost);
    
    gemm_cpu(h_A, h_B, h_C_cpu, num_rows, num_cols, shared_dim);
    
    printf("Verifying results...\n");
    bool is_correct = compare_matrices(h_C_gpu, h_C_cpu, num_elements_C);
    
    if (is_correct) {
        printf("Verification passed! GPU and CPU results match.\n");
        
        printf("\nFirst 5 results (GPU vs CPU):\n");
        for (int i = 0; i < (5 < num_elements_C ? 5 : num_elements_C); i++) {
            printf("  Index %d: GPU = %f, CPU = %f\n", i, h_C_gpu[i], h_C_cpu[i]);
        }
        
        printf("\nSample of result matrix C (first 5x5):\n");
        for (int i = 0; i < 5 && i < num_rows; i++) {
            for (int j = 0; j < 5 && j < num_cols; j++) {
                printf("%8.4f ", h_C_gpu[i * num_cols + j]);
            }
            printf("\n");
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
