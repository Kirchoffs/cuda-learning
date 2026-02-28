#include <cuda_runtime.h>
#include <stdio.h>

__global__ void transpose_kernel(const int *in, int *out, int num_rows, int num_cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < num_rows && col < num_cols) {
        int in_idx = row * num_cols + col;
        int out_idx = col * num_rows + row;
        out[out_idx] = in[in_idx];
    }
}

int main() {
    int num_rows = 1000;
    int num_cols = 500;
    int num_elements = num_rows * num_cols;
    size_t matrix_size = num_elements * sizeof(int);

    int *h_in = (int *)malloc(matrix_size);
    int *h_out = (int *)malloc(matrix_size);

    for (int i = 0; i < num_elements; i++) {
        h_in[i] = i + 1;
    }

    int *d_in, *d_out;
    cudaMalloc(&d_in, matrix_size);
    cudaMalloc(&d_out, matrix_size);

    cudaMemcpy(d_in, h_in, matrix_size, cudaMemcpyHostToDevice);

    dim3 threads_per_block(16, 16);
    dim3 blocks_per_grid(
        (num_cols + threads_per_block.x - 1) / threads_per_block.x,
        (num_rows + threads_per_block.y - 1) / threads_per_block.y
    );

    transpose_kernel<<<blocks_per_grid, threads_per_block>>>(d_in, d_out, num_rows, num_cols);

    cudaMemcpy(h_out, d_out, matrix_size, cudaMemcpyDeviceToHost);

    printf("Verifying all elements...\n");
    int correct = 1;
    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < num_cols; j++) {
            int original = h_in[i * num_cols + j];
            int transposed = h_out[j * num_rows + i];
            
            if (original != transposed) {
                printf(
                    "Mismatch at (%d,%d): original = %d, transposed = %d\n", 
                    i, j, original, transposed
                );
                correct = 0;
                break;
            }
        }
        if (!correct) break;
    }
    
    if (correct) {
        printf("All %d elements verified! Transpose is correct.\n", num_elements);
    } else {
        printf("Transpose is incorrect!\n");
    }

    free(h_in);
    free(h_out);
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}
