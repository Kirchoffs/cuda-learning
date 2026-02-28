#include <cuda_runtime.h>
#include <iostream>
#include <random>

__global__ void tensor_add_3d_kernel(const int *A, const int *B, int *C, int depth, int height, int width) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (z < depth && y < height && x < width) {
        int index = z * (height * width) + y * width + x;
        C[index] = A[index] + B[index];
    }
}

int main() {
    int depth = 32, height = 256, width = 256;
    size_t size = depth * height * width * sizeof(int);

    int *h_A = (int*)malloc(size);
    int *h_B = (int*)malloc(size);
    int *h_C = (int*)malloc(size);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, 42);
    for (int i = 0; i < depth * height * width; i++) {
        h_A[i] = dis(gen);
        h_B[i] = dis(gen);
    }

    int *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 threads_per_block(8, 8, 8);
    dim3 blocks_per_grid(
        (width + threads_per_block.x - 1) / threads_per_block.x,
        (height + threads_per_block.y - 1) / threads_per_block.y,
        (depth + threads_per_block.z - 1) / threads_per_block.z
    );

    tensor_add_3d_kernel<<<blocks_per_grid, threads_per_block>>>(d_A, d_B, d_C, depth, height, width);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    bool success = true;
    int error_count = 0;

    for (int i = 0; i < depth * height * width; i++) {
        int expected = h_A[i] + h_B[i];
        
        if (h_C[i] != expected) {
            if (error_count < 15) {
                printf("Error at index %d: GPU=%d, CPU=%d\n", i, h_C[i], expected);
            }
            success = false;
            error_count++;
        }
    }

    if (success) {
        printf("PASS: All %d elements are correct!\n", depth * height * width);
    } else {
        printf("FAIL: Found %d errors in total.\n", error_count);
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
