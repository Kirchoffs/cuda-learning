#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <ctime>

__global__ void conv2d_kernel(const int *in, int *out, const int *kernel, int in_height, int in_width, int kernel_side) {
    int out_row = blockIdx.y * blockDim.y + threadIdx.y;
    int out_col = blockIdx.x * blockDim.x + threadIdx.x;

    int out_height = in_height - kernel_side + 1;
    int out_width = in_width - kernel_side + 1;

    if (out_row < out_height && out_col < out_width) {
        int sum = 0;
        for (int kernel_row = 0; kernel_row < kernel_side; kernel_row++) {
            for (int kernel_col = 0; kernel_col < kernel_side; kernel_col++) {
                int in_row = out_row + kernel_row;
                int in_col = out_col + kernel_col;
                sum += in[in_row * in_width + in_col] * kernel[kernel_row * kernel_side + kernel_col];
            }
        }
        out[out_row * out_width + out_col] = sum;
    }
}

void conv2d_cpu(const int *in, int *out, const int *kernel, int in_height, int in_width, int kernel_side) {
    int out_height = in_height - kernel_side + 1;
    int out_width = in_width - kernel_side + 1;

    for (int out_row = 0; out_row < out_height; out_row++) {
        for (int out_col = 0; out_col < out_width; out_col++) {
            int sum = 0;
            for (int kernel_row = 0; kernel_row < kernel_side; kernel_row++) {
                for (int kernel_col = 0; kernel_col < kernel_side; kernel_col++) {
                    int in_row = out_row + kernel_row;
                    int in_col = out_col + kernel_col;

                    sum += in[in_row * in_width + in_col] * kernel[kernel_row * kernel_side + kernel_col];
                }
            }
            out[out_row * out_width + out_col] = sum;
        }
    }
}

int main() {
    srand(static_cast<unsigned int>(time(NULL)));

    int in_height = 999;
    int in_width = 499;
    int kernel_side = 49;

    int out_height = in_height - kernel_side + 1;
    int out_width = in_width - kernel_side + 1;

    size_t in_size = in_height * in_width * sizeof(int);
    size_t kernel_size = kernel_side * kernel_side * sizeof(int);
    size_t out_size = out_height * out_width * sizeof(int);

    int *h_in = (int *)malloc(in_size);
    int *h_kernel = (int *)malloc(kernel_size);
    int *h_out = (int *)malloc(out_size);

    for (int i = 0; i < in_height * in_width; i++) {
        h_in[i] = rand() % 256; 
    }

    for (int i = 0; i < kernel_side * kernel_side; i++) {
        h_kernel[i] = (rand() % 5) - 2;
    }

    int *d_in, *d_kernel, *d_out;
    cudaMalloc(&d_in, in_size);
    cudaMalloc(&d_kernel, kernel_size);
    cudaMalloc(&d_out, out_size);

    cudaMemcpy(d_in, h_in, in_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, kernel_size, cudaMemcpyHostToDevice);

    dim3 block_size(16, 16);
    dim3 grid_size(
        (out_width + block_size.x - 1) / block_size.x, 
        (out_height + block_size.y - 1) / block_size.y
    );

    conv2d_kernel<<<grid_size, block_size>>>(d_in, d_out, d_kernel, in_height, in_width, kernel_side);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    cudaMemcpy(h_out, d_out, out_size, cudaMemcpyDeviceToHost);

    int *expected_out = (int *)malloc(out_size);
    conv2d_cpu(h_in, expected_out, h_kernel, in_height, in_width, kernel_side);

    bool matched = true;
    for (int i = 0; i < out_height; i++) {
        for (int j = 0; j < out_width; j++) {
            int idx = i * out_width + j;

            if (h_out[idx] != expected_out[idx]) {
                matched = false;
                break;
            }
        }
    }

    std::cout << (matched ? "Passed!" : "Failed") << std::endl;

    free(h_in); free(h_kernel); free(h_out);
    cudaFree(d_in); cudaFree(d_kernel); cudaFree(d_out);
    free(expected_out);

    return 0;
}
