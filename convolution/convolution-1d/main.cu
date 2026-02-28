#include <cuda_runtime.h>
#include <random>
#include <iostream>

__global__ void conv1d_kernel(const float *in, float *out, const float *kernel, int in_size, int kernel_size) {
    int out_size = in_size - kernel_size + 1;
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (out_idx < out_size) {
        float sum = 0.0f;
        for (int k_idx = 0; k_idx < kernel_size; k_idx++) {
            sum += in[out_idx + k_idx] * kernel[k_idx];
        }
        out[out_idx] = sum;
    }
}

void conv1d_cpu(const float *in, float *out, const float *kernel, int in_size, int kernel_size) {
    int out_size = in_size - kernel_size + 1;

    for (int i = 0; i < out_size; i++) {
        float sum = 0.0f;
        for (int j = 0; j < kernel_size; j++) {
            sum += in[i + j] * kernel[j];
        }
        out[i] = sum;
    }
}

int main() {
    int in_size = 99999;
    int kernel_size = 99;
    int out_size = in_size - kernel_size + 1;

    float *h_in = (float *)malloc(in_size * sizeof(float));
    float *h_kernel = (float *)malloc(kernel_size * sizeof(float));
    float *h_out = (float *)malloc(out_size * sizeof(float));

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    for (int i = 0; i < in_size; i++) {
        h_in[i] = dis(gen);
    }
    for (int i = 0; i < kernel_size; i++) {
        h_kernel[i] = dis(gen);
    }

    float *d_in, *d_kernel, *d_out;
    cudaMalloc(&d_in, in_size * sizeof(float));
    cudaMalloc(&d_kernel, kernel_size * sizeof(float));
    cudaMalloc(&d_out, out_size * sizeof(float));

    cudaMemcpy(d_in, h_in, in_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, kernel_size * sizeof(float), cudaMemcpyHostToDevice);

    int threads_per_block = 256;
    int blocks_per_grid = (out_size + threads_per_block - 1) / threads_per_block;
    conv1d_kernel<<<blocks_per_grid, threads_per_block>>>(d_in, d_out, d_kernel, in_size, kernel_size);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, out_size * sizeof(float), cudaMemcpyDeviceToHost);

    float *expected_out = (float *)malloc(out_size * sizeof(float));
    conv1d_cpu(h_in, expected_out, h_kernel, in_size, kernel_size);

    float tolerance = 1e-5;
    for (int i = 0; i < out_size; i++) {
        if (fabs(h_out[i] - expected_out[i]) > tolerance) {
            std::cout << "Fail" << std::endl;
            break;
        }
    }

    std::cout << "Succeed" << std::endl;

    free(h_in);
    free(h_kernel);
    free(h_out);
    free(expected_out);
    cudaFree(d_in);
    cudaFree(d_kernel);
    cudaFree(d_out);
    return 0;
}
