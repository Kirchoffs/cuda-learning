#include <cuda_runtime.h>
#include <random>
#include <iostream>
#include <cmath>
#include <cfloat>
#include <cstdlib>

__global__ void max_pool2d_kernel(const float *in, float *out, int in_height, int in_width, int pool_side) {
    int out_row = blockIdx.y * blockDim.y + threadIdx.y;
    int out_col = blockIdx.x * blockDim.x + threadIdx.x;

    int out_height = in_height / pool_side;
    int out_width = in_width / pool_side;

    if (out_row < out_height && out_col < out_width) {
        float max_val = -FLT_MAX;
        for (int pool_row = 0; pool_row < pool_side; pool_row++) {
            for (int pool_col = 0; pool_col < pool_side; pool_col++) {
                int in_row = out_row * pool_side + pool_row;
                int in_col = out_col * pool_side + pool_col;

                float val = in[in_row * in_width + in_col];
                if (val > max_val) {
                    max_val = val;
                }
            }
        }

        out[out_row * out_width + out_col] = max_val;
    }
}

void max_pool2d_cpu(const float *in, float *out, int in_height, int in_width, int pool_side) {
    int out_height = in_height / pool_side;
    int out_width = in_width / pool_side;

    for (int out_row = 0; out_row < out_height; out_row++) {
        for (int out_col = 0; out_col < out_width; out_col++) {
            float max_val = -FLT_MAX;
            for (int pool_row = 0; pool_row < pool_side; pool_row++) {
                for (int pool_col = 0; pool_col < pool_side; pool_col++) {
                    int in_row = out_row * pool_side + pool_row;
                    int in_col = out_col * pool_side + pool_col;

                    float val = in[in_row * in_width + in_col];
                    if (val > max_val) {
                        max_val = val;
                    }
                }
            }

            out[out_row * out_width + out_col] = max_val;
        }
    }
}

int main() {
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dist_real{0.0, 5.0};

    int in_height = 8192;
    int in_width = 8192;
    int pool_side = 16;

    int out_height = in_height / pool_side;
    int out_width = in_width / pool_side;

    int in_size = in_height * in_width * sizeof(float);
    int out_size = out_height * out_width * sizeof(float);

    float *h_in = (float *)malloc(in_size);
    float *h_out = (float *)malloc(out_size);

    for (int i = 0; i < in_height * in_width; i++) {
        h_in[i] = dist_real(gen);
    }

    float *d_in, *d_out;
    cudaMalloc(&d_in, in_size);
    cudaMalloc(&d_out, out_size);

    cudaMemcpy(d_in, h_in, in_size, cudaMemcpyHostToDevice);

    dim3 threads_per_block(16, 16);
    dim3 blocks_per_grid(
        (out_width + threads_per_block.x - 1) / threads_per_block.x,
        (out_height + threads_per_block.y - 1) / threads_per_block.y
    );
    max_pool2d_kernel<<<blocks_per_grid, threads_per_block>>>(d_in, d_out, in_height, in_width, pool_side);

    cudaMemcpy(h_out, d_out, out_size, cudaMemcpyDeviceToHost);

    float *expected_out = (float *)malloc(out_size);
    max_pool2d_cpu(h_in, expected_out, in_height, in_width, pool_side);

    bool matched = true;
    for (int i = 0; i < out_height * out_width; i++) {
        if (fabs(h_out[i] - expected_out[i]) > 1e-5f) {
            matched = false;
            break;
        }
    }

    std::cout << (matched ? "Success" : "Failure") << std::endl;

    free(h_in); free(h_out);
    cudaFree(d_in); cudaFree(d_out);
    free(expected_out);
}
