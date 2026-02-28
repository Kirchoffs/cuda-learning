#include <cuda_runtime.h>
#include <random>
#include <math.h>

__global__ void softmax_naive_kernel(const float *in, float *out, int num_rows, int num_cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < num_rows && col < num_cols) {
        float max_val = in[row * num_cols + col];
        for (int col_idx = 1; col_idx < num_cols; col_idx++) {
            max_val = max(max_val, in[row * num_cols + col_idx]);
        }

        float sum_exp = 0.0f;
        for (int col_idx = 0; col_idx < num_cols; col_idx++) {
            sum_exp += expf(in[row * num_cols + col_idx] - max_val);
        }

        out[row * num_cols + col] = expf(in[row * num_cols + col] - max_val) / sum_exp;
    }
}

void softmax_cpu(const float *in, float *out, int num_rows, int num_cols) {
    for (int row = 0; row < num_rows; row++) {
        float max_val = in[row * num_cols];

        for (int col = 0; col < num_cols; col++) {
            if (in[row * num_cols + col] > max_val) {
                max_val = in[row * num_cols + col];
            }
        }

        float sum_exp = 0;
        for (int col = 0; col < num_cols; col++) {
            sum_exp += expf(in[row * num_cols + col] - max_val);
        }

        for (int col = 0; col < num_cols; col++) {
            out[row * num_cols + col] = expf(in[row * num_cols + col] - max_val) / sum_exp;
        }
    }
}

int main() {
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dist_real(0.0, 1.0);

    int num_rows = 512, num_cols = 512;
    size_t size = num_rows * num_cols * sizeof(float);
    
    float *h_in = (float *)malloc(size);
    float *h_out = (float *)malloc(size);
    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < num_cols; j++) {
            h_in[i * num_cols + j] = dist_real(gen);
        }
    }

    float *d_in, *d_out;
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);

    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    dim3 threads_per_block(32, 32);
    dim3 blocks_per_grid(
        (num_cols + threads_per_block.x - 1) / threads_per_block.x,
        (num_rows + threads_per_block.y - 1) / threads_per_block.y
    );

    softmax_naive_kernel<<<blocks_per_grid, threads_per_block>>>(d_in, d_out, num_rows, num_cols);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);

    float *cpu_h_out = (float *)malloc(size);
    softmax_cpu(h_in, cpu_h_out, num_rows, num_cols);

    bool is_correct = true;
    float tolerance = 1e-5f;
    int error_count = 0;
    for (int i = 0; i < num_rows * num_cols; i++) {
        float diff = fabs(h_out[i] - cpu_h_out[i]);
        
        if (diff > tolerance) {
            is_correct = false;

            if (error_count <= 15) {
                printf("Mismatch at index %d: GPU=%f, CPU=%f, diff=%f\n", i, h_out[i], cpu_h_out[i], diff);
            }

            error_count++;
        }
    }

    if (is_correct) {
        printf("PASS: GPU results match CPU results\n");
    } else {
        printf("FAIL: GPU results differ from CPU results\n");
    }

    free(h_in);
    free(h_out);
    free(cpu_h_out);
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}
