#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

__global__ void vector_add_kernel(float *A, float *B, float *C, int N) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

void vector_add_cpu(float *A, float *B, float *C, int N) {
    for (int i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    int N = 999'999'999;
    size_t size = N * sizeof(float);

    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);
    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(i * 2);
    }

    float *d_A;
    float *d_B;
    float *d_C;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    int threads_per_block = 1024;
    int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;

    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);

    cudaEventRecord(start_event, 0);
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);

    float gpu_copy_time = 0;
    cudaEventElapsedTime(&gpu_copy_time, start_event, stop_event);

    std::cout<< std::fixed << "Time to copy data to GPU: " << gpu_copy_time << " ms" << std::endl;

    cudaEventRecord(start_event, 0);

    vector_add_kernel<<<blocks_per_grid, threads_per_block>>>(d_A, d_B, d_C, N);

    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);

    float gpu_execution_time = 0;
    cudaEventElapsedTime(&gpu_execution_time, start_event, stop_event);

    std::cout<< std::fixed << "Time to execute on GPU: " << gpu_execution_time << " ms" << std::endl;

    cudaEventRecord(start_event, 0);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);

    float gpu_retrieve_time = 0;
    cudaEventElapsedTime(&gpu_retrieve_time, start_event, stop_event);

    std::cout<< std::fixed << "Time taken to copy results back GPU: " << gpu_retrieve_time << " ms" << std::endl << std::endl;

    float gpu_duration = (gpu_copy_time + gpu_execution_time + gpu_retrieve_time);
    std::cout << "Time taken by GPU: " << gpu_duration << " ms" << std::endl;


    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);


    auto start = std::chrono::high_resolution_clock::now();

    vector_add_cpu(h_A, h_B, h_C, N);

    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_duration = (stop - start);

    std::cout << "Time taken by CPU: " << cpu_duration.count() << " ms" << std::endl;
    std::cout << "========================================== " << std::endl;

    std::cout << "speed up (execution time only): " << cpu_duration.count() / gpu_execution_time << std::endl;
    std::cout << "speed up (GPU total time): " << cpu_duration.count() / gpu_duration << std::endl;


    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
