#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

__global__ void check_prime_kernel(long long start, long long end) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    long long num = start + (tid * 2);

    bool is_prime = true;

    if (num <= 1) {
        is_prime = false;
        return;
    }

    if (num == 2) {
        is_prime = true;
        return;
    }

    if (num % 2 == 0) {
        is_prime = false;
        return;
    }

    if (num > end) {
        return;
    }

    
    for (long long i = 3; i * i <= num; i += 2) {
        if (num % i == 0) {
            is_prime = false;
            break;
        }
    }
}

bool check_prime_cpu(long long num) {
    if (num <= 1) {
        return false;
    }

    if (num == 2) {
        return true;
    }

    if (num % 2 == 0) {
        return false;
    }

    for (long long i = 3; i * i <= num; i += 2) {
        if (num % i == 0) {
            return false;
        }
    }

    return true;
}


int main() {
    long long start =  199'999LL;
    long long end   =  399'999LL;

    int threads_per_block = 256;
    int total_numbers = (end - start) / 2 + 1;
    int blocks_per_grid = (total_numbers + threads_per_block - 1) / threads_per_block;

    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);

    cudaEventRecord(start_event, 0);

    check_prime_kernel<<<blocks_per_grid, threads_per_block>>>(start, end);

    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);

    float gpu_duration = 0;
    cudaEventElapsedTime(&gpu_duration, start_event, stop_event);

    std::cout << "Time taken on GPU: " << gpu_duration << " ms" << std::endl;

    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);

    auto startTime = std::chrono::high_resolution_clock::now();

    for (long long num = start; num <= end; num += 2) {
        check_prime_cpu(num);
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_duration = endTime - startTime;

    std::cout << "Time taken on CPU: " << std::fixed << cpu_duration.count() << " ms" << std::endl;
    std::cout << "speed up : " << cpu_duration.count() / gpu_duration << std::endl;

    return 0;
}
