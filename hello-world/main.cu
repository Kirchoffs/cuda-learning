#include <cuda_runtime.h>
#include <iostream>

__global__ void hello() {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    printf("Hello Thread %d\n", tid);
}

int main() {
    hello<<<1, 10>>>();
    cudaDeviceSynchronize();
    return 0;
}
