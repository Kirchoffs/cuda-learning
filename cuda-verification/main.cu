#include <cuda_runtime.h>
#include <iostream>

int main() {
    cudaDeviceProp device_prop;
    int dev = 0;

    cudaGetDeviceProperties(&device_prop, dev);

    std::cout << "Device " << dev << ": " << device_prop.name << std::endl;
    
    std::cout << " CUDA Capability Major/Minor version number: " 
              << device_prop.major << "." << device_prop.minor << std::endl;
    
    std::cout << " Total amount of shared memory per block: " 
              << device_prop.sharedMemPerBlock << " bytes" << std::endl;
    
    std::cout << " Maximum number of threads per block: " 
              << device_prop.maxThreadsPerBlock << std::endl;

    return 0;
}
