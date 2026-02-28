#include <cuda_runtime.h>
#include <iostream>

struct Point {
    float x;
    float z;
    float y;
};

__global__ void calculateEuclideanDistanceKernel(Point *line_a, Point *line_b, float *distances, int num_points) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < num_points) {
        float dx = line_a[idx].x - line_b[idx].x;
        float dy = line_a[idx].y - line_b[idx].y;
        float dz = line_a[idx].z - line_b[idx].z;
        
        distances[idx] = sqrtf(dx * dx + dy * dy + dz * dz);
        
    }
}

int main() {
    int num_points = 99'999'999;
    size_t size_points = num_points * sizeof(Point);
    size_t size_distances = num_points * sizeof(float);

    Point *h_line_a = (Point *)malloc(size_points);
    Point *h_line_b = (Point *)malloc(size_points);
    float *h_distances = (float *)malloc(size_distances);
    for (int i = 0; i < num_points; i++) {
        h_line_a[i].x = i * 1.0f;
        h_line_a[i].y = i * 2.0f;
        h_line_a[i].z = i * 3.0f;

        h_line_b[i].x = i * 1.5f;
        h_line_b[i].y = i * 2.5f;
        h_line_b[i].z = i * 3.5f;
    }

    Point *d_line_a;
    Point *d_line_b;
    float *d_distances;
    cudaMalloc((void **)&d_line_a, size_points);
    cudaMalloc((void **)&d_line_b, size_points);
    cudaMalloc((void **)&d_distances, size_distances);

    cudaMemcpy(d_line_a, h_line_a, size_points, cudaMemcpyHostToDevice);
    cudaMemcpy(d_line_b, h_line_b, size_points, cudaMemcpyHostToDevice);

    int block_size = 1024;
    int grid_size = (num_points + block_size - 1) / block_size;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    calculateEuclideanDistanceKernel<<<grid_size, block_size>>>(d_line_a, d_line_b, d_distances, num_points);

    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float gpu_duration = 0;
    cudaEventElapsedTime(&gpu_duration, start, stop);

    std::cout<< std::fixed << "Time taken: " << gpu_duration << " ms" << std::endl;

    cudaMemcpy(h_distances, d_distances, size_distances, cudaMemcpyDeviceToHost);

    cudaFree(d_line_a);
    cudaFree(d_line_b);
    cudaFree(d_distances);

    free(h_line_a);
    free(h_line_b);
    free(h_distances);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
