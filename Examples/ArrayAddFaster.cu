#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <iostream>
#include <cstdlib>
#include <time.h>

#define MAX_VALUE 10

__global__ void initializeArray(int* arr, int N, unsigned long long seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        curandState state;
        curand_init(seed, i, 0, &state);
        arr[i] = curand(&state) % MAX_VALUE;
    }
}

__global__ void ArrayAdd(int* A, int* B, int* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    int N = 1000000000; // Define array size here
    int *result;
    int *gpu1, *gpu2, *gpuresult;

    clock_t start, end;
    double total_program_runtime;
    
    // Start timing
    start = clock();

    // Allocating GPU memory
    cudaMalloc(&gpu1, N * sizeof(int));
    cudaMalloc(&gpu2, N * sizeof(int));
    cudaMalloc(&gpuresult, N * sizeof(int));

    // Calculating required no of Blocks
    int THREADS = 1024;
    int BLOCKS = (N + THREADS - 1) / THREADS;

    // Initialize arrays on GPU
    unsigned long long seed = time(NULL);
    initializeArray<<<BLOCKS, THREADS>>>(gpu1, N, seed);
    initializeArray<<<BLOCKS, THREADS>>>(gpu2, N, seed + 1);  // Different seed for second array

    // Calling the device function to add arrays
    ArrayAdd<<<BLOCKS, THREADS>>>(gpu1, gpu2, gpuresult, N);
    
    // Allocate host memory for result
    result = (int*)malloc(N * sizeof(int));

    // Copying the sum back to Host memory
    cudaMemcpy(result, gpuresult, N * sizeof(int), cudaMemcpyDeviceToHost);

    end = clock();
    
    total_program_runtime = ((double) (end - start)) / CLOCKS_PER_SEC;

    std::cout << "The program to add 2 arrays of size " << N << " took " << total_program_runtime << " seconds." << std::endl;

    // Free GPU memory
    cudaFree(gpu1);
    cudaFree(gpu2);
    cudaFree(gpuresult);

    // Free CPU memory
    free(result);

    return 0;
}
