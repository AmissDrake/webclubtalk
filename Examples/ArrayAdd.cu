#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <cstdlib> // For randomizing input array
#include <time.h> // For measuring runtime

#define MAX_VALUE 10 // Define max value of array member

__global__ void ArrayAdd(int* A, int* B, int* C, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) {
		C[i] = A[i] + B[i];
	}
	return;
}

int main() {
    int N = 1000000000; // Define array size here
    int *arr1, *arr2, *result;
    int *gpu1, *gpu2, *gpuresult;

    clock_t start, start_actual, end, end_actual;
    double total_program_runtime, gpu_time_used;
    
    // Seed random numbr generator
    srand(time(NULL));

    // Dynamically allocate memory for the arrays
    arr1 = (int*)malloc(N * sizeof(int));
    arr2 = (int*)malloc(N * sizeof(int));
    result = (int*)malloc(N * sizeof(int));

    if (arr1 == NULL || arr2 == NULL || result == NULL) {
        printf("Memory allocation failed\n");
        return 1;
    }

    // Generate random numbers for the arrays
    for (int i = 0; i < N; i++) {
        arr1[i] = rand() % MAX_VALUE;
        arr2[i] = rand() % MAX_VALUE;
    }

    // Start timing
    start = clock();

    // Allocating GPU memory
    cudaMalloc(&gpu1, sizeof(arr1));
    cudaMalloc(&gpu2, sizeof(arr2));
    cudaMalloc(&gpuresult, sizeof(result));

    // Copying the arrays into the GPU memory
    cudaMemcpy(gpu1, arr1, sizeof(arr1), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu2, arr2, sizeof(arr2), cudaMemcpyHostToDevice);

    start_actual = clock();

    // Calculating required no of Blocks
    int THREADS = 1024;
    int BLOCKS = (N+THREADS-1)/THREADS;

    // Calling the device function
    ArrayAdd <<<BLOCKS,THREADS>>> (gpu1,gpu2,gpuresult,N);
    cudaDeviceSynchronize();

    end_actual = clock();

    // Copying the sum back to Host memory
    cudaMemcpy(result, gpuresult, sizeof(gpuresult), cudaMemcpyDeviceToHost);

    end = clock();
    
    total_program_runtime = ((double) (end - start)) / CLOCKS_PER_SEC;
    gpu_time_used = ((double) (end_actual - start_actual)) / CLOCKS_PER_SEC;

    std::cout<< "The program to add 2 arrays of size "<<N<<" took "<<total_program_runtime<<" seconds."<<std::endl;
    std::cout<< "The program to add 2 arrays of size "<<N<<" actually took "<<gpu_time_used<<" seconds."<<std::endl;

    // Freeing memory
    cudaFree(gpu1);
    cudaFree(gpu2);
    cudaFree(gpuresult);
    free(arr1);
    free(arr2);
    free(result);

}