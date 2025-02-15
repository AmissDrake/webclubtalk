#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>

#define TILE_WIDTH 16

__global__ void matrixMulKernel(float *A, float *B, float *C, int width)
{
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (Row < width && Col < width) {
        float Cvalue = 0;
        for (int k = 0; k < width; ++k) {
            Cvalue += A[Row * width + k] * B[k * width + Col];
        }
        C[Row * width + Col] = Cvalue;
    }
}


int main()
{
    int width = 1800;
    size_t size = width * width * sizeof(float);

    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;

    // Allocate host memory
    h_A = (float *)malloc(size);
    h_B = (float *)malloc(size);
    h_C = (float *)malloc(size);

    // Initialize host arrays
    for (int i = 0; i < width * width; i++)
    {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // Allocate device memory
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copy input data from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (width + threadsPerBlock.y - 1) / threadsPerBlock.y);
    matrixMulKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, width);

    // Copy result back to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Verify result
    printf("First element of result: %f\n", h_C[0]);
    printf("Expected result: %f\n", 1.0f * 2.0f * width);  // Should be 3600.0f

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
