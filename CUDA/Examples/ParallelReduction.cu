#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void reduce(float *input, float *output, int N)
{
    __shared__ float shared_data[256];
    unsigned int threadidx = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    shared_data[threadidx] = (i < N) ? input[i] : 0;
    __syncthreads();

    for (unsigned int s = blockDim.x/2; s > 0; s >>= 1)
    {
        if (threadidx < s)
        {
            shared_data[threadidx] += shared_data[threadidx + s];
        }
        __syncthreads();
    }

    if (threadidx == 0) output[blockIdx.x] = shared_data[0];
    __syncthreads();
}

int main()
{
    const int N = 1000000000; // Input array size here
    float *h_input, *h_output;
    float *d_input, *d_output;
    const int Reduced_Size = (N/256)+1;

    cudaEvent_t start, stop, start_total, stop_total;
    float gpu_time = 0.0f;
    float gpu_time_total = 0.0f;

    // Set up timing
    cudaEventCreate(&start_total);
    cudaEventCreate(&stop_total);
    cudaEventRecord(start_total, 0);

    // Allocate and initialize host memory
    h_input = new float[N];
    h_output = new float[Reduced_Size];
    for (int i = 0; i < N; i++) h_input[i] = 1.0f;

    // Allocate device memory
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, (Reduced_Size) * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    // Set up timing
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    

    // Launch kernel
    reduce<<<Reduced_Size, 256>>>(d_input, d_output, N);

    // Record time
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);

    // Copy result back to host
    cudaMemcpy(h_output, d_output, (Reduced_Size) * sizeof(float), cudaMemcpyDeviceToHost);

    // Record time
    cudaEventRecord(stop_total, 0);
    cudaEventSynchronize(stop_total);
    cudaEventElapsedTime(&gpu_time_total, start_total, stop_total);

    // Compute final sum on CPU
    float sum = 0;
    for (int i = 0; i < Reduced_Size; i++) sum += h_output[i];

    printf("Sum: %f\n Calculation of the sum took %f seconds.\n", sum, gpu_time / 1000.0f);
    printf("Calculation of the sum took %f seconds including memory operations.\n", sum, gpu_time_total / 1000.0f);

    // Clean up
    delete[] h_input;
    delete[] h_output;
    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}
