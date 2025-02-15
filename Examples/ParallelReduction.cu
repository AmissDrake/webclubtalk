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
}

int main()
{
    const int N = 1000000;
    float *h_input, *h_output;
    float *d_input, *d_output;

    // Allocate and initialize host memory
    h_input = new float[N];
    h_output = new float[N/256];
    for (int i = 0; i < N; i++) h_input[i] = 1.0f;

    // Allocate device memory
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, (N/256) * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    reduce<<<N/256, 256>>>(d_input, d_output, N);

    // Copy result back to host
    cudaMemcpy(h_output, d_output, (N/256) * sizeof(float), cudaMemcpyDeviceToHost);

    // Compute final sum on CPU
    float sum = 0;
    for (int i = 0; i < N/256; i++) sum += h_output[i];

    printf("Sum: %f\n", sum);

    // Clean up
    delete[] h_input;
    delete[] h_output;
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
