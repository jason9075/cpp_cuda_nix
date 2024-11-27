#include <cuda_runtime.h>

#include <iostream>

#include "vector.hpp"

// CUDA kernel for vector addition
__global__ void vectorAddKernel(const float* A, const float* B, float* C, int N) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < N) {
    C[i] = A[i] + B[i];
  }
}

// Vector addition function
void vectorAdd(const float* h_A, const float* h_B, float* h_C, int N) {
  size_t size = N * sizeof(float);

  // Allocate device memory
  float* d_A;
  float* d_B;
  float* d_C;
  cudaMalloc((void**)&d_A, size);
  cudaMalloc((void**)&d_B, size);
  cudaMalloc((void**)&d_C, size);

  // Copy data from host to device
  cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

  // Launch the kernel
  int threadsPerBlock = 256;
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
  vectorAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "CUDA kernel launch error: " << cudaGetErrorString(err) << std::endl;
  }
  // Synchronize device
  cudaDeviceSynchronize();

  // Copy result back to host
  cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

  // Free device memory
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}
