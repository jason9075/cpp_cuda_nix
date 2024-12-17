#include <cuda_runtime.h>

#include <iostream>

#include "matrixMul.hpp"

const int TILE_WIDTH = 16;

// CUDA kernel for vector addition
__global__ void matrixMulKernel(const float* A, const float* B, float* C, int M, int N, int K) {
  __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
  __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];

  int blockX = blockIdx.x;
  int blockY = blockIdx.y;

  int threadX = threadIdx.x;
  int threadY = threadIdx.y;

  int row = blockX * TILE_WIDTH + threadX;
  int col = blockY * TILE_WIDTH + threadY;

  float Cvalue = 0.0;
  for (int i = 0; i < (N + TILE_WIDTH - 1) / TILE_WIDTH; i++) {
    if (row < M && i * TILE_WIDTH + threadY < N) {
      tileA[threadX][threadY] = A[row * N + i * TILE_WIDTH + threadY];
    } else {
      tileA[threadX][threadY] = 0.0;
    }

    if (i * TILE_WIDTH + threadX < N && col < K) {
      tileB[threadX][threadY] = B[(i * TILE_WIDTH + threadX) * K + col];
    } else {
      tileB[threadX][threadY] = 0.0;
    }

    __syncthreads();

    for (int n = 0; n < TILE_WIDTH; n++) {
      Cvalue += tileA[threadX][n] * tileB[n][threadY];
    }

    __syncthreads();
  }

  if (row < M && col < K) {
    C[row * K + col] = Cvalue;
  }
}

void matrixMul(const float* matrixA, const float* matrixB, float* matrixC, int M, int N, int K) {
  size_t sizeA = M * N * sizeof(float);
  size_t sizeB = N * K * sizeof(float);
  size_t sizeC = M * K * sizeof(float);

  // Allocate device memory
  float* d_A;
  float* d_B;
  float* d_C;
  cudaMalloc((void**)&d_A, sizeA);
  cudaMalloc((void**)&d_B, sizeB);
  cudaMalloc((void**)&d_C, sizeC);

  // Copy data from host to device
  cudaMemcpy(d_A, matrixA, sizeA, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, matrixB, sizeB, cudaMemcpyHostToDevice);

  // Launch the kernel
  dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
  dim3 blocksPerGrid((M + TILE_WIDTH - 1) / TILE_WIDTH, (K + TILE_WIDTH - 1) / TILE_WIDTH);
  matrixMulKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "CUDA kernel launch error: " << cudaGetErrorString(err) << std::endl;
  }
  // Synchronize device
  cudaDeviceSynchronize();

  // Copy result back to host
  cudaMemcpy(matrixC, d_C, sizeC, cudaMemcpyDeviceToHost);

  // Free device memory
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}
