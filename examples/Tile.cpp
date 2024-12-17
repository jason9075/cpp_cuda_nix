#include <iostream>

#include "matrixMul.hpp"

int main() {
  const int M = 3;
  const int N = 4;
  const int K = 5;

  // Allocate and initialize host memory
  float* matrixA = new float[M * N];
  float* matrixB = new float[N * K];
  float* matrixC = new float[M * K];

  for (int i = 0; i < M * N; i++) {
    matrixA[i] = static_cast<float>(i);
  }

  for (int i = 0; i < N * K; i++) {
    matrixB[i] = static_cast<float>(i * 2);
  }

  // Call the vector addition function
  matrixMul(matrixA, matrixB, matrixC, M, N, K);

  // Display the results A * B = C
  std::cout << "Matrix A:" << std::endl;
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      std::cout << matrixA[i * N + j] << "\t";
    }
    std::cout << std::endl;
  }

  std::cout << "Matrix B:" << std::endl;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < K; j++) {
      std::cout << matrixB[i * K + j] << "\t";
    }
    std::cout << std::endl;
  }

  std::cout << "Matrix C:" << std::endl;
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < K; j++) {
      std::cout << matrixC[i * K + j] << "\t";
    }
    std::cout << std::endl;
  }

  // Free host memory
  delete[] matrixA;
  delete[] matrixB;
  delete[] matrixC;

  return 0;
}
