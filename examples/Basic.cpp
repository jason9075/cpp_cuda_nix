#include <iostream>

#include "vector.hpp"
#include "version.hpp"

int main() {
  printVersion();
  const int N = 1000;
  size_t size = N * sizeof(float);

  // Allocate and initialize host memory
  float* h_A = new float[N];
  float* h_B = new float[N];
  float* h_C = new float[N];

  for (int i = 0; i < N; i++) {
    h_A[i] = static_cast<float>(i);
    h_B[i] = static_cast<float>(i * 2);
  }

  // Call the vector addition function
  vectorAdd(h_A, h_B, h_C, N);

  // Display the first 10 results
  for (int i = 0; i < 10; i++) {
    std::cout << h_A[i] << " + " << h_B[i] << " = " << h_C[i] << std::endl;
  }

  // Free host memory
  delete[] h_A;
  delete[] h_B;
  delete[] h_C;

  return 0;
}
