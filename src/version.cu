#include <cuda_runtime.h>

#include <iostream>

#include "version.hpp"

void printVersion() {
  int run_version, driver_version;
  std::cout << "Return Code Runtime Version: " << cudaRuntimeGetVersion(&run_version) << "\n";
  std::cout << "Return Code Driver Version: " << cudaDriverGetVersion(&driver_version) << "\n";
  std::cout << "Runtime Version: " << run_version << "\n";
  std::cout << "Driver Version: " << driver_version << "\n";

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "CUDA kernel launch error: " << cudaGetErrorString(err) << std::endl;
  }
}
