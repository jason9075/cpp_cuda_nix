/*
Run this file

nvcc single_test.cu -o test.app \
    -I/nix/store/ca1xhh6rxcx0a23qnsla0gsg44vwgwl4-cuda-merged-12.2/include \
    -L/nix/store/ca1xhh6rxcx0a23qnsla0gsg44vwgwl4-cuda-merged-12.2/lib \
    -lcudart && ./test.app
*/
#include <cuda_runtime.h>

#include <iostream>

int main() {
  int run_version, driver_version;
  std::cout << "Return Code Runtime Version: " << cudaRuntimeGetVersion(&run_version) << "\n";
  std::cout << "Return Code Driver Version: " << cudaDriverGetVersion(&driver_version) << "\n";
  std::cout << "Runtime Version: " << run_version << "\n";
  std::cout << "Driver Version: " << driver_version << "\n";

  return 0;
}
