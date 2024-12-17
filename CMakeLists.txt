cmake_minimum_required(VERSION 3.18)
project(CUDAExample LANGUAGES CXX CUDA)


# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Specify the CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR})
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include_directories(include)

# Add the source files
file(GLOB SOURCES src/*.cpp src/*.cu)

# Enable CUDA as a required language
enable_language(CUDA)

# Specify the executable target
add_executable(Basic.app examples/Basic.cpp ${SOURCES})
add_executable(Tile.app examples/Tile.cpp ${SOURCES})

# Set the target properties for CUDA compilation
set_target_properties(Basic.app PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
set_target_properties(Tile.app PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Link CUDA libraries
find_package(CUDAToolkit REQUIRED)
target_link_libraries(Basic.app PRIVATE CUDA::cudart)
target_link_libraries(Tile.app PRIVATE CUDA::cudart)
