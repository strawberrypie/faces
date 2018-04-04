cmake_minimum_required(VERSION 3.10)
project(hnsw_index)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-std=c++17 -O0 -fopenmp")

include_directories(PYTHON_INCLUDE_DIR)

add_executable(index hnsw/Index.cpp)
#add_executable(bindings python_bindings/bindings.cpp)