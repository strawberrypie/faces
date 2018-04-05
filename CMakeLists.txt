cmake_minimum_required(VERSION 3.10)
project(hnsw_index)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-std=c++17 -O0 -fopenmp")

include_directories(PYTHON_INCLUDE_DIR)

add_executable(index hnsw/index.cpp)

set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
FIND_PACKAGE(Boost REQUIRED COMPONENTS system)

if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
    target_link_libraries(index ${Boost_LIBRARIES})
endif()