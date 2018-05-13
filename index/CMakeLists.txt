cmake_minimum_required(VERSION 3.9)
project(hnsw_index)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-std=c++17 -O0 -fopenmp")

include_directories(PYTHON_INCLUDE_DIR)
include_directories(./include)

add_executable(tests hnsw/tests.cpp)

set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
FIND_PACKAGE(Boost REQUIRED COMPONENTS system)

if(Boost_FOUND)
    message(STATUS "Boost Found!")
    message(STATUS "Boost is ${BOOST_ROOT}")
    message(STATUS "Boost directories found at ${Boost_INCLUDE_DIRS}")
    message(STATUS "Boost libraries found at ${Boost_LIBRARY_DIRS}")
    message(STATUS "Boost libraries prefix is ${Boost_LIB_PREFIX}")
    message(STATUS "Boost component libraries to be linked are ${Boost_LIBRARIES}")
    message(STATUS "Boost version found is ${Boost_VERSION}")

    include_directories(${Boost_INCLUDE_DIRS})
    target_link_libraries(tests ${Boost_LIBRARIES})
endif()
