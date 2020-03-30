cmake_minimum_required(VERSION 3.15)
project(LP_simple_IPM)

set(CMAKE_CXX_STANDARD 14)

add_executable(LP_simple_IPM main.cpp)
target_include_directories(LP_simple_IPM PRIVATE include)