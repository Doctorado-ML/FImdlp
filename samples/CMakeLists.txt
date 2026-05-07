cmake_minimum_required(VERSION 3.20)
project(sample)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Debug)
include_directories(../src/cpp)

add_executable(sample sample.cpp ../src/cpp/ArffFiles.cpp ../src/cpp/Metrics.cpp ../src/cpp/CPPFImdlp.cpp)
