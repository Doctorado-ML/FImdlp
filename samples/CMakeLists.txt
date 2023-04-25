cmake_minimum_required(VERSION 3.20)
project(sample)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

add_executable(sample sample.cpp ../src/cppmdlp/tests/ArffFiles.cpp ../src/cppmdlp/Metrics.cpp ../src/cppmdlp/CPPFImdlp.cpp)
