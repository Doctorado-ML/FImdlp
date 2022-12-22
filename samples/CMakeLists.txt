cmake_minimum_required(VERSION 3.20)
project(main)

set(CMAKE_CXX_STANDARD 14)

add_executable(sample sample.cpp ../src/cppmdlp/tests/ArffFiles.cpp ../src/cppmdlp/Metrics.cpp ../src/cppmdlp/CPPFImdlp.cpp)
