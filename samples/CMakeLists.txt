cmake_minimum_required(VERSION 3.20)
project(main)

set(CMAKE_CXX_STANDARD 14)

add_executable(sample sample.cpp ArffFiles.cpp ../src/fimdlp/cppmdlp/Metrics.cpp ../src/fimdlp/cppmdlp/CPPFImdlp.cpp)
