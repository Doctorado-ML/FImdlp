cmake_minimum_required(VERSION 3.24)
project(main)

set(CMAKE_CXX_STANDARD 17)

add_executable(sample sample.cpp ArffFiles.cpp ../Metrics.cpp ../CPPFImdlp.cpp)
