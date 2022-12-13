cmake_minimum_required(VERSION 3.14)
project(FImdlp)

# GoogleTest requires at least C++14
set(CMAKE_CXX_STANDARD 14)
include(FetchContent)

include_directories(${GTEST_INCLUDE_DIRS})

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(Metrics_unittest ../Metrics.cpp Metrics_unittest.cpp)
add_executable(FImdlp_unittest ../CPPFImdlp.cpp ../Metrics.cpp FImdlp_unittest.cpp)
target_link_libraries(Metrics_unittest GTest::gtest_main)
target_link_libraries(FImdlp_unittest GTest::gtest_main)
target_compile_options(Metrics_unittest PRIVATE --coverage)
target_compile_options(FImdlp_unittest PRIVATE --coverage)
target_link_options(Metrics_unittest PRIVATE --coverage)
target_link_options(FImdlp_unittest PRIVATE --coverage)

include(GoogleTest)
gtest_discover_tests(Metrics_unittest)
gtest_discover_tests(FImdlp_unittest)

