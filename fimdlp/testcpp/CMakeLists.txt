cmake_minimum_required(VERSION 3.14)
project(FImdlp)

# GoogleTest requires at least C++14
set(CMAKE_CXX_STANDARD 14)

include(FetchContent)
include_directories(
        /Users/rmontanana/Code/FImdlp/fimdlp/testcpp/build/_deps/googletest-src
        /Users/rmontanana/Code/FImdlp/fimdlp/testcpp/build/_deps/googletest-src/googletest
        /Users/rmontanana/Code/FImdlp/fimdlp/testcpp/build/_deps/googletest-src/googletest/include
        /Users/rmontanana/Code/FImdlp/fimdlp/testcpp/build/_deps/googletest-src/googletest/src
        /Users/rmontanana/Code/FImdlp/fimdlp/testcpp/build/_deps/googletest-src/googletest/test/
        /Users/rmontanana/Code/FImdlp/fimdlp/testcpp/build/_deps/googletest-src/googlemock
        /Users/rmontanana/Code/FImdlp/fimdlp/testcpp/build/_deps/googletest-src/googlemock/include
        /Users/rmontanana/Code/FImdlp/fimdlp/testcpp/build/_deps/googletest-src/googlemock/test
        /Library/Frameworks/Python.framework/Versions/Current/
)
#include_directories(/Users/rmontanana/Code/FImdlp/fimdlp/testcpp/googletest-release-1.12.1/googletest/include /Users/rmontanana/Code/FImdlp/fimdlp/testcpp/googletest-release-1.12.1/googletest)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(Metrics_unittest ../Metrics.cpp Metrics_unittest.cc)
add_executable(FImdlp_unittest ../CPPFImdlp.cpp ../Metrics.cpp FImdlp_unittest.cc)
target_link_libraries(Metrics_unittest GTest::gtest_main)
target_link_libraries(FImdlp_unittest GTest::gtest_main)


include(GoogleTest)
gtest_discover_tests(Metrics_unittest)
gtest_discover_tests(FImdlp_unittest)
