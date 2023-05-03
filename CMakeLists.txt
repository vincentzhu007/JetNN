cmake_minimum_required(VERSION 3.10)

project(JetNN LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)

# JetNN build options.
option(JETNN_ENABLE_TEST "Enable JetNN test" ON)
message(STATUS "JETNN_ENABLE_TEST = ${JETNN_ENABLE_TEST}")

# Here import third party dependencies.
include(third_party/googletest.cmake)

# Source code here.
add_subdirectory(src)

# Build tests
if (JETNN_ENABLE_TEST)
    enable_testing()
    add_subdirectory(tests)
endif()