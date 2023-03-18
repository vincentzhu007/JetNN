cmake_minimum_required(VERSION 3.10)

project(JetNN LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)

option(JETNN_ENABLE_TEST "Enable JetNN test" ON)

message(STATUS "JETNN_ENABLE_TEST = ${JETNN_ENABLE_TEST}")

add_subdirectory(third_party)

add_subdirectory(src)

if (JETNN_ENABLE_TEST)
    # build tests
    enable_testing()
    add_subdirectory(tests)
endif()