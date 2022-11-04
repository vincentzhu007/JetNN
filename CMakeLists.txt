cmake_minimum_required(VERSION 3.10)

project(JetNN LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)

add_subdirectory(3rd_party)

add_subdirectory(src)

enable_testing()
add_subdirectory(tests)