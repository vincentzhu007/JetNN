# Import googletest headers
include_directories(${CMAKE_SOURCE_DIR}/3rd_party/googletest/googletest/include)
include_directories(${CMAKE_SOURCE_DIR}/3rd_party/googletest/googlemock/include/gmock)

include_directories(${CMAKE_SOURCE_DIR}/src)

add_subdirectory(core)
