
file(GLOB JETNN_UT_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
add_executable(jetnn-test ${JETNN_UT_SRC})
target_link_libraries(jetnn-test PRIVATE gtest gtest_main)
target_link_libraries(jetnn-test PRIVATE jetnn-core)
include(GoogleTest)
gtest_discover_tests(jetnn-test)