enable_testing()

find_package(GTest REQUIRED)

add_executable(jetnn-test)
target_sources(jetnn-test PRIVATE tensor_test.cpp)
target_link_libraries(jetnn-test PRIVATE GTest::gtest GTest::gtest_main)
target_link_libraries(jetnn-test PRIVATE jetnn-core)

include(GoogleTest)
gtest_discover_tests(jetnn-test)