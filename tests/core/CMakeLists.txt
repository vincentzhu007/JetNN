add_executable(jetnn-core-test)
target_sources(jetnn-core-test PRIVATE
        tensor_test.cpp)
target_link_libraries(jetnn-core-test PRIVATE gtest gtest_main)
target_link_libraries(jetnn-core-test PRIVATE jetnn-core)
