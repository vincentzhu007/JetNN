add_library(jetnn-core)
file(GLOB JETNN_CORE_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
target_sources(jetnn-core PRIVATE ${JETNN_CORE_SRC})
target_include_directories(jetnn-core PUBLIC "${CMAKE_SOURCE_DIR}/src")