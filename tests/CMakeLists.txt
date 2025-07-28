enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)

# Fetch GTest if not found
find_package(GTest QUIET)
if (NOT GTest_FOUND)
    message(STATUS "GTest not found, fetching with FetchContent...")
    include(FetchContent)
    FetchContent_Declare(
      googletest
      URL https://github.com/google/googletest/releases/download/v1.17.0/googletest-1.17.0.tar.gz
    )
    FetchContent_MakeAvailable(googletest)
endif()

# Add all .cu test files manually or via GLOB
set(TEST_SOURCES
    #test_cudart.cu
    #test_cublas.cu
    #test_curand.cu
    test_cudnn.cu
    #test_cuda.cu
)

# Create an output directory for tests
set(TEST_BIN_DIR ${CMAKE_BINARY_DIR}/tests)
file(MAKE_DIRECTORY ${TEST_BIN_DIR})

# Create each test target individually
foreach(TEST_SRC ${TEST_SOURCES})
    get_filename_component(TEST_NAME ${TEST_SRC} NAME_WE)
    
    add_executable(${TEST_NAME} ${TEST_SRC})
    set_target_properties(${TEST_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        RUNTIME_OUTPUT_DIRECTORY ${TEST_BIN_DIR}
    )

    target_include_directories(${TEST_NAME} PRIVATE
        ${GTEST_INCLUDE_DIRS}
    )

    target_link_libraries(${TEST_NAME} PRIVATE
        GTest::GTest
        GTest::Main
        cuda
        curand
        cudart
        cublas
        cufft
        cudnn
        cusolver
        cusparse
        nvrtc
    )

    add_test(NAME ${TEST_NAME} COMMAND ${TEST_BIN_DIR}/${TEST_NAME})
endforeach()
