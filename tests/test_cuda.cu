#include <gtest/gtest.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(stmt) ASSERT_EQ((stmt), cudaSuccess)

// __global__ void incrementKernel(int* data) {
//     int idx = threadIdx.x;
//     data[idx] += 1;
// }


TEST(CUDA_Runtime, EventTiming) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    EXPECT_GE(ms, 0.0f);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

TEST(CUDA_Runtime, MallocMemcpyMemsetFree) {
    const size_t n = 100;
    int *d_ptr;
    CUDA_CHECK(cudaMalloc(&d_ptr, n * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_ptr, 0, n * sizeof(int)));
    int h_data[n] = {1};
    CUDA_CHECK(cudaMemcpy(d_ptr, h_data, sizeof(h_data), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaFree(d_ptr));
}

TEST(CUDA_Runtime, MallocHostFreeHost) {
    int* h_ptr = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_ptr, 100 * sizeof(int)));
    ASSERT_NE(h_ptr, nullptr);
    CUDA_CHECK(cudaFreeHost(h_ptr));
}

TEST(CUDA_Runtime, MemcpyAsyncWithStream) {
    const int n = 10;
    int h_src[n] = {1,2,3,4,5,6,7,8,9,10};
    int h_dst[n] = {0};
    int *d_ptr;
    cudaStream_t stream;
    CUDA_CHECK(cudaMalloc(&d_ptr, sizeof(h_src)));
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaMemcpyAsync(d_ptr, h_src, sizeof(h_src), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(h_dst, d_ptr, sizeof(h_dst), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (int i = 0; i < n; ++i) EXPECT_EQ(h_src[i], h_dst[i]);
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_ptr));
}

TEST(CUDA_Runtime, StreamCreateWithFlags) {
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamDestroy(stream));
}

TEST(CUDA_Runtime, DeviceInfo) {
    int count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    ASSERT_GT(count, 0);
    CUDA_CHECK(cudaSetDevice(0));
    int current = -1;
    CUDA_CHECK(cudaGetDevice(&current));
    EXPECT_EQ(current, 0);
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    ASSERT_GT(prop.totalGlobalMem, 0);
}

TEST(CUDA_Runtime, MemGetInfo) {
    size_t freeMem = 0, totalMem = 0;
    CUDA_CHECK(cudaMemGetInfo(&freeMem, &totalMem));
    ASSERT_GT(freeMem, 0);
    ASSERT_GT(totalMem, 0);
}

TEST(CUDA_Runtime, ErrorQuerying) {
    cudaError_t peek = cudaPeekAtLastError();
    EXPECT_EQ(peek, cudaSuccess);
    cudaError_t last = cudaGetLastError();
    EXPECT_EQ(last, cudaSuccess);
    const char* errStr = cudaGetErrorString(last);
    ASSERT_NE(errStr, nullptr);
}

__global__ void addOneKernel(int* data) {
    int idx = threadIdx.x;
    data[idx] += 1;
}

TEST(CUDA_Runtime, LaunchKernel) {
    const int N = 4;
    int h_data[N] = {0, 1, 2, 3};
    int* d_data = nullptr;

    CUDA_CHECK(cudaMalloc(&d_data, sizeof(h_data)));
    CUDA_CHECK(cudaMemcpy(d_data, h_data, sizeof(h_data), cudaMemcpyHostToDevice));

    // Launch the kernel using the standard CUDA kernel call syntax
    addOneKernel<<<1, N>>>(d_data);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_data, d_data, sizeof(h_data), cudaMemcpyDeviceToHost));
    EXPECT_EQ(h_data[0], 1);
    EXPECT_EQ(h_data[3], 4);

    CUDA_CHECK(cudaFree(d_data));
}



TEST(CUDA_Runtime, EventCreateDestroy) {
    cudaEvent_t evt;
    CUDA_CHECK(cudaEventCreate(&evt));
    CUDA_CHECK(cudaEventDestroy(evt));
}

TEST(CUDA_Runtime, EventRecordAndSync) {
    cudaEvent_t evt;
    CUDA_CHECK(cudaEventCreate(&evt));
    CUDA_CHECK(cudaEventRecord(evt));
    CUDA_CHECK(cudaEventSynchronize(evt));
    CUDA_CHECK(cudaEventDestroy(evt));
}

TEST(CUDA_Runtime, GetSetDevice) {
    int devCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devCount));
    ASSERT_GT(devCount, 0);
    CUDA_CHECK(cudaSetDevice(0));
    int current;
    CUDA_CHECK(cudaGetDevice(&current));
    EXPECT_EQ(current, 0);
}

TEST(CUDA_Runtime, GetDeviceProperties) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    EXPECT_GT(prop.totalGlobalMem, 0);
}

TEST(CUDA_Runtime, GetErrorInfo) {
    // Force an error by launching an invalid kernel configuration
    cudaError_t err = cudaMemcpy(nullptr, nullptr, 10, cudaMemcpyHostToDevice); // Invalid
    EXPECT_NE(err, cudaSuccess);

    // Check GetLastError and GetErrorString
    cudaError_t last = cudaGetLastError();
    EXPECT_EQ(last, err);

    const char* errStr = cudaGetErrorString(last);
    ASSERT_NE(errStr, nullptr);
    std::cout << "Captured CUDA error: " << errStr << std::endl;

    // Clear error
    EXPECT_EQ(cudaGetLastError(), cudaSuccess);
}
