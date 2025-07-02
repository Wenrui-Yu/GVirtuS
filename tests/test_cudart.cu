#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda.h> /* cuuint64_t */

#define CUDA_CHECK(err) ASSERT_EQ((err), cudaSuccess)

__device__ int intDeviceVariable = 0;

//Unit tests for only those functions which are missing with openpose integration!

//cudaFree
TEST(cudaRT, MallocAndFree) {
    int* device_ptr = nullptr;

    ASSERT_EQ(cudaMalloc((void**)&device_ptr, 256 * sizeof(int)), cudaSuccess);
    ASSERT_NE(device_ptr, nullptr);

    ASSERT_EQ(cudaFree(device_ptr), cudaSuccess);
}

//StreamCreateWithFlagsAndDestroy
TEST(cudaRT, StreamCreateWithFlagsAndDestroy) {
    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
    ASSERT_NE(stream, nullptr);
    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}


// RegisterFatBinaryIndirectCheck
// __global__ void dummyKernel(int* data) {
//     data[0] = 123;
// }

// TEST(cudaRT, RegisterFatBinaryIndirectCheck) {
//     int* d_ptr;
//     ASSERT_EQ(cudaMalloc(&d_ptr, sizeof(int)), cudaSuccess);
//     dummyKernel<<<1, 1>>>(d_ptr);
//     ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
//     ASSERT_EQ(cudaFree(d_ptr), cudaSuccess);
// }


//cudaEventElapsedTime
TEST(cudaRT, EventElapsedTime) {
    cudaEvent_t start, stop;
    float elapsed_ms = 0.0f;

    ASSERT_EQ(cudaEventCreate(&start), cudaSuccess);
    ASSERT_EQ(cudaEventCreate(&stop), cudaSuccess);

    ASSERT_EQ(cudaEventRecord(start), cudaSuccess);
    ASSERT_EQ(cudaEventRecord(stop), cudaSuccess);
    ASSERT_EQ(cudaEventSynchronize(stop), cudaSuccess);

    ASSERT_EQ(cudaEventElapsedTime(&elapsed_ms, start, stop), cudaSuccess);
    ASSERT_GT(elapsed_ms, 0.0f);

    ASSERT_EQ(cudaEventDestroy(start), cudaSuccess);
    ASSERT_EQ(cudaEventDestroy(stop), cudaSuccess);
}

//__cudaRegisterFunction
// __global__ void dummyKernel(int* out) {
//     *out = 42;
// }

// TEST(cudaRT, RegisterFunctionIndirectCheck) {
//     int* d_ptr;
//     int h_val = 0;

//     ASSERT_EQ(cudaMalloc(&d_ptr, sizeof(int)), cudaSuccess);
//     dummyKernel<<<1, 1>>>(d_ptr);
//     ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
//     ASSERT_EQ(cudaMemcpy(&h_val, d_ptr, sizeof(int), cudaMemcpyDeviceToHost), cudaSuccess);
//     ASSERT_EQ(cudaFree(d_ptr), cudaSuccess);

//     ASSERT_EQ(h_val, 42);
// }


// cudaGetDeviceProperties_v2
TEST(cudaRT, GetDevicePropertiesV2) {
    int device;
    cudaDeviceProp prop;

    ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);
    ASSERT_EQ(cudaGetDeviceProperties(&prop, device), cudaSuccess);
    ASSERT_GT(prop.totalGlobalMem, 0);
}


//cudaMemset
TEST(cudaRT, Memset) {
    const size_t size = 10 * sizeof(int);
    int* d_ptr = nullptr;

    ASSERT_EQ(cudaMalloc(&d_ptr, size), cudaSuccess);
    ASSERT_EQ(cudaMemset(d_ptr, 0xAB, size), cudaSuccess);
    ASSERT_EQ(cudaFree(d_ptr), cudaSuccess);
}

//cudaMemGetInfo
TEST(cudaRT, MemGetInfo) {
    size_t free_mem = 0;
    size_t total_mem = 0;

    ASSERT_EQ(cudaMemGetInfo(&free_mem, &total_mem), cudaSuccess);
    ASSERT_GT(total_mem, 0);
    ASSERT_GT(free_mem, 0);
}

//cudaStreamDestroy
TEST(cudaRT, StreamDestroy) {
    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
    ASSERT_NE(stream, nullptr);
    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

//cudaEventRecord
TEST(cudaRT, EventRecord) {
    cudaEvent_t event;
    ASSERT_EQ(cudaEventCreate(&event), cudaSuccess);
    ASSERT_EQ(cudaEventRecord(event), cudaSuccess);
    ASSERT_EQ(cudaEventSynchronize(event), cudaSuccess);
    ASSERT_EQ(cudaEventDestroy(event), cudaSuccess);
}


//cudaMallocHost
TEST(cudaRT, MallocHost) {
    void* host_ptr = nullptr;
    ASSERT_EQ(cudaMallocHost(&host_ptr, 1024), cudaSuccess);
    ASSERT_NE(host_ptr, nullptr);
    ASSERT_EQ(cudaFreeHost(host_ptr), cudaSuccess);
}

//cudaEventSynchronize
TEST(cudaRT, EventSynchronize) {
    cudaEvent_t event;
    ASSERT_EQ(cudaEventCreate(&event), cudaSuccess);
    ASSERT_EQ(cudaEventRecord(event), cudaSuccess);
    ASSERT_EQ(cudaEventSynchronize(event), cudaSuccess);
    ASSERT_EQ(cudaEventDestroy(event), cudaSuccess);
}


//cudaMemcpyAsync
TEST(cudaRT, MemcpyAsync) {
    const int N = 16;
    int h_src[N], h_dst[N];
    for (int i = 0; i < N; ++i) h_src[i] = i;

    int* d_ptr = nullptr;
    ASSERT_EQ(cudaMalloc(&d_ptr, N * sizeof(int)), cudaSuccess);

    ASSERT_EQ(cudaMemcpyAsync(d_ptr, h_src, N * sizeof(int), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(h_dst, d_ptr, N * sizeof(int), cudaMemcpyDeviceToHost), cudaSuccess);

    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaFree(d_ptr), cudaSuccess);

    for (int i = 0; i < N; ++i) {
        ASSERT_EQ(h_dst[i], h_src[i]);
    }
}

//cudaGetDevice
TEST(cudaRT, GetDevice) {
    int device = -1;
    ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);
    ASSERT_GE(device, 0);
}


//cudaStreamCreate
TEST(cudaRT, StreamCreate) {
    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
    ASSERT_NE(stream, nullptr);
    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}


//cudaMemcpy
TEST(cudaRT, Memcpy) {
    const int N = 16;
    int h_src[N], h_dst[N];
    for (int i = 0; i < N; ++i) h_src[i] = i;

    int* d_ptr = nullptr;
    ASSERT_EQ(cudaMalloc(&d_ptr, N * sizeof(int)), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(d_ptr, h_src, N * sizeof(int), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(h_dst, d_ptr, N * sizeof(int), cudaMemcpyDeviceToHost), cudaSuccess);

    ASSERT_EQ(cudaFree(d_ptr), cudaSuccess);

    for (int i = 0; i < N; ++i) {
        ASSERT_EQ(h_dst[i], h_src[i]);
    }
}


//cudaFreeHost
TEST(cudaRT, FreeHost) {
    void* host_ptr = nullptr;
    ASSERT_EQ(cudaMallocHost(&host_ptr, 1024), cudaSuccess);
    ASSERT_NE(host_ptr, nullptr);
    ASSERT_EQ(cudaFreeHost(host_ptr), cudaSuccess);
}


//cudaEventCreate
TEST(cudaRT, EventCreate) {
    cudaEvent_t event;
    ASSERT_EQ(cudaEventCreate(&event), cudaSuccess);
    ASSERT_NE(event, nullptr);
    ASSERT_EQ(cudaEventDestroy(event), cudaSuccess);
}


//cudaGetDeviceCount
TEST(cudaRT, GetDeviceCount) {
    int count = 0;
    ASSERT_EQ(cudaGetDeviceCount(&count), cudaSuccess);
    ASSERT_GT(count, 0);
}


// cudaMalloc
TEST(cudaRT, Malloc) {
    int* d_ptr = nullptr;
    ASSERT_EQ(cudaMalloc(&d_ptr, 1024 * sizeof(int)), cudaSuccess);
    ASSERT_NE(d_ptr, nullptr);
    ASSERT_EQ(cudaFree(d_ptr), cudaSuccess);
}


//cudaPeekAtLastError
TEST(cudaRT, PeekAtLastError) {
    // Launch an invalid kernel (no implementation)
    void* invalid_ptr = nullptr;
    cudaError_t err = cudaMemcpy(invalid_ptr, invalid_ptr, 100, cudaMemcpyDeviceToDevice);

    // Peek at the error (should not be cudaSuccess)
    cudaError_t peek = cudaPeekAtLastError();
    ASSERT_NE(peek, cudaSuccess);
}


//cudaStreamSynchronize
TEST(cudaRT, StreamSynchronize) {
    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

//cudaEventDestroy
TEST(cudaRT, EventDestroy) {
    cudaEvent_t event;
    ASSERT_EQ(cudaEventCreate(&event), cudaSuccess);
    ASSERT_EQ(cudaEventDestroy(event), cudaSuccess);
}


//cudaSetDevice
TEST(cudaRT, SetDevice) {
    int device_count = 0;
    ASSERT_EQ(cudaGetDeviceCount(&device_count), cudaSuccess);
    ASSERT_GT(device_count, 0);

    // Set to device 0
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

    int current_device = -1;
    ASSERT_EQ(cudaGetDevice(&current_device), cudaSuccess);
    ASSERT_EQ(current_device, 0);
}

//cudaGetLastError
TEST(cudaRT, GetLastError) {
    // Clear any previous error
    cudaError_t reset = cudaGetLastError();
    (void)reset; // suppress unused warning

    // Intentionally trigger an error
    cudaError_t err = cudaFree(nullptr);
    ASSERT_NE(err, cudaSuccess);

    // Now check that last error matches
    cudaError_t last = cudaGetLastError();
    ASSERT_EQ(last, err);
}


//cudaLaunchKernel
__global__ void launchKernelCheck(int* out) {
    if (threadIdx.x == 0) *out = 123;
}

TEST(cudaRT, LaunchKernel) {
    int* d_out = nullptr;
    int h_out = 0;

    ASSERT_EQ(cudaMalloc(&d_out, sizeof(int)), cudaSuccess);

    void* args[] = { &d_out };
    dim3 grid(1), block(1);

    ASSERT_EQ(cudaLaunchKernel((void*)launchKernelCheck, grid, block, args, 0, nullptr), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(&h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaFree(d_out), cudaSuccess);

    ASSERT_EQ(h_out, 123);
}


//cudaGetErrorString
TEST(cudaRT, GetErrorString) {
    cudaError_t err = cudaFree(nullptr);  // expected failure
    const char* str = cudaGetErrorString(err);

    ASSERT_NE(err, cudaSuccess);
    ASSERT_NE(str, nullptr);
}

