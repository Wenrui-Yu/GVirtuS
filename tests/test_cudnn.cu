#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cudnn.h>

#define CUDA_CHECK(err) ASSERT_EQ((err), cudaSuccess) << "CUDA Error: " << cudaGetErrorString(err)
#define CUDNN_CHECK(err) ASSERT_EQ((err), CUDNN_STATUS_SUCCESS) << "CUDNN Error: " << cudnnGetErrorString(err)

//cudnnCreate
// TEST(cuDNN, Create) {
//     cudnnHandle_t handle;
//     CUDNN_CHECK(cudnnCreate(&handle));
//     CUDNN_CHECK(cudnnDestroy(handle));
// }


// //cudnnDestroy
// TEST(cuDNN, Destroy) {
//     cudnnHandle_t handle;
//     CUDNN_CHECK(cudnnCreate(&handle));
//     CUDNN_CHECK(cudnnDestroy(handle));
// }


// //cudnnCreateTensorDescriptor
// TEST(cuDNN, CreateTensorDescriptor) {
//     cudnnTensorDescriptor_t desc;
//     CUDNN_CHECK(cudnnCreateTensorDescriptor(&desc));
//     CUDNN_CHECK(cudnnDestroyTensorDescriptor(desc));
// }


// //cudnnDestroyTensorDescriptor
// TEST(cuDNN, DestroyTensorDescriptor) {
//     cudnnTensorDescriptor_t desc;
//     CUDNN_CHECK(cudnnCreateTensorDescriptor(&desc));
//     CUDNN_CHECK(cudnnDestroyTensorDescriptor(desc));
// }

// //cudnnCreateActivationDescriptor
// TEST(cuDNN, CreateActivationDescriptor) {
//     cudnnActivationDescriptor_t desc;
//     CUDNN_CHECK(cudnnCreateActivationDescriptor(&desc));
//     CUDNN_CHECK(cudnnDestroyActivationDescriptor(desc));
// }


// //cudnnDestroyActivationDescriptor
// TEST(cuDNN, DestroyActivationDescriptor) {
//     cudnnActivationDescriptor_t desc;
//     CUDNN_CHECK(cudnnCreateActivationDescriptor(&desc));
//     CUDNN_CHECK(cudnnDestroyActivationDescriptor(desc));
// }

// //cudnnCreateConvolutionDescriptor
// TEST(cuDNN, CreateConvolutionDescriptor) {
//     cudnnConvolutionDescriptor_t desc;
//     CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&desc));
//     CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(desc));
// }

// //cudnnDestroyConvolutionDescriptor
// TEST(cuDNN, DestroyConvolutionDescriptor) {
//     cudnnConvolutionDescriptor_t desc;
//     CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&desc));
//     CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(desc));
// }


// //cudnnCreateFilterDescriptor
// TEST(cuDNN, CreateFilterDescriptor) {
//     cudnnFilterDescriptor_t desc;
//     CUDNN_CHECK(cudnnCreateFilterDescriptor(&desc));
//     CUDNN_CHECK(cudnnDestroyFilterDescriptor(desc));
// }


// //cudnnDestroyFilterDescriptor
// TEST(cuDNN, DestroyFilterDescriptor) {
//     cudnnFilterDescriptor_t desc;
//     CUDNN_CHECK(cudnnCreateFilterDescriptor(&desc));
//     CUDNN_CHECK(cudnnDestroyFilterDescriptor(desc));
// }


// //cudnnCreatePoolingDescriptor
// TEST(cuDNN, CreatePoolingDescriptor) {
//     cudnnPoolingDescriptor_t desc;
//     CUDNN_CHECK(cudnnCreatePoolingDescriptor(&desc));
//     CUDNN_CHECK(cudnnDestroyPoolingDescriptor(desc));
// }

// //cudnnDestroyPoolingDescriptor
// TEST(cuDNN, DestroyPoolingDescriptor) {
//     cudnnPoolingDescriptor_t desc;
//     CUDNN_CHECK(cudnnCreatePoolingDescriptor(&desc));
//     CUDNN_CHECK(cudnnDestroyPoolingDescriptor(desc));
// }


// //cudnnCreateLRNDescriptor
// TEST(cuDNN, CreateLRNDescriptor) {
//     cudnnLRNDescriptor_t desc;
//     CUDNN_CHECK(cudnnCreateLRNDescriptor(&desc));
//     CUDNN_CHECK(cudnnDestroyLRNDescriptor(desc));
// }


// //cudnnDestroyLRNDescriptor
// TEST(cuDNN, DestroyLRNDescriptor) {
//     cudnnLRNDescriptor_t desc;
//     CUDNN_CHECK(cudnnCreateLRNDescriptor(&desc));
//     CUDNN_CHECK(cudnnDestroyLRNDescriptor(desc));
// }

//cudnnSetTensor4dDescriptorEx
TEST(cuDNN, SetTensor4dDescriptorEx) {
    cudnnHandle_t handle;
    cudnnTensorDescriptor_t desc;

    CUDNN_CHECK(cudnnCreate(&handle));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&desc));

    // Tensor dims: NCHW = 1x1x2x2
    int n = 1, c = 1, h = 2, w = 2;
    int nStride = c * h * w;     // 4
    int cStride = h * w;         // 4
    int hStride = w;             // 2
    int wStride = 1;

    CUDNN_CHECK(cudnnSetTensor4dDescriptorEx(
        desc,
        CUDNN_DATA_FLOAT,
        n, c, h, w,
        nStride, cStride, hStride, wStride));

    CUDNN_CHECK(cudnnDestroyTensorDescriptor(desc));
    CUDNN_CHECK(cudnnDestroy(handle));
}



//cudnnSetActivationDescriptor
// TEST(cuDNN, SetActivationDescriptor) {
//     cudnnActivationDescriptor_t desc;
//     CUDNN_CHECK(cudnnCreateActivationDescriptor(&desc));
//     CUDNN_CHECK(cudnnSetActivationDescriptor(desc, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0.0));
//     CUDNN_CHECK(cudnnDestroyActivationDescriptor(desc));
// }

// //cudnnSetConvolution2dDescriptor
// TEST(cuDNN, SetConvolution2dDescriptor) {
//     cudnnConvolutionDescriptor_t desc;
//     CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&desc));
//     CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
//         desc, 1, 1, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
//     CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(desc));
// }


//cudnnSetFilter4dDescriptor
// TEST(cuDNN, SetFilter4dDescriptor) {
//     cudnnFilterDescriptor_t desc;
//     CUDNN_CHECK(cudnnCreateFilterDescriptor(&desc));
//     CUDNN_CHECK(cudnnSetFilter4dDescriptor(
//         desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 1, 1, 3, 3));
//     CUDNN_CHECK(cudnnDestroyFilterDescriptor(desc));
// }


// //cudnnSetLRNDescriptor
// TEST(cuDNN, SetLRNDescriptor) {
//     cudnnLRNDescriptor_t desc;
//     CUDNN_CHECK(cudnnCreateLRNDescriptor(&desc));
//     CUDNN_CHECK(cudnnSetLRNDescriptor(desc, 5, 1.0, 0.75, 1.0));
//     CUDNN_CHECK(cudnnDestroyLRNDescriptor(desc));
// }

// //cudnnSetPooling2dDescriptor
// TEST(cuDNN, SetPooling2dDescriptor) {
//     cudnnPoolingDescriptor_t desc;
//     CUDNN_CHECK(cudnnCreatePoolingDescriptor(&desc));
//     CUDNN_CHECK(cudnnSetPooling2dDescriptor(
//         desc, CUDNN_POOLING_MAX, CUDNN_PROPAGATE_NAN,
//         2, 2, 0, 0, 2, 2));
//     CUDNN_CHECK(cudnnDestroyPoolingDescriptor(desc));
// }


// //cudnnSetStream
// TEST(cuDNN, SetStream) {
//     cudnnHandle_t handle;
//     cudaStream_t stream;
//     CUDNN_CHECK(cudnnCreate(&handle));
//     CUDA_CHECK(cudaStreamCreate(&stream));

//     CUDNN_CHECK(cudnnSetStream(handle, stream));

//     CUDA_CHECK(cudaStreamDestroy(stream));
//     CUDNN_CHECK(cudnnDestroy(handle));
// }

// //cudnnPoolingForward
// TEST(cuDNN, PoolingForward) {
//     cudnnHandle_t handle;
//     CUDNN_CHECK(cudnnCreate(&handle));

//     cudnnTensorDescriptor_t inDesc, outDesc;
//     cudnnPoolingDescriptor_t poolDesc;

//     float alpha = 1.0f, beta = 0.0f;
//     int n = 1, c = 1, h = 4, w = 4;
//     int poolH = 2, poolW = 2;

//     float input[16] = {
//         1, 2, 3, 4,
//         5, 6, 7, 8,
//         9,10,11,12,
//        13,14,15,16
//     };
//     float output[4] = {0};

//     float *d_in, *d_out;
//     CUDA_CHECK(cudaMalloc(&d_in, sizeof(input)));
//     CUDA_CHECK(cudaMalloc(&d_out, sizeof(output)));
//     CUDA_CHECK(cudaMemcpy(d_in, input, sizeof(input), cudaMemcpyHostToDevice));

//     CUDNN_CHECK(cudnnCreateTensorDescriptor(&inDesc));
//     CUDNN_CHECK(cudnnCreateTensorDescriptor(&outDesc));
//     CUDNN_CHECK(cudnnCreatePoolingDescriptor(&poolDesc));

//     CUDNN_CHECK(cudnnSetTensor4dDescriptor(inDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w));
//     CUDNN_CHECK(cudnnSetTensor4dDescriptor(outDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, 2, 2));
//     CUDNN_CHECK(cudnnSetPooling2dDescriptor(poolDesc, CUDNN_POOLING_MAX, CUDNN_PROPAGATE_NAN,
//                                             poolH, poolW, 0, 0, poolH, poolW));

//     CUDNN_CHECK(cudnnPoolingForward(handle, poolDesc, &alpha, inDesc, d_in, &beta, outDesc, d_out));

//     CUDA_CHECK(cudaMemcpy(output, d_out, sizeof(output), cudaMemcpyDeviceToHost));
//     EXPECT_FLOAT_EQ(output[0], 6.0f);  // max(1,2,5,6)
//     EXPECT_FLOAT_EQ(output[1], 8.0f);  // max(3,4,7,8)
//     EXPECT_FLOAT_EQ(output[2], 14.0f); // max(9,10,13,14)
//     EXPECT_FLOAT_EQ(output[3], 16.0f); // max(11,12,15,16)

//     CUDA_CHECK(cudaFree(d_in));
//     CUDA_CHECK(cudaFree(d_out));
//     CUDNN_CHECK(cudnnDestroyTensorDescriptor(inDesc));
//     CUDNN_CHECK(cudnnDestroyTensorDescriptor(outDesc));
//     CUDNN_CHECK(cudnnDestroyPoolingDescriptor(poolDesc));
//     CUDNN_CHECK(cudnnDestroy(handle));
// }


// //cudnnSoftmaxForward
// TEST(cuDNN, SoftmaxForward) {
//     cudnnHandle_t handle;
//     CUDNN_CHECK(cudnnCreate(&handle));

//     cudnnTensorDescriptor_t tensorDesc;
//     float alpha = 1.0f, beta = 0.0f;
//     float input[] = {1.0f, 2.0f, 3.0f};
//     float output[3] = {0};

//     float *d_in, *d_out;
//     CUDA_CHECK(cudaMalloc(&d_in, sizeof(input)));
//     CUDA_CHECK(cudaMalloc(&d_out, sizeof(output)));
//     CUDA_CHECK(cudaMemcpy(d_in, input, sizeof(input), cudaMemcpyHostToDevice));

//     CUDNN_CHECK(cudnnCreateTensorDescriptor(&tensorDesc));
//     CUDNN_CHECK(cudnnSetTensor4dDescriptor(tensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 1, 3));

//     CUDNN_CHECK(cudnnSoftmaxForward(handle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE,
//                                     &alpha, tensorDesc, d_in, &beta, tensorDesc, d_out));

//     CUDA_CHECK(cudaMemcpy(output, d_out, sizeof(output), cudaMemcpyDeviceToHost));
//     float sum = output[0] + output[1] + output[2];
//     EXPECT_NEAR(sum, 1.0f, 1e-5);  // Softmax should sum to 1

//     CUDA_CHECK(cudaFree(d_in));
//     CUDA_CHECK(cudaFree(d_out));
//     CUDNN_CHECK(cudnnDestroyTensorDescriptor(tensorDesc));
//     CUDNN_CHECK(cudnnDestroy(handle));
// }

// //cudnnActivationForward
// TEST(cuDNN, ActivationForward) {
//     cudnnHandle_t handle;
//     CUDNN_CHECK(cudnnCreate(&handle));

//     cudnnTensorDescriptor_t desc;
//     cudnnActivationDescriptor_t act;
//     float alpha = 1.0f, beta = 0.0f;
//     float input[] = {-1.0f, 0.0f, 1.0f};
//     float output[3];

//     float *d_in, *d_out;
//     CUDA_CHECK(cudaMalloc(&d_in, sizeof(input)));
//     CUDA_CHECK(cudaMalloc(&d_out, sizeof(output)));
//     CUDA_CHECK(cudaMemcpy(d_in, input, sizeof(input), cudaMemcpyHostToDevice));

//     CUDNN_CHECK(cudnnCreateTensorDescriptor(&desc));
//     CUDNN_CHECK(cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 1, 3));
//     CUDNN_CHECK(cudnnCreateActivationDescriptor(&act));
//     CUDNN_CHECK(cudnnSetActivationDescriptor(act, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0.0));

//     CUDNN_CHECK(cudnnActivationForward(handle, act, &alpha, desc, d_in, &beta, desc, d_out));

//     CUDA_CHECK(cudaMemcpy(output, d_out, sizeof(output), cudaMemcpyDeviceToHost));
//     EXPECT_FLOAT_EQ(output[0], 0.0f);
//     EXPECT_FLOAT_EQ(output[1], 0.0f);
//     EXPECT_FLOAT_EQ(output[2], 1.0f);

//     CUDA_CHECK(cudaFree(d_in));
//     CUDA_CHECK(cudaFree(d_out));
//     CUDNN_CHECK(cudnnDestroyActivationDescriptor(act));
//     CUDNN_CHECK(cudnnDestroyTensorDescriptor(desc));
//     CUDNN_CHECK(cudnnDestroy(handle));
// }


// //cudnnActivationBackward
// TEST(cuDNN, ActivationBackward) {
//     cudnnHandle_t handle;
//     CUDNN_CHECK(cudnnCreate(&handle));

//     cudnnTensorDescriptor_t desc;
//     cudnnActivationDescriptor_t act;
//     float alpha = 1.0f, beta = 0.0f;

//     float x[] = {-1.0f, 0.0f, 1.0f};
//     float y[] = {0.0f, 0.0f, 1.0f}; // result of ReLU
//     float dy[] = {1.0f, 1.0f, 1.0f}; // gradient from next layer
//     float dx[3];

//     float *d_x, *d_y, *d_dy, *d_dx;
//     CUDA_CHECK(cudaMalloc(&d_x, sizeof(x)));
//     CUDA_CHECK(cudaMalloc(&d_y, sizeof(y)));
//     CUDA_CHECK(cudaMalloc(&d_dy, sizeof(dy)));
//     CUDA_CHECK(cudaMalloc(&d_dx, sizeof(dx)));

//     CUDA_CHECK(cudaMemcpy(d_x, x, sizeof(x), cudaMemcpyHostToDevice));
//     CUDA_CHECK(cudaMemcpy(d_y, y, sizeof(y), cudaMemcpyHostToDevice));
//     CUDA_CHECK(cudaMemcpy(d_dy, dy, sizeof(dy), cudaMemcpyHostToDevice));

//     CUDNN_CHECK(cudnnCreateTensorDescriptor(&desc));
//     CUDNN_CHECK(cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 1, 3));
//     CUDNN_CHECK(cudnnCreateActivationDescriptor(&act));
//     CUDNN_CHECK(cudnnSetActivationDescriptor(act, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0.0));

//     CUDNN_CHECK(cudnnActivationBackward(handle, act, &alpha, desc, d_y, desc, d_dy, desc, d_x, &beta, desc, d_dx));

//     CUDA_CHECK(cudaMemcpy(dx, d_dx, sizeof(dx), cudaMemcpyDeviceToHost));
//     EXPECT_FLOAT_EQ(dx[0], 0.0f); // ReLU gradient = 0 if x < 0
//     EXPECT_FLOAT_EQ(dx[1], 0.0f); // ReLU gradient = 0 if x = 0
//     EXPECT_FLOAT_EQ(dx[2], 1.0f); // ReLU gradient = 1 if x > 0

//     CUDA_CHECK(cudaFree(d_x));
//     CUDA_CHECK(cudaFree(d_y));
//     CUDA_CHECK(cudaFree(d_dy));
//     CUDA_CHECK(cudaFree(d_dx));
//     CUDNN_CHECK(cudnnDestroyActivationDescriptor(act));
//     CUDNN_CHECK(cudnnDestroyTensorDescriptor(desc));
//     CUDNN_CHECK(cudnnDestroy(handle));
// }


// //cudnnPoolingBackward
// TEST(cuDNN, PoolingBackward) {
//     cudnnHandle_t handle;
//     CUDNN_CHECK(cudnnCreate(&handle));

//     cudnnTensorDescriptor_t inDesc, outDesc;
//     cudnnPoolingDescriptor_t poolDesc;

//     float alpha = 1.0f, beta = 0.0f;
//     float x[] = {1.0f, 5.0f, 3.0f, 4.0f};  // 2x2
//     float y[] = {5.0f};                   // max pooled
//     float dy[] = {1.0f};                  // gradient from next layer
//     float dx[4];

//     float *d_x, *d_y, *d_dy, *d_dx;
//     CUDA_CHECK(cudaMalloc(&d_x, sizeof(x)));
//     CUDA_CHECK(cudaMalloc(&d_y, sizeof(y)));
//     CUDA_CHECK(cudaMalloc(&d_dy, sizeof(dy)));
//     CUDA_CHECK(cudaMalloc(&d_dx, sizeof(dx)));

//     CUDA_CHECK(cudaMemcpy(d_x, x, sizeof(x), cudaMemcpyHostToDevice));
//     CUDA_CHECK(cudaMemcpy(d_y, y, sizeof(y), cudaMemcpyHostToDevice));
//     CUDA_CHECK(cudaMemcpy(d_dy, dy, sizeof(dy), cudaMemcpyHostToDevice));

//     CUDNN_CHECK(cudnnCreateTensorDescriptor(&inDesc));
//     CUDNN_CHECK(cudnnCreateTensorDescriptor(&outDesc));
//     CUDNN_CHECK(cudnnCreatePoolingDescriptor(&poolDesc));

//     CUDNN_CHECK(cudnnSetTensor4dDescriptor(inDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 2, 2));
//     CUDNN_CHECK(cudnnSetTensor4dDescriptor(outDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 1, 1));
//     CUDNN_CHECK(cudnnSetPooling2dDescriptor(poolDesc, CUDNN_POOLING_MAX, CUDNN_PROPAGATE_NAN,
//                                             2, 2, 0, 0, 2, 2));

//     CUDNN_CHECK(cudnnPoolingBackward(handle, poolDesc, &alpha,
//                                      outDesc, d_y, outDesc, d_dy,
//                                      inDesc, d_x, &beta,
//                                      inDesc, d_dx));

//     CUDA_CHECK(cudaMemcpy(dx, d_dx, sizeof(dx), cudaMemcpyDeviceToHost));
//     EXPECT_FLOAT_EQ(dx[0], 0.0f);
//     EXPECT_FLOAT_EQ(dx[1], 1.0f); // max was at position 1
//     EXPECT_FLOAT_EQ(dx[2], 0.0f);
//     EXPECT_FLOAT_EQ(dx[3], 0.0f);

//     CUDA_CHECK(cudaFree(d_x));
//     CUDA_CHECK(cudaFree(d_y));
//     CUDA_CHECK(cudaFree(d_dy));
//     CUDA_CHECK(cudaFree(d_dx));
//     CUDNN_CHECK(cudnnDestroyTensorDescriptor(inDesc));
//     CUDNN_CHECK(cudnnDestroyTensorDescriptor(outDesc));
//     CUDNN_CHECK(cudnnDestroyPoolingDescriptor(poolDesc));
//     CUDNN_CHECK(cudnnDestroy(handle));
// }


// //cudnnSoftmaxBackward
// TEST(cuDNN, SoftmaxBackward) {
//     cudnnHandle_t handle;
//     CUDNN_CHECK(cudnnCreate(&handle));

//     cudnnTensorDescriptor_t tensorDesc;
//     float alpha = 1.0f, beta = 0.0f;

//     float y[]  = {0.1f, 0.6f, 0.3f};
//     float dy[] = {1.0f, 1.0f, 1.0f};
//     float dx[3] = {0};

//     float *d_y, *d_dy, *d_dx;
//     CUDA_CHECK(cudaMalloc(&d_y, sizeof(y)));
//     CUDA_CHECK(cudaMalloc(&d_dy, sizeof(dy)));
//     CUDA_CHECK(cudaMalloc(&d_dx, sizeof(dx)));

//     CUDA_CHECK(cudaMemcpy(d_y, y, sizeof(y), cudaMemcpyHostToDevice));
//     CUDA_CHECK(cudaMemcpy(d_dy, dy, sizeof(dy), cudaMemcpyHostToDevice));

//     CUDNN_CHECK(cudnnCreateTensorDescriptor(&tensorDesc));
//     // NCHW layout: 1 sample, 1 channel, 1x3 spatial
//     CUDNN_CHECK(cudnnSetTensor4dDescriptor(
//         tensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 1, 3));

//     CUDNN_CHECK(cudnnSoftmaxBackward(
//         handle,
//         CUDNN_SOFTMAX_ACCURATE,
//         CUDNN_SOFTMAX_MODE_INSTANCE,
//         &alpha,
//         tensorDesc, d_y,
//         tensorDesc, d_dy,
//         &beta,
//         tensorDesc, d_dx));

//     CUDA_CHECK(cudaFree(d_y));
//     CUDA_CHECK(cudaFree(d_dy));
//     CUDA_CHECK(cudaFree(d_dx));
//     CUDNN_CHECK(cudnnDestroyTensorDescriptor(tensorDesc));
//     CUDNN_CHECK(cudnnDestroy(handle));
// }


