# 🧠 OpenPose with GVirtuS Integration (06/06/2025)

This guide details how to run OpenPose inside a Docker container with GVirtuS support, addresses common issues, and outlines multiple approaches for integrating OpenPose with GVirtuS (including caveats).

---

## 🚀 Docker Run Command

```bash
docker run -it --name openpose_gvirtus_env \
  --network=host \
  --privileged \
  --gpus all \
  --env DISPLAY=$DISPLAY \
  --env QT_X11_NO_MITSHM=1 \
  --volume /tmp/.X11-unix:/tmp/.X11-unix:rw \
  --volume /dev:/dev \
  openpose-gvirtus-image
```

### 🖥️ Enable GUI Window Access

```bash
xhost +local:root
```

### 🔓 Access Running Docker Container

```bash
docker exec -it openpose_gvirtus_env bash
```

---

## 🎥 Run OpenPose on Video

Make sure MIT-SHM (Shared Memory) permissions are disabled:

```bash
export MIT_SHM_DISABLE=1
./build/examples/openpose/openpose.bin
```

📎 Related Issue: [openpose#2321](https://github.com/CMU-Perceptual-Computing-Lab/openpose/issues/2321)

---

## ⚙️ Custom OpenPose Script via CLI

### ➤ Without GVirtuS

1. **Navigate to script directory**:

```bash
cd openpose/examples/gvirtus_api
```

2. **Compile the script**:

```bash
g++ 00_test.cpp -o try \
  -I/home/openpose/include \
  -I/usr/include/opencv4 \
  -L/home/openpose/build/src/openpose -lopenpose \
  -lgflags -lglog -lprotobuf -pthread \
  `pkg-config --cflags --libs opencv4` \
  -std=c++11 -Wno-unused-result -Wno-write-strings
```

3. **Run the binary**:

```bash
cd /home/openpose
./examples/gvirtus_api/try
```

---

### ➤ With GVirtuS (⚠️ Not Working Yet)

Same compilation steps as above, but execution with GVirtuS fails. Needs further investigation.

---

## ❗ Important Note on GVirtuS Integration

OpenPose's Caffe backend **requires real CUDA libraries** to compile. GVirtuS **cannot be used to build** OpenPose or Caffe directly due to missing symbols during linking:

```txt
undefined reference to `__cudaRegisterFatBinaryEnd@libcudart.so.12'
undefined reference to `cudaFree@libcudart.so.12'
undefined reference to `cublasCreate_v2@libcublas.so.12'
```

### ✅ Solution Attempt 1 (CPU Mode Build)

1. Edit OpenPose `CMakeLists.txt`:

```cmake
# Line 724:
set(CAFFE CPU_ONLY ON)
```

2. Build OpenPose in CPU-only mode:

```bash
cd /home/openpose
rm -rf build && mkdir build && cd build

cmake \
  -DBUILD_PYTHON=ON \
  -DUSE_CUDA=OFF \
  -DCPU_ONLY=ON \
  -DUSE_CUDNN=OFF \
  -DBUILD_CAFFE=ON \
  -DBUILD_OPENPOSE=ON \
  -DBUILD_SHARED_LIBS=ON \
  -DPYTHON_EXECUTABLE=$(which python3.8) \
  -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.8.so.1.0 \
  ..
```

3. Compile custom CUDA script using GVirtuS runtime:

```bash
export GVIRTUS_HOME=/home/GVirtuS
export LD_LIBRARY_PATH=${GVIRTUS_HOME}/lib:${GVIRTUS_HOME}/lib/frontend:/home/openpose/build/src/openpose:/home/openpose/build/caffe/lib:$LD_LIBRARY_PATH

nvcc 01_test.cu -o openpose_demo_gvirtus \
  -std=c++14 \
  -I/home/openpose/include \
  -I/usr/include/opencv4 \
  -I/home/openpose/3rdparty/caffe/include \
  -I/home/openpose/3rdparty/caffe/build/include \
  -L/home/openpose/build/src/openpose -lopenpose \
  -L/home/openpose/build/caffe/lib -lcaffe \
  -L${GVIRTUS_HOME}/lib -L${GVIRTUS_HOME}/lib/frontend \
  -lcuda -lcudart -lcublas -lcufft -lcudnn \
  -lgflags -lglog \
  -Xcompiler -pthread \
  `pkg-config --cflags --libs opencv4`
```

### 🔧 Force CPU-Only In Your Script

Modify OpenPose wrapper configuration:

```cpp
// CPU-only configuration for OpenPose
op::WrapperStructPose poseConfig;
poseConfig.poseMode = op::PoseMode::Enabled;
poseConfig.poseModel = op::PoseModel::BODY_25;
poseConfig.netInputSize = op::Point<int>{656, 368};
poseConfig.outputSize = op::Point<int>{-1, -1};
poseConfig.renderMode = op::RenderMode::None; // No GPU rendering
poseConfig.alphaKeypoint = 0.6;
poseConfig.alphaHeatMap = 0.7;
poseConfig.defaultPartToRender = 0; // BODY_25
poseConfig.enableGoogleLogging = true;

// 🚫 Disable GPU usage
poseConfig.gpuNumber = 0;
poseConfig.gpuNumberStart = 0;

opWrapper.configure(poseConfig);
opWrapper.start();
```

📝 **Note**: Script runs with GVirtuS in CPU-only mode, but keypoint detection performance was **poor**.

---

## 🧪 Potential 3rd Solution (Experimental)

Enable GPU in the wrapper config and let GVirtuS attempt CUDA virtualization:

* Compile Caffe with real CUDA
* Link runtime to GVirtuS
* Allow OpenPose to run with GPU enabled

🚫 Currently **not functional** – likely due to GVirtuS lacking full support for CUDA 12.2 symbols.

---

## 📌 Summary

| Mode                        | Builds? | Runs? | Real GPU Usage? | Output Quality |
|----------------------------|:-------:|:-----:|:----------------:|:---------------:|
| Real CUDA                  | ✅      | ✅    | ✅               | ✅ Full         |
| GVirtuS + CPU_ONLY         | ✅      | ✅    | ❌               | ⚠️ Low          |
| GVirtuS + real CUDA build  | ❌      | ❌    | ❌               | ❌ (linker errors) |

---

## List of missing functions in GVirtuS to integrate with openpose

Here's a full segregation of the missing symbols by CUDA library:

---

### 🟢 **`libcudart.so.12` (CUDA Runtime)**

These functions are part of the CUDA Runtime API and are generally defined in `libcudart`.

```
1.  __cudaRegisterFatBinaryEnd
2.  cudaFree
6.  cudaStreamCreateWithFlags
7.  cudaEventElapsedTime
10. __cudaRegisterFunction
13. cudaGetDeviceProperties_v2
14. cudaMemset
15. cudaMemGetInfo
16. cudaStreamDestroy
17. cudaGetLastError
18. cudaEventRecord
19. cudaMallocHost
22. cudaEventSynchronize
38. __cudaPopCallConfiguration
39. cudaMemcpyAsync
41. cudaGetDevice
46. cudaStreamCreate
47. cudaMemcpy
51. cudaLaunchKernel
58. cudaFreeHost
62. __cudaPushCallConfiguration
64. cudaEventCreate
70. __cudaRegisterFatBinary
71. cudaGetDeviceCount
73. cudaMalloc
76. cudaPeekAtLastError
78. cudaGetErrorString
81. cudaStreamSynchronize
83. __cudaRegisterVar
85. cudaEventDestroy
86. cudaSetDevice
87. __cudaUnregisterFatBinary
```

---

### 🔵 **`libcublas.so.12` (cuBLAS - Linear Algebra)**

These are from NVIDIA's cuBLAS library, which is used for GPU-accelerated linear algebra.

```
5.  cublasCreate_v2
8.  cublasGetStream_v2
12. cublasSaxpy_v2
21. cublasDestroy_v2
25. cublasSetStream_v2
35. cublasSgemm_v2
42. cublasDcopy_v2
48. cublasDgemv_v2
49. cublasDasum_v2
53. cublasSdot_v2
57. cublasSgemv_v2
59. cublasSasum_v2
60. cublasDgemm_v2
65. cublasDscal_v2
67. cublasSscal_v2
69. cublasDdot_v2
74. cublasDaxpy_v2
75. cublasScopy_v2
```

---

### 🔴 **`libcudnn.so.8` (cuDNN - Deep Neural Networks)**

These belong to NVIDIA's cuDNN library, for accelerating deep learning primitives.

```
3.  cudnnSetActivationDescriptor
4.  cudnnDestroyConvolutionDescriptor
9.  cudnnConvolutionBackwardData
11. cudnnDivisiveNormalizationForward
20. cudnnGetConvolutionBackwardDataAlgorithm_v7
23. cudnnCreateTensorDescriptor
24. cudnnDestroy
26. cudnnGetConvolutionForwardAlgorithm_v7
27. cudnnCreateActivationDescriptor
28. cudnnDivisiveNormalizationBackward
29. cudnnConvolutionBackwardBias
30. cudnnSetConvolution2dDescriptor
31. cudnnSetTensor4dDescriptorEx
32. cudnnSoftmaxBackward
33. cudnnSoftmaxForward
34. cudnnCreate
36. cudnnLRNCrossChannelBackward
37. cudnnDestroyFilterDescriptor
40. cudnnSetStream
43. cudnnConvolutionBackwardFilter
44. cudnnSetFilter4dDescriptor
45. cudnnSetLRNDescriptor
50. cudnnCreateFilterDescriptor
52. cudnnCreatePoolingDescriptor
54. cudnnDestroyPoolingDescriptor
55. cudnnDestroyTensorDescriptor
56. cudnnPoolingBackward
61. cudnnPoolingForward
63. cudnnSetPooling2dDescriptor
66. cudnnCreateLRNDescriptor
68. cudnnActivationBackward
72. cudnnDestroyActivationDescriptor
77. cudnnAddTensor
79. cudnnLRNCrossChannelForward
80. cudnnActivationForward
82. cudnnConvolutionForward
84. cudnnCreateConvolutionDescriptor
```

---

### Summary Table

| Library       | Missing Symbols Count |
| ------------- | --------------------- |
| **libcudart** | 34                    |
| **libcublas** | 19                    |
| **libcudnn**  | 34                    |

---

### 🔧 Suggestion for GVirtuS Integration

For each of the above symbols, GVirtuS will likely need wrappers (client/server RPC functions) if not already implemented. Check the respective plugin directories:

* `gvirtus-plugin-cuda`
* `gvirtus-plugin-cublas`
* `gvirtus-plugin-cudnn`

If a function is missing:

* Create a server-side C++ wrapper calling the native CUDA/cuBLAS/cuDNN function.
* Create a client-side function with the same signature that calls into GVirtuS via the RPC mechanism.
* Add the signature to the relevant `.xml` or `.json` API mapping file if used.

---
