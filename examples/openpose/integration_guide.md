# 🧠 OpenPose with GVirtuS Integration

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
  -lcuda -lcudart -lcublas -lcufft -lcudnn -lcurand \
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

# GVirtuS - OpenPose CUDA Compatibility Summary (Full Library Coverage)

This document summarizes the analysis and verification of CUDA functions used by the OpenPose project compared against the GVirtuS frontend library implementation, across all major CUDA libraries.

---

## 🔍 Goal

Evaluate which CUDA functions used by OpenPose (and its dependencies) are supported by GVirtuS, and prepare for testing and further implementation.

---

## 🛠 Steps Taken

1. **Extract Used CUDA Functions**
   - Commands:
     ```bash
     nm -D /home/openpose/build/caffe/lib/libcaffe.so | c++filt | grep -E 'cuda|curand|cublas|cudnn|cufft|cusolver|cusparse|nvrtc'
     nm -D /home/openpose/build/src/openpose/libopenpose.so | c++filt | grep -E 'cuda|curand|cublas|cudnn|cufft|cusolver|cusparse|nvrtc'
     ```

2. **Extract GVirtuS Implemented Functions**
   - Command:
     ```bash
     nm -D ${GVIRTUS_HOME}/lib/frontend/lib*.so | c++filt > gvirtus_all_symbols.txt
     ```

3. **Compare and Generate Checklist**
   - Script-based comparison of all required vs. available symbols across:
     `cuda`, `cudart`, `cublas`, `curand`, `cudnn`, `cufft`, `cusolver`, `cusparse`, `nvrtc`.

---

## ✅ CUDA Function Coverage Checklist

| Function | Implemented | Tested | Unit test results | Notes |
|----------|-------------|--------|---------|------- |
| __cudaPopCallConfiguration | ❌ | ❌ | ❓ |  Not implemented @GVirtuS
| __cudaPushCallConfiguration | ❌ | ❌ | ❓ |  Not implemented @GVirtuS
| __cudaRegisterFatBinary | ❌ | ❌ | ❓ |  Not implemented @GVirtuS
| __cudaRegisterFatBinaryEnd | ❌ | ❌ | ❓ |  Not implemented @GVirtuS
| __cudaRegisterFunction | ❌ | ❌ | ❓ |  Not implemented @GVirtuS
| __cudaRegisterVar | ❌ | ❌ | ❓ |  Not implemented @GVirtuS
| __cudaUnregisterFatBinary | ❌ | ❌ | ❓ |  Not implemented @GVirtuS
| cublasCreate_v2 | ✅ | ✅ | ✅ |  
| cublasDasum_v2 | ✅ | ✅ | ❌ |  unit test failed
| cublasDaxpy_v2 | ✅ | ✅ | ✅ |  
| cublasDcopy_v2 | ✅ | ✅ | ✅ |  
| cublasDdot_v2 | ✅ | ✅ | ✅ |  
| cublasDestroy_v2 | ✅ | ✅ | ✅ |  
| cublasDgemm_v2 | ✅ | ✅ | ✅ |  
| cublasDgemv_v2 | ✅ | ✅ | ✅ |  
| cublasDscal_v2 | ✅ | ✅ | ✅ |  
| cublasGetStream_v2 | ✅ | ✅ | ❌ |  unit test failed
| cublasSasum_v2 | ✅ | ✅ | ✅ |  
| cublasSaxpy_v2 | ✅ | ✅ | ✅ |  
| cublasScopy_v2 | ✅ | ✅ | ✅ |  
| cublasSdot_v2 | ✅ | ✅ | ✅ |  
| cublasSetStream_v2 | ✅ | ✅ | ✅ |  
| cublasSgemm_v2 | ✅ | ✅ | ✅ |  
| cublasSgemv_v2 | ✅ | ✅ | ✅ |  
| cublasSscal_v2 | ✅ | ✅ | ✅ |  
| cudaEventCreate | ✅ | ❌ | ❓ |  
| cudaEventDestroy | ✅ | ❌ | ❓ |  
| cudaEventElapsedTime | ✅ | ❌ | ❓ |  
| cudaEventRecord | ✅ | ❌ | ❓ |  
| cudaEventSynchronize | ✅ | ❌ | ❓ |  
| cudaFree | ✅ | ❌ | ❓ |  
| cudaFreeHost | ✅ | ❌ | ❓ |  
| cudaGetDevice | ✅ | ❌ | ❓ |  
| cudaGetDeviceCount | ✅ | ❌ | ❓ |  
| cudaGetDeviceProperties_v2 | ✅ | ❌ | ❓ |  
| cudaGetErrorString | ✅ | ❌ | ❓ |  
| cudaGetLastError | ✅ | ❌ | ❓ |  
| cudaLaunchKernel | ✅ | ❌ | ❓ |  
| cudaMalloc | ✅ | ❌ | ❓ |  
| cudaMallocHost | ✅ | ❌ | ❓ |  
| cudaMemGetInfo | ✅ | ❌ | ❓ |  
| cudaMemcpy | ✅ | ❌ | ❓ |  
| cudaMemcpyAsync | ✅ | ❌ | ❓ |  
| cudaMemset | ✅ | ❌ | ❓ |  
| cudaPeekAtLastError | ✅ | ❌ | ❓ |  
| cudaSetDevice | ✅ | ❌ | ❓ |  
| cudaStreamCreate | ✅ | ❌ | ❓ |  
| cudaStreamCreateWithFlags | ✅ | ❌ | ❓ |  
| cudaStreamDestroy | ✅ | ❌ | ❓ |  
| cudaStreamSynchronize | ✅ | ❌ | ❓ |  
| cudnnActivationBackward | ✅ | ✅ | ✅ |   
| cudnnActivationForward | ✅ | ✅ | ✅ |   
| cudnnAddTensor | ✅ | ✅ | ✅ |   
| cudnnConvolutionBackwardBias | ✅ | ✅ | ✅ |   
| cudnnConvolutionBackwardData | ✅ | ✅ | ✅ |   
| cudnnConvolutionBackwardFilter | ✅ | ✅ | ✅ |   
| cudnnConvolutionForward | ✅ | ✅ | ✅ |   
| cudnnCreate | ✅ | ✅ | ✅ |   
| cudnnCreateActivationDescriptor | ✅ | ✅ | ✅ |   
| cudnnCreateConvolutionDescriptor | ✅ | ✅ | ✅ |   
| cudnnCreateFilterDescriptor | ✅ | ✅ | ✅ |   
| cudnnCreateLRNDescriptor | ✅ | ✅ | ✅ |   
| cudnnCreatePoolingDescriptor | ✅ | ✅ | ✅ |   
| cudnnCreateTensorDescriptor | ✅ | ✅ | ✅ |   
| cudnnDestroy | ✅ | ✅ | ✅ |   
| cudnnDestroyActivationDescriptor | ✅ | ✅ | ✅ |   
| cudnnDestroyConvolutionDescriptor | ✅ | ✅ | ✅ |   
| cudnnDestroyFilterDescriptor | ✅ | ✅ | ✅ |   
| cudnnDestroyPoolingDescriptor | ✅ | ✅ | ✅ |   
| cudnnDestroyTensorDescriptor | ✅ | ✅ | ✅ |   
| cudnnDivisiveNormalizationBackward | ✅ | ✅ | ✅ |   
| cudnnDivisiveNormalizationForward | ✅ | ✅ | ✅ |   
| cudnnGetConvolutionBackwardDataAlgorithm_v7 | ✅ | ✅ | ✅ |   
| cudnnGetConvolutionForwardAlgorithm_v7 | ✅ | ✅ | ✅ |   
| cudnnLRNCrossChannelBackward | ✅ | ✅ | ✅ |   
| cudnnLRNCrossChannelForward | ✅ | ✅ | ✅ |   
| cudnnPoolingBackward | ✅ | ✅ | ✅ |   
| cudnnPoolingForward | ✅ | ✅ | ✅ |   
| cudnnSetActivationDescriptor | ✅ | ✅ | ✅ |   
| cudnnSetConvolution2dDescriptor | ✅ | ✅ | ✅ |   
| cudnnSetFilter4dDescriptor | ✅ | ✅ | ✅ |   
| cudnnSetLRNDescriptor | ✅ | ✅ | ✅ |  
| cudnnSetPooling2dDescriptor | ✅ | ✅ | ✅ |   
| cudnnSetStream | ✅ | ✅ | ✅ |   
| cudnnSetTensor4dDescriptorEx | ✅ | ✅ | ❌ |  Unit test failed
| cudnnSoftmaxBackward | ✅ | ✅ | ❌ |  Unit test failed
| cudnnSoftmaxForward | ✅ | ✅ | ✅ |   
| curandCreateGenerator | ✅ | ✅ | ✅ |  
| curandDestroyGenerator | ✅ | ✅ | ✅ |  
| curandGenerate | ✅ | ✅ | ✅ |  
| curandGenerateNormal | ✅ | ✅ | ✅ |  
| curandGenerateNormalDouble | ✅ | ✅ | ✅ |  
| curandGenerateUniform | ✅ | ✅ | ✅ |  
| curandGenerateUniformDouble | ✅ | ✅ | ✅ |  
| curandSetGeneratorOffset | ❌ | ❌ | ❓ |  Not implemented @GVirtuS
| curandSetPseudoRandomGeneratorSeed | ✅ | ✅ | ✅ |  
