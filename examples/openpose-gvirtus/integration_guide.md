# 🧠 OpenPose with GVirtuS Integration

This guide details how to run OpenPose inside a Docker container with GVirtuS support, addresses common issues, and outlines multiple approaches for integrating OpenPose with GVirtuS (including caveats).

---

## 🚀 Docker Run Command

```bash
docker run -it --name openpose-dev1 \
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
docker start openpose-dev1 
docker exec -it openpose-dev1 bash
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
cd openpose/examples/gvirtus
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
./examples/gvirtus/try
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
| __cudaPopCallConfiguration | ❌ | ❌ | ❓ |  Not implemented in GVirtuS
| __cudaPushCallConfiguration | ❌ | ❌ | ❓ |  Not implemented in GVirtuS
| __cudaRegisterFatBinary | ❌ | ❌ | ❓ |  Not implemented in VirtuS
| __cudaRegisterFatBinaryEnd | ❌ | ❌ | ❓ |  Not implemented in GVirtuS
| __cudaRegisterFunction | ❌ | ❌ | ❓ |  Not implemented in GVirtuS
| __cudaRegisterVar | ❌ | ❌ | ❓ |  Not implemented in GVirtuS
| __cudaUnregisterFatBinary | ❌ | ❌ | ❓ |  Not implemented in GVirtuS
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
| cudaEventCreate | ✅ | ✅ | ✅ |  
| cudaEventDestroy | ✅ | ✅ | ✅ |  
| cudaEventElapsedTime | ✅ | ✅ | ✅ |  
| cudaEventRecord | ✅ | ✅ | ✅ |  
| cudaEventSynchronize | ✅ | ✅ | ✅ |  
| cudaFree | ✅ | ✅ | ✅ |  
| cudaFreeHost | ✅ | ✅ | ✅ | 
| cudaGetDevice | ✅ | ✅ | ✅ |  
| cudaGetDeviceCount | ✅ | ✅ | ✅ |  
| cudaGetDeviceProperties_v2 | ✅ | ✅ | ✅ |  
| cudaGetErrorString | ✅ | ✅ | ✅ |  
| cudaGetLastError | ✅ | ✅ | ✅ |  
| cudaLaunchKernel | ✅ | ✅ | ❌ |  unit test failed
| cudaMalloc | ✅ | ✅ | ✅ |  
| cudaMallocHost | ✅ | ✅ | ✅ |  
| cudaMemGetInfo | ✅ | ✅ | ✅ |  
| cudaMemcpy | ✅ | ✅ | ✅ |  
| cudaMemcpyAsync | ✅ | ✅ | ✅ |  
| cudaMemset | ✅ | ✅ | ✅ |  
| cudaPeekAtLastError | ✅ | ✅ | ✅ |  
| cudaSetDevice | ✅ | ✅ | ✅ |  
| cudaStreamCreate | ✅ | ✅ | ✅ |  
| cudaStreamCreateWithFlags | ✅ | ✅ | ✅ |  
| cudaStreamDestroy | ✅ | ✅ | ✅ |  
| cudaStreamSynchronize | ✅ | ✅ | ✅ |  
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
| cudnnSetTensor4dDescriptorEx | ✅ | ✅ | ✅ |  Solved
| cudnnSoftmaxBackward | ✅ | ✅ | ❌ |  Unit test failed
| cudnnSoftmaxForward | ✅ | ✅ | ✅ |   
| curandCreateGenerator | ✅ | ✅ | ✅ |  
| curandDestroyGenerator | ✅ | ✅ | ✅ |  
| curandGenerate | ✅ | ✅ | ✅ |  
| curandGenerateNormal | ✅ | ✅ | ✅ |  
| curandGenerateNormalDouble | ✅ | ✅ | ✅ |  
| curandGenerateUniform | ✅ | ✅ | ✅ |  
| curandGenerateUniformDouble | ✅ | ✅ | ✅ |  
| curandSetGeneratorOffset | ✅ | ✅ | ✅ |  Newly implemented in GVirtuS
| curandSetPseudoRandomGeneratorSeed | ✅ | ✅ | ✅ |  

--------------------
Error 1:
./run_openpose.sh 
INFO - GVirtuS frontend version /home/darshan/GVirtuS/etc/properties.json
Starting OpenPose with CUDA (.cu file)...

Error:
Cuda check failed (100 vs. 0): no CUDA-capable device is detected

Coming from:
- /home/darshan/openpose/src/openpose/gpu/cuda.cpp:getCudaGpuNumber():48

Root Cause:
fails with error code 100, which corresponds to:

    cudaErrorNoDevice = 100: No CUDA-capable device is detected.

  ---------
Error 2:
  ./openpose_demo_gvirtus 
INFO - GVirtuS frontend version /home/darshan/GVirtuS/etc/properties.json
Starting OpenPose with CUDA (.cu file)...

Error:
Cuda check failed (36 vs. 0): API call is not supported in the installed CUDA driver

Root cause:
The actual error returned by cudaGetDeviceCount() is code 36, which is:

    cudaErrorNotSupported = 36: API call is not supported on the installed CUDA driver
It means the code reached the GPU backend, tried to run a CUDA API call, and the driver explicitly rejected the API — because it's unsupported, disabled, or unavailable in that context.

Lets figure out this error coming with openpose or normal cuda_application integration with GVirtuS
cudaGetDeviceCount() in OpenPose	❌ Fails with error 36
cudaGetDeviceCount() in minimal CUDA test	❌ Fails with error 36
Other CUDA functions (like cudaMalloc, cudaMemcpy)	✅ Work under GVirtuS
GVirtuS backend handler for cudaGetDeviceCount()	✅ Exists, but uses cudaGetDeviceCount() internally
Driver APIs (cuDeviceGetCount)	✅ Present and usable

🎯 Root Cause

GVirtuS backend is using cudaGetDeviceCount() internally (i.e. calling itself).
That leads to:

    ❌ A recursive call chain or unresolved runtime context → returns cudaErrorNotSupported (36)

Solution: Modified cudaGetDeviceCount() function
-----

## 🛠 Resolving `Frontend.cpp:128` JSON Error in GVirtuS-OpenPose

### ❗ Problem

When running OpenPose with GVirtuS integration, the following error was encountered:

```
ERROR - "Frontend.cpp":128: Exception occurred: [json.exception.type_error.304] cannot use at() with null
```

This occurred when the OpenPose frontend attempted to load and parse `properties.json` during initialization.

---

### 🧠 Root Cause

The error was traced to this line inside `EndpointFactory::get_endpoint()`:

```cpp
if ("tcp/ip" == j["communicator"][ind_endpoint]["endpoint"].at("suite"))
```

This line uses `.at("suite")`, which **throws an exception if the key is missing or the value is `null`**.

Additionally, `ind_endpoint` was a static index variable that incremented with each call — causing an **out-of-bounds access** on subsequent `get_endpoint()` calls.

---

### ✅ Solution

We resolved this by **safely refactoring** the `EndpointFactory` logic:

#### ✔ Defensive JSON access

* Replaced `.at()` with safe checks using `.contains()` and `.get<std::string>()`
* Added validation for JSON structure (presence of `communicator`, `endpoint`, and `suite`)

#### ✔ Removed `ind_endpoint` logic

* Static index was removed, as only a single endpoint is used throughout the application's lifecycle
* A static `index()` method returning `0` was retained for compatibility with other modules

#### ✔ Integrated logging

* Replaced `std::cout` with `log4cplus` logging to remain consistent with GVirtuS logging practices

---

### 🔧 Final Highlights (in `EndpointFactory.h`)

* Uses only the first entry from `"communicator"` array (`index 0`)
* Validates and logs the JSON content
* Prevents crashes from malformed configs or index misuse

---

### 📌 Result

* The original `json.exception.type_error.304` crash is completely resolved
* GVirtuS frontend works reliably across repeated CUDA calls
* Logging and compatibility with existing GVirtuS modules is preserved

---

#### Error> E0728 12:40:53.495074 53596 common.cpp:114] Cannot create Cublas handle. Cublas won't be available. E0728 12:40:53.495978 53596 common.cpp:121] Cannot create Curand generator. Curand won't be available.

This is error due to the backend was not linking well with GVirtuS Home directory.
On the terminal where you're running backend.sh file try the below commands,

echo $GVIRTUS_HOME
This suppose to print '/home/darshan/GVirtuS' if it printed none then export to GVirtuS Home properly.
export GVIRTUS_HOME=/home/darshan/GVirtuS
After execute the backend,
LD_LIBRARY_PATH="${GVIRTUS_HOME}/lib:${LD_LIBRARY_PATH}" \
${GVIRTUS_HOME}/bin/gvirtus-backend ${GVIRTUS_HOME}/etc/properties.json

---

### Modified commands to compile openpose with using GVirtuS environment.
     ```bash
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_CAFFE=ON \
  -DBUILD_EXAMPLES=ON \
  -DBUILD_DOCS=OFF \
  -DCUDA_USE_STATIC_CUDA_RUNTIME=OFF \
  -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.6 \
  -DCUDA_NVCC_EXECUTABLE=/usr/local/cuda-12.6/bin/nvcc \
  -DCUDA_CUDART_LIBRARY=/home/darshan/GVirtuS/lib/frontend/libcudart.so \
  -DCUDA_LIBRARY=/home/darshan/GVirtuS/lib/frontend/libcuda.so \
  -DCUDA_rt_LIBRARY=/usr/lib/x86_64-linux-gnu/librt.so \
  -DCUDA_INCLUDE_DIRS="/home/darshan/GVirtuS/include;/usr/local/cuda-12.6/include" \
  -DCMAKE_CXX_FLAGS="-L/home/darshan/GVirtuS/lib -L/home/darshan/GVirtuS/lib/frontend -lgvirtus-common -lgvirtus-frontend -lgvirtus-communicators -lgvirtus-plugin-cuda -lgvirtus-plugin-cudart -lgvirtus-plugin-cublas -lgvirtus-plugin-cudnn -lgvirtus-plugin-curand -lgvirtus-plugin-cufft -lgvirtus-plugin-cusolver -lgvirtus-plugin-cusparse -lgvirtus-plugin-nvrtc -lcudart -lcuda -lcublas -lcudnn -lcurand -lcufft -lcusolver -lcusparse -lnvrtc"
   ```
### Build command for openpoes-gvirtus integration script
    ```bash
nvcc 01_test.cu -o openpose_demo_gvirtus \
  -std=c++14 \
  -I/home/darshan/openpose/include \
  -I/usr/include/opencv4 \
  -I/home/darshan/openpose/3rdparty/caffe/include \
  -I/home/darshan/openpose/3rdparty/caffe/build/include \
  -L/home/darshan/openpose/build/src/openpose -lopenpose \
  -L/home/darshan/openpose/build/caffe/lib -lcaffe \
  -L${GVIRTUS_HOME}/lib \
  -L${GVIRTUS_HOME}/lib/frontend \
  -lgvirtus-frontend -lgvirtus-communicators -lgvirtus-common \
  -lcuda -lcudart -lcublas -lcufft -lcudnn -lcurand \
  -lgflags -lglog \
  -Xcompiler -pthread \
  $(pkg-config --cflags --libs opencv4)
```bash

---

