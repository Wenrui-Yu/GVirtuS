Here's a clean, well-structured version of your notes for a GitHub README or documentation file:

---

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

| Approach                        | Status | Notes                                 |
| ------------------------------- | ------ | ------------------------------------- |
| Docker GUI Setup                | ✅      | Works with `xhost` and proper volumes |
| OpenPose CLI (No GVirtuS)       | ✅      | Successful                            |
| OpenPose CLI (With GVirtuS)     | ❌      | Fails to link correctly               |
| CPU-Only OpenPose with GVirtuS  | ⚠️     | Compiles and runs, but poor detection |
| GPU Virtualization with GVirtuS | ❌      | Not currently supported               |

---

Let me know if you'd like this formatted as a `README.md` or need help testing alternatives.
