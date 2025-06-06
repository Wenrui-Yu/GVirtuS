Docker run command:
docker run -it --name openpose_gvirtus_env \
  --network=host \
  --privileged \
  --gpus all \
  --env DISPLAY=$DISPLAY \
  --env QT_X11_NO_MITSHM=1 \
  --volume /tmp/.X11-unix:/tmp/.X11-unix:rw \
  --volume /dev:/dev \
  openpose-gvirtus-image

Access to GUI window through docker:
xhost +local:root

Open docker
docker exec -it openpose_env bash


Error solved!
https://github.com/CMU-Perceptual-Computing-Lab/openpose/issues/2321


Command to run Video with openpose
******Set the MIT-SHM (Shared Memory) Permissions
export MIT_SHM_DISABLE=1
./build/examples/openpose/openpose.bin


**** Important 

Custom Openpose script running with CLI (Without GVIRTUS)

Step 1: Get inside the directory where script locate (cd openpose/examples/gvirtus_api )
g++ 00_test.cpp -o try     -I/home/openpose/include     -I/usr/include/opencv4     -L/home/openpose/build/src/openpose -lopenpose     -lgflags -lglog -lprotobuf -pthread     `pkg-config --cflags --libs opencv4`     -std=c++11 -Wno-unused-result -Wno-write-strings

Step 2: Get back to openpose root directory and run the file
$ openpose
  ./examples/gvirtus_api/try




Custom Openpose script running with CLI (With GVIRTUS)

Step 1: Get inside the directory where script locate (cd openpose/examples/gvirtus_api )
g++ 00_test.cpp -o try     -I/home/openpose/include     -I/usr/include/opencv4     -L/home/openpose/build/src/openpose -lopenpose     -lgflags -lglog -lprotobuf -pthread     `pkg-config --cflags --libs opencv4`     -std=c++11 -Wno-unused-result -Wno-write-strings

Step 2: Get back to openpose root directory and run the file
$ openpose
  ./examples/gvirtus_api/try

Note: not working!!!!!! Still have to figure out


06/06/2025 Openpose-GvirtuS integration
Openpose requires real cuda libraies to build openpose caffe library. So we can't use GvirtuS runtime librarues to compile openpose directly with GVirtuS.

The possible solution would be be, use real cuda libraries to build Openpose. Then using GVirtuS runtime libries compile the frontend application.
But unfortunately this solution also didn't works!
I got the following errors!
/usr/bin/ld: /home/openpose/build/caffe/lib/libcaffe.so: undefined reference to `__cudaRegisterFatBinaryEnd@libcudart.so.12'
/usr/bin/ld: /home/openpose/build/caffe/lib/libcaffe.so: undefined reference to `cudaFree@libcudart.so.12'
/usr/bin/ld: /home/openpose/build/caffe/lib/libcaffe.so: undefined reference to `cublasCreate_v2@libcublas.so.12'

This means your Caffe was compiled with CUDA, but you're now trying to link it against GVirtuS, which doesn't support full CUDA 12.2 symbols or runtime linking like real CUDA does.

The second idea would be disble cuda and compile caffe library with CPU only while doing cmake compailation with openpose.
-- CPU_ONLY = ON
-- USE_CUDA = OFF

For that get into openpose/CMakeLists.txt file and edit line 724 as 'set(CAFFE CPU_ONLY ON)'
And then build openpose with belo instructions,
cd /home/openpose
rm -rf build && mkdir build && cd build

cmake \
  -DBUILD_PYTHON=ON \
  -DUSE_CUDA=OFF \
  -DCPU_ONLY=ON \  # 👈 Pass this through
  -DUSE_CUDNN=OFF \
  -DBUILD_CAFFE=ON \
  -DBUILD_OPENPOSE=ON \
  -DBUILD_SHARED_LIBS=ON \
  -DPYTHON_EXECUTABLE=$(which python3.8) \
  -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.8.so.1.0 \
  ..

After successfull compilation get into 01_test.cu file location and run the below command to build the file with GVirtuS libraries,
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


Note: Observed issue with GPU enable with openpose script, so disable GPU by modifing below block of code.
Update this block in your code:

        // Configure OpenPose
        op::Wrapper opWrapper{op::ThreadManagerMode::Asynchronous};
        if (FLAGS_disable_multi_thread)
            opWrapper.disableMultiThreading();

        opWrapper.start();

To this:

      // CPU-only configuration for older OpenPose
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

      // 👇 THIS IS THE KEY TO CPU MODE
      poseConfig.gpuNumber = 0;     // Use 0 GPUs
      poseConfig.gpuNumberStart = 0;

      // Apply configuration
      opWrapper.configure(poseConfig);
      opWrapper.start();


This compiles and runs with GVirtuS, but i didn't get proper openpose keypoint detection results with this solutin.

The 3rd potential solution would be, enable gpu in the above block of code and allow GVirtuS somehow virtualise cuda calls. or Build caffe library with cuda calls and enable GPU in the script and run with GVirtuS.
