FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Prevent tzdata prompt
RUN ln -fs /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
    echo "Etc/UTC" > /etc/timezone && \
    apt-get update && apt-get install -y tzdata && \
    dpkg-reconfigure -f noninteractive tzdata

# Now install other dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    curl \
    unzip \
    pkg-config \
    libprotobuf-dev \
    protobuf-compiler \
    libgoogle-glog-dev \
    libgflags-dev \
    libatlas-base-dev \
    libboost-all-dev \
    libhdf5-dev \
    libleveldb-dev \
    libsnappy-dev \
    libopencv-dev \
    libopenblas-dev \
    liblmdb-dev \
    python3-dev \
    python3-pip \
    python3-setuptools \
    libxmu-dev \
    libxi-dev \
    libgl-dev \
    libosmesa-dev \
    liblog4cplus-dev \
    librdmacm-dev \
    libibverbs-dev \
    libgtest-dev \
    liblz4-dev \
    cmake \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install opencv-python

# Clone OpenPose
WORKDIR /home
RUN git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose.git --depth=1
WORKDIR /home/openpose

# ✅ Copy custom Cuda.cmake file
COPY cmake/Cuda.cmake /home/openpose/cmake/Cuda.cmake

# ✅ Copy models directory into OpenPose models folder
#COPY models/ /home/openpose/models/

# ✅ Then patch Cuda.cmake AFTER cloning OpenPose (and its 3rdparty/caffe)
COPY caffe/ /home/openpose/3rdparty/caffe/

# Build OpenPose
RUN mkdir build && cd build && \
    cmake -DBUILD_PYTHON=ON \
    -DPYTHON_EXECUTABLE=$(which python3.10) \
    -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.10.so.1.0 .. && \
    make -j$(nproc) && \
    make install


WORKDIR /home
RUN git clone -b gvirtus-cuda-12-ref https://github.com/ecn-aau/GVirtuS.git

WORKDIR /home/GVirtuS
RUN mkdir build && cd build && \
    cmake -DCMAKE_PREFIX_PATH=/home/GVirtuS .. && \
    make -j$(nproc) && \
    make install

ENV GVIRTUS_HOME=/home/GVirtuS
ENV LD_LIBRARY_PATH=${GVIRTUS_HOME}/lib:${GVIRTUS_HOME}/lib/frontend:${LD_LIBRARY_PATH}

WORKDIR /home