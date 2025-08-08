#!/bin/bash
set -e

export OPENPOSE_ROOT=/opt/openpose
export GVIRTUS_HOME=/opt/GVirtuS
export LD_LIBRARY_PATH=$OPENPOSE_ROOT/build/src/openpose:$GVIRTUS_HOME/lib:$GVIRTUS_HOME/lib/frontend:$LD_LIBRARY_PATH

echo "🛠️ Compiling OpenPose test..."
cd /opt/openpose/examples/gvirtus

nvcc 00_test.cpp -g -o 00_test \
  -I$OPENPOSE_ROOT/include \
  -I$OPENPOSE_ROOT/3rdparty/caffe/include \
  -L$OPENPOSE_ROOT/build/src/openpose \
  -L$OPENPOSE_ROOT/build/caffe/lib \
  -lopenpose -lcaffe -lgflags $(pkg-config --cflags --libs opencv4)

echo "🚀 Running OpenPose test..."
cd $OPENPOSE_ROOT
./examples/gvirtus/00_test
