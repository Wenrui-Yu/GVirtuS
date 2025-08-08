#!/bin/bash

# --- Configurable variables ---
IMAGE_NAME="openpose_img"
CONTAINER_NAME="openpose_test_container"
LOCAL_PROPERTIES="./openpose-gvirtus/properties.json"
LOCAL_SCRIPT="./openpose-gvirtus/00_test.cpp"
OPENPOSE_SCRIPT_DIR="./openpose-gvirtus/openpose_scripts"  # optional dir for scripts
MEDIA_FOLDER="./media"

GVIRTUS_TARGET_DIR="/opt/GVirtuS/etc"
OPENPOSE_EXAMPLES_DIR="/opt/openpose/examples/gvirtus"
MEDIA_FOLDER="./openpose-gvirtus/media/"
MEDIA_MOUNT_TARGET="/opt/openpose/examples/media/"


# 🐳 Start container in detached mode with media folder mounted
docker run -dit --rm \
    --name $CONTAINER_NAME \
    --network host \
    -v "${MEDIA_FOLDER}:${MEDIA_MOUNT_TARGET}" \
    $IMAGE_NAME \
    bash

echo "✅ Container started: $CONTAINER_NAME"

# 📁 Copy properties.json into container
docker cp "$LOCAL_PROPERTIES" "$CONTAINER_NAME:$GVIRTUS_TARGET_DIR/properties.json"
echo "📁 Copied properties.json into container"

# 📁 Ensure examples dir exists
docker exec $CONTAINER_NAME mkdir -p "$OPENPOSE_EXAMPLES_DIR"

# 📁 Copy 00_test.cpp into examples
docker cp "$LOCAL_SCRIPT" "$CONTAINER_NAME:$OPENPOSE_EXAMPLES_DIR/00_test.cpp"
echo "📁 Copied 00_test.cpp into OpenPose examples"

# 📁 Copy any additional scripts
if [ -d "$OPENPOSE_SCRIPT_DIR" ]; then
  docker cp "$OPENPOSE_SCRIPT_DIR/." "$CONTAINER_NAME:$OPENPOSE_EXAMPLES_DIR/"
  echo "📁 Additional scripts copied to OpenPose examples"
fi

# 🛠️ Compile in examples/gvirtus, then run from openpose root
docker exec $CONTAINER_NAME bash -c "
  export OPENPOSE_ROOT=/opt/openpose && \
  export GVIRTUS_HOME=/opt/GVirtuS && \
  export LD_LIBRARY_PATH=\$OPENPOSE_ROOT/build/src/openpose:\$GVIRTUS_HOME/lib:\$GVIRTUS_HOME/lib/frontend:\$LD_LIBRARY_PATH && \

  echo '🛠️ Compiling...' && \
  cd $OPENPOSE_EXAMPLES_DIR && \
  nvcc 00_test.cpp -g -o 00_test \
    -I\$OPENPOSE_ROOT/include \
    -I\$OPENPOSE_ROOT/3rdparty/caffe/include \
    -L\$OPENPOSE_ROOT/build/src/openpose \
    -L\$OPENPOSE_ROOT/build/caffe/lib \
    -lopenpose -lcaffe -lgflags \$(pkg-config --cflags --libs opencv4) && \

  echo '🚀 Running from \$OPENPOSE_ROOT...' && \
  cd \$OPENPOSE_ROOT && \
  ./examples/gvirtus/00_test
"
