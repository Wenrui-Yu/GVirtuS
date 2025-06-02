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
