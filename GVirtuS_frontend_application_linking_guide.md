# GVirtuS Frontend Application Linking Guide

This guide walks through how to properly set up and verify GVirtuS integration with your CUDA frontend application. It ensures that fake CUDA libraries provided by GVirtuS are correctly linked and used during compilation and execution.

---

## 🔧 Step 1: Locate GVirtuS Installation

Before compiling any frontend application with GVirtuS, locate the directory where GVirtuS is installed. Typically, this includes subdirectories like `lib/`, `lib/frontend/`, `plugins/`, and `etc/`.

Example installation path:

```

/home/GVirtuS/

````

Set this path as your GVIRTUS_HOME:

```bash
export GVIRTUS_HOME=/home/GVirtuS
````

---

## 📂 Step 2: Verify Frontend Library Path

GVirtuS uses fake CUDA libraries in its `lib/frontend/` folder to intercept CUDA calls. Ensure the path exists:

```bash
ls ${GVIRTUS_HOME}/lib/frontend
```

Expected output (partial):

```
libcudart.so
libcudart.so.12
libcudnn.so
libcudnn.so.9
libcublas.so
libcublas.so.12
libcusolver.so
libnvrtc.so
...
```

If these files exist, you're ready to continue.

---

## 🧬 Step 3: Set Library Path Environment Variable

Update your `LD_LIBRARY_PATH` to include GVirtuS libraries:

```bash
export LD_LIBRARY_PATH=${GVIRTUS_HOME}/lib:${GVIRTUS_HOME}/lib/frontend:${LD_LIBRARY_PATH}
```

Verify:

```bash
echo $LD_LIBRARY_PATH
```

Example output:

```
/home/GVirtuS/lib:/home/GVirtuS/lib/frontend:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
```

This confirms the GVirtuS paths are injected.

---

## 🧪 Step 4: Compile a CUDA Application

Compile your CUDA frontend application using `nvcc`, and explicitly link CUDA libraries that GVirtuS fakes:

```bash
nvcc cnn.cu -o example -L -lcuda -lcudnn --cudart=shared
```

This command links against:

* `cuda`
* `cudnn`
* `cudart`

---

## 🔍 Step 5: Verify Dynamic Linking

After compilation, use `ldd` to inspect which libraries your binary is linked to:

```bash
ldd ./example
```

Example output:

```
libcudart.so.12 => /home/GVirtuS/lib/frontend/libcudart.so.12
libgvirtus-frontend.so => /home/GVirtuS/lib/libgvirtus-frontend.so
libgvirtus-communicators.so => /home/GVirtuS/lib/libgvirtus-communicators.so
...
```

Make sure paths are resolving to **`GVIRTUS_HOME/lib/frontend/`** — this confirms the binary is using GVirtuS to intercept CUDA calls.

---

## 🚀 Step 6: Run the Binary

Run your compiled binary:

```bash
./example
```

If everything is set up correctly, the application will:

* Load GVirtuS frontend
* Redirect CUDA calls to the backend system

---

## ❗ Troubleshooting: Connection Errors

If you see an error like:

```
INFO - GVirtuS frontend version /home/GVirtuS/etc/properties.json
ERROR - "Frontend.cpp":122: Exception occurred: TcpCommunicator: Can't connect to socket: Connection refused.
```

This means:

* GVirtuS frontend can't reach the backend server over the network.

**Solution:**

1. Open `${GVIRTUS_HOME}/etc/properties.json`
2. Set the correct IP address or hostname of your backend machine
3. Restart the application

Example JSON snippet:

```json
{
  "communicator": {
    "type": "tcp",
    "host": "192.168.1.10",
    "port": 8080
  }
}
```

---

## 🎉 You're Done!

You have successfully:

* Linked your CUDA frontend app with GVirtuS fake libraries
* Verified dynamic linking
* Run the app with CUDA offloading to GVirtuS backend

Well done! 🎯

```
