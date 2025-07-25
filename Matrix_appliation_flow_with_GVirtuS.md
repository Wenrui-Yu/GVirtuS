# 🚀 Matrix Multiplication with GVirtuS

This example demonstrates how to perform a simple CUDA-based matrix multiplication using the **GVirtuS framework**, which allows remote GPU computation through a frontend-backend architecture.

---

## 📦 1. Prerequisites

* ✅ GVirtuS properly installed (both frontend and backend)
* ✅ NVIDIA GPU and CUDA driver installed on backend system
* ✅ CUDA Toolkit (e.g. `/usr/local/cuda-12.6`)
* ✅ Backend should be running before starting the app

---

## 🛠️ 2. Compile the Matrix Multiplication Code

```bash
nvcc -o matrix_mul_gvirtus matrix_mul.cu \
  -L ${GVIRTUS_HOME}/lib/frontend \
  -L ${GVIRTUS_HOME}/lib \
  -lcudart --cudart=shared
```

### 🔍 What This Does

* Compiles your CUDA `.cu` file using `nvcc`
* Links it against **GVirtuS's custom `libcudart.so`** in the frontend
* Ensures your CUDA runtime calls will be **intercepted by GVirtuS**

---

## ▶️ 3. Run the Application

```bash
LD_LIBRARY_PATH=${GVIRTUS_HOME}/lib/frontend \
LD_PRELOAD=${GVIRTUS_HOME}/lib/frontend/libcuda.so \
./matrix_mul_gvirtus
```

---

## 🔄 4. GVirtuS Execution Flow

Here’s what happens under the hood when the matrix multiplication runs:

---

### 🧭 **Step-by-Step Flow**

#### 1️⃣ **Application Launches**

```cpp
cudaMalloc(&A_d, size);
cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
```

These CUDA APIs are called from your app.

#### 2️⃣ **Frontend Intercepts Calls**

GVirtuS’s custom `libcudart.so` captures each CUDA function like:

* `cudaMalloc`
* `cudaMemcpy`
* `cudaLaunchKernel`

These are serialized into a **binary message** and sent to the backend over a socket.

#### 3️⃣ **Backend Executes on Real GPU**

GVirtuS backend receives the request and runs:

```cpp
cuMemAlloc()
cuMemcpyHtoD()
cuLaunchKernel()
```

on the actual **physical GPU** using NVIDIA’s CUDA driver.

#### 4️⃣ **Result Returns to Frontend**

The backend packs the result (if any), sends it back to the frontend, and your app continues execution — as if it was running locally on the GPU.

---

### Sample Output

```bash
Running Matrix Multiplication with GVirtuS...
Allocating memory on GPU...
Copying matrices A and B to GPU...
Launching CUDA kernel...
Copying result matrix C back to host...
Matrix multiplication completed successfully.
```

---

### Summary

    🟢 Frontend: Builds and runs the CUDA app, but links to GVirtuS’s libcudart.so

    🔁 Intercept: GVirtuS captures CUDA calls, sends them to the backend

    🔧 Backend: Actually executes GPU logic using NVIDIA’s driver (libcuda.so)

    📦 Response: Returns results to the frontend for use in your app

    🚀 GPU: Is only ever touched by the backend — transparent to the frontend

-----
### Schematic Flow:

┌───────────────────────────────────────────────┐
│          1. App Launch (Frontend)             │
│                                               │
│ ➤ Runs CUDA code (e.g., cudaMalloc, memcpy)   │
└───────────────────────────────────────────────┘
                     │
                     ▼
┌───────────────────────────────────────────────┐
│         2. GVirtuS Frontend Intercepts        │
│                                               │
│ ➤ libcudart.so wraps the CUDA API             │
│ ➤ Serializes function name + arguments        │
│ ➤ Sends request buffer to GVirtuS backend     │
└───────────────────────────────────────────────┘
                     │
                     ▼
┌───────────────────────────────────────────────┐
│             3. Backend Dispatches             │
│                                               │
│ ➤ Receives buffer via socket                  │
│ ➤ Looks up correct plugin handler             │
│ ➤ Calls real CUDA Driver API (cu*)            │
└───────────────────────────────────────────────┘
                     │
                     ▼
┌───────────────────────────────────────────────┐
│             4. GPU Executes Task              │
│                                               │
│ ➤ Executes kernel or memory ops               │
│ ➤ Returns result to backend                   │
└───────────────────────────────────────────────┘
                     │
                     ▼
┌───────────────────────────────────────────────┐
│           5. Backend Sends Result             │
│                                               │
│ ➤ Packs return value into buffer              │
│ ➤ Sends result back to frontend               │
└───────────────────────────────────────────────┘
                     │
                     ▼
┌───────────────────────────────────────────────┐
│        6. Frontend Returns to App Logic       │
│                                               │
│ ➤ Unpacks result                              │
│ ➤ Application logic resumes                   │
└───────────────────────────────────────────────┘
