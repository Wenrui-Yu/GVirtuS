
---

## 🟢 libcudart.so.12 (CUDA Runtime) - libcudart missing function

| Function Name                 | Implemented | Tested | Working | Notes                      |
| ----------------------------- | ----------- | ------ | ------- | -------------------------- |
| `__cudaRegisterFatBinaryEnd`  | ✅           | ✅      | ❓       |                            |
| `cudaFree`                    | ✅           | ✅      | ❓       | Basic unit test passed     |
| `cudaStreamCreateWithFlags`   | ✅           | ✅      | ❓       |                            |
| `cudaEventElapsedTime`        | ✅           | ✅      | ❓       |                            |
| `__cudaRegisterFunction`      | ✅           | ✅      | ❓       |                            |
| `cudaGetDeviceProperties_v2`  | ✅           | ✅      | ❓       |                            |
| `cudaMemset`                  | ✅           | ✅      | ❓       |                            |
| `cudaMemGetInfo`              | ✅           | ✅      | ❓       |                            |
| `cudaStreamDestroy`           | ✅           | ✅      | ❓       |                            |
| `cudaGetLastError`            | ❌           | ❌      | ❓       |  Function failed during unit test |
| `cudaEventRecord`             | ✅           | ✅      | ❓       |                            |
| `cudaMallocHost`              | ✅           | ✅      | ❓       |                            |
| `cudaEventSynchronize`        | ✅           | ✅      | ❓       |                            |
| `__cudaPopCallConfiguration`  | ✅           | ✅      | ❓       |                            |
| `cudaMemcpyAsync`             | ✅           | ✅      | ❓       |                            |
| `cudaGetDevice`               | ✅           | ✅      | ❓       |                            |
| `cudaStreamCreate`            | ✅           | ✅      | ❓       |                            |
| `cudaMemcpy`                  | ✅           | ✅      | ❓       |                            |
| `cudaLaunchKernel`            | ❌           | ❌      | ❓       | Function failed during unit test |
| `cudaFreeHost`                | ✅           | ✅      | ❓       |                            |
| `__cudaPushCallConfiguration` | ✅           | ✅      | ❓       |                            |
| `cudaEventCreate`             | ✅           | ✅      | ❓       |                            |
| `__cudaRegisterFatBinary`     | ✅           | ✅      | ❓       |                            |
| `cudaGetDeviceCount`          | ✅           | ✅      | ❓       |                            |
| `cudaMalloc`                  | ✅           | ✅      | ❓       |                            |
| `cudaPeekAtLastError`         | ✅           | ✅      | ❓       |                            |
| `cudaGetErrorString`          | ❌           | ❌      | ❓       |  Function failed during unit test |
| `cudaStreamSynchronize`       | ✅           | ✅      | ❓       |                            |
| `__cudaRegisterVar`           | ✅           | ✅      | ❓       |                            |
| `cudaEventDestroy`            | ✅           | ✅      | ❓       |                            |
| `cudaSetDevice`               | ✅           | ✅      | ❓       |                            |
| `__cudaUnregisterFatBinary`   | ✅           | ✅      | ❓       |                            |

---
