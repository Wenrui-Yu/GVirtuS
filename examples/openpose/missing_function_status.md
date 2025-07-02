
---

## 🟢 libcudart.so.12 (CUDA Runtime) – GVirtuS Support Matrix

| Function Name                 | Implemented | Tested | Working | Notes                      |
| ----------------------------- | ----------- | ------ | ------- | -------------------------- |
| `__cudaRegisterFatBinaryEnd`  | ❓           | ❌      | ❓       |                            |
| `cudaFree`                    | ✅           | ✅      | ✅       | Basic unit test passed     |
| `cudaStreamCreateWithFlags`   | ❓           | ❌      | ❓       |                            |
| `cudaEventElapsedTime`        | ✅           | ✅      | ✅       | Tested in event sync test  |
| `__cudaRegisterFunction`      | ❓           | ❌      | ❓       |                            |
| `cudaGetDeviceProperties_v2`  | ❓           | ❌      | ❓       |                            |
| `cudaMemset`                  | ❓           | ❌      | ❓       |                            |
| `cudaMemGetInfo`              | ❓           | ❌      | ❓       |                            |
| `cudaStreamDestroy`           | ❓           | ❌      | ❓       |                            |
| `cudaGetLastError`            | ❓           | ❌      | ❓       |                            |
| `cudaEventRecord`             | ✅           | ✅      | ✅       | Part of elapsed time test  |
| `cudaMallocHost`              | ❓           | ❌      | ❓       |                            |
| `cudaEventSynchronize`        | ✅           | ✅      | ✅       |                            |


| `__cudaPopCallConfiguration`  | ❓           | ❌      | ❓       |                            |
| `cudaMemcpyAsync`             | ❓           | ❌      | ❓       |                            |
| `cudaGetDevice`               | ❓           | ❌      | ❓       |                            |
| `cudaStreamCreate`            | ❓           | ❌      | ❓       |                            |
| `cudaMemcpy`                  | ❓           | ❌      | ❓       |                            |
| `cudaLaunchKernel`            | ❓           | ❌      | ❓       |                            |
| `cudaFreeHost`                | ❓           | ❌      | ❓       |                            |
| `__cudaPushCallConfiguration` | ❓           | ❌      | ❓       |                            |
| `cudaEventCreate`             | ✅           | ✅      | ✅       | Used in event timing test  |
| `__cudaRegisterFatBinary`     | ❓           | ❌      | ❓       |                            |
| `cudaGetDeviceCount`          | ❓           | ❌      | ❓       |                            |
| `cudaMalloc`                  | ✅           | ✅      | ✅       | Used in cudaFree test      |
| `cudaPeekAtLastError`         | ❓           | ❌      | ❓       |                            |
| `cudaGetErrorString`          | ❓           | ❌      | ❓       |                            |
| `cudaStreamSynchronize`       | ❓           | ❌      | ❓       |                            |
| `__cudaRegisterVar`           | ❓           | ❌      | ❓       |                            |
| `cudaEventDestroy`            | ✅           | ✅      | ✅       | Paired with event creation |
| `cudaSetDevice`               | ❓           | ❌      | ❓       |                            |
| `__cudaUnregisterFatBinary`   | ❓           | ❌      | ❓       |                            |

----
