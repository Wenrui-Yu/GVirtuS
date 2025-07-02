
---

## 🟢 libcudart.so.12 (CUDA Runtime) - libcudart missing function

| Function Name                 | Implemented | Tested | Working | Notes                      |
| ----------------------------- | ----------- | ------ | ------- | -------------------------- |
<<<<<<< HEAD
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
=======
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
>>>>>>> 456566b10eb9d221f8c5e60578b61c8b142dff5b

----
