#include <stdio.h>
#include <stdlib.h>

#define N 512  // Matrix size: N x N

// CUDA kernel for matrix multiplication
__global__ void matrixMul(float *A, float *B, float *C, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // Row index
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // Column index

    if (row < width && col < width) {
        float sum = 0.0f;
        for (int k = 0; k < width; ++k) {
            sum += A[row * width + k] * B[k * width + col];
        }
        C[row * width + col] = sum;
    }
}

// Host function to initialize matrices
void initializeMatrix(float *mat, int size) {
    for (int i = 0; i < size; ++i) {
        mat[i] = rand() % 10;  // Random float between 0-9
    }
}

// Host function to verify the result (optional)
void verifyResult(float *A, float *B, float *C, int width) {
    for (int row = 0; row < width; ++row) {
        for (int col = 0; col < width; ++col) {
            float expected = 0.0f;
            for (int k = 0; k < width; ++k) {
                expected += A[row * width + k] * B[k * width + col];
            }
            float diff = abs(C[row * width + col] - expected);
            if (diff > 1e-4) {
                printf("Mismatch at (%d, %d): GPU = %f, CPU = %f\n",
                       row, col, C[row * width + col], expected);
                return;
            }
        }
    }
    printf("Result verified successfully!\n");
}

int main() {
    int size = N * N;
    size_t bytes = size * sizeof(float);

    // Allocate host memory
    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C = (float *)malloc(bytes);

    // Initialize input matrices
    initializeMatrix(h_A, size);
    initializeMatrix(h_B, size);

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, bytes);
    cudaMalloc((void **)&d_B, bytes);
    cudaMalloc((void **)&d_C, bytes);

    // Copy data to device
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (N + blockDim.y - 1) / blockDim.y);

    // Launch the kernel
    matrixMul<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);

    // Copy result back to host
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    // Verify result
    verifyResult(h_A, h_B, h_C, N);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
