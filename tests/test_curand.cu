#include <gtest/gtest.h>
#include <iostream>
#include <curand.h>

#define CUDA_CHECK(err) ASSERT_EQ((err), cudaSuccess) << "CUDA error: " << cudaGetErrorString(err)
#define CURAND_CHECK(err) ASSERT_EQ((err), CURAND_STATUS_SUCCESS)

//curandCreateGenerator
TEST(cuRAND, CreateGenerator) {
    curandGenerator_t gen;
    CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandDestroyGenerator(gen));
}


//curandDestroyGenerator
TEST(cuRAND, DestroyGenerator) {
    curandGenerator_t gen;
    CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandDestroyGenerator(gen));
}


//curandGenerate
TEST(cuRAND, Generate) {
    curandGenerator_t gen;
    CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));

    const size_t n = 10;
    unsigned int* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, n * sizeof(unsigned int)));

    CURAND_CHECK(curandGenerate(gen, d_data, n));

    CUDA_CHECK(cudaFree(d_data));
    CURAND_CHECK(curandDestroyGenerator(gen));
}

//curandGenerateNormal
TEST(cuRAND, GenerateNormal) {
    curandGenerator_t gen;
    CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));

    const size_t n = 10;
    float* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, n * sizeof(float)));

    CURAND_CHECK(curandGenerateNormal(gen, d_data, n, 0.0f, 1.0f));

    CUDA_CHECK(cudaFree(d_data));
    CURAND_CHECK(curandDestroyGenerator(gen));
}


//curandGenerateNormalDouble
TEST(cuRAND, GenerateNormalDouble) {
    curandGenerator_t gen;
    CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));

    const size_t n = 10;
    double* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, n * sizeof(double)));

    CURAND_CHECK(curandGenerateNormalDouble(gen, d_data, n, 0.0, 1.0));

    CUDA_CHECK(cudaFree(d_data));
    CURAND_CHECK(curandDestroyGenerator(gen));
}


//curandGenerateUniform
TEST(cuRAND, GenerateUniform) {
    curandGenerator_t gen;
    CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));

    const size_t n = 10;
    float* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, n * sizeof(float)));

    CURAND_CHECK(curandGenerateUniform(gen, d_data, n));

    CUDA_CHECK(cudaFree(d_data));
    CURAND_CHECK(curandDestroyGenerator(gen));
}


//curandGenerateUniformDouble
TEST(cuRAND, GenerateUniformDouble) {
    curandGenerator_t gen;
    CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));

    const size_t n = 10;
    double* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, n * sizeof(double)));

    CURAND_CHECK(curandGenerateUniformDouble(gen, d_data, n));

    CUDA_CHECK(cudaFree(d_data));
    CURAND_CHECK(curandDestroyGenerator(gen));
}


//curandSetPseudoRandomGeneratorSeed
TEST(cuRAND, SetPseudoRandomGeneratorSeed) {
    curandGenerator_t gen;
    CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));
    CURAND_CHECK(curandDestroyGenerator(gen));
}
