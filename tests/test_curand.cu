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


//curandSetGeneratorOffset
TEST(cuRAND, SetGeneratorOffset) {
    curandGenerator_t gen1, gen2;

    // Create 2 identical generators
    CURAND_CHECK(curandCreateGenerator(&gen1, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandCreateGenerator(&gen2, CURAND_RNG_PSEUDO_DEFAULT));

    // Set same seed
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen1, 1234ULL));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen2, 1234ULL));

    // Set different offsets
    size_t offset = 1000;
    CURAND_CHECK(curandSetGeneratorOffset(gen1, offset));

    // Generate same count from both, but gen2 will generate offset + count
    const int count = 10;
    std::vector<float> out1(count), out2(offset + count);

    float *d_out1, *d_out2;
    cudaMalloc(&d_out1, count * sizeof(float));
    cudaMalloc(&d_out2, (offset + count) * sizeof(float));

    // Generate for both
    CURAND_CHECK(curandGenerateUniform(gen1, d_out1, count));
    CURAND_CHECK(curandGenerateUniform(gen2, d_out2, offset + count));

    // Copy and compare
    cudaMemcpy(out1.data(), d_out1, count * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(out2.data(), d_out2, (offset + count) * sizeof(float), cudaMemcpyDeviceToHost);

    // The output of gen1 should match gen2's values starting at [offset]
    for (int i = 0; i < count; ++i) {
        EXPECT_FLOAT_EQ(out1[i], out2[i + offset]);
    }

    CURAND_CHECK(curandDestroyGenerator(gen1));
    CURAND_CHECK(curandDestroyGenerator(gen2));
    cudaFree(d_out1);
    cudaFree(d_out2);
}