#include "../kernels.cuh"
#include "../mnist.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cublas_v2.h>

void initialize_data(float *data, size_t size) {
    for (size_t i = 0; i < size; i++) {
        data[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f; // Random values between -1 and 1
    }
}

int compare_arrays(float *a, float *b, size_t size) {
    for (size_t i = 0; i < size; i++) {
        if (fabs(a[i] - b[i]) > 1e-4) {
            return 0;
        }
    }
    return 1;
}

#ifdef RUN_LINEAR_TEST
int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <input_size> <output_size>\n", argv[0]);
        return 1;
    }
    
    int in_size = atoi(argv[1]);
    int out_size = atoi(argv[2]);
    
    if (in_size <= 0 || out_size <= 0) {
        fprintf(stderr, "Invalid size arguments. All sizes must be positive.\n");
        return 1;
    }

    printf("LINEAR_RESULTS:\n");
    printf("InputSize: %d\n", in_size);
    printf("OutputSize: %d\n", out_size);
    
    // Fixed seed for reproducibility
    srand(42);
    
    // Allocate host memory
    float *h_input = (float *)malloc(in_size * sizeof(float));
    float *h_output_cpu = (float *)malloc(out_size * sizeof(float));
    float *h_output_cublas = (float *)malloc(out_size * sizeof(float));
    
    // Initialize input data
    initialize_data(h_input, in_size);
    
    // Create CPU layer
    GenericLayer cpu_layer = {0};
    GenericLayer gpu_layer = {0};
    
    // Initialize CPU layer
    cpu_layer.in_size = in_size;
    cpu_layer.out_size = out_size;
    cpu_layer.weights = (float *)malloc(in_size * out_size * sizeof(float));
    cpu_layer.biases = (float *)malloc(out_size * sizeof(float));
    
    // Initialize with random values using He initialization
    float scale = sqrtf(2.0f / in_size);
    for (size_t i = 0; i < in_size * out_size; i++) {
        cpu_layer.weights[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale;
    }
    for (size_t i = 0; i < out_size; i++) {
        cpu_layer.biases[i] = 0.0f;  // Initialize biases to zero
    }
    
    // Initialize GPU layer with same values
    gpu_layer.in_size = in_size;
    gpu_layer.out_size = out_size;
    
    CUDA_CHECK(cudaMalloc((void **)&gpu_layer.weights, in_size * out_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&gpu_layer.biases, out_size * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(gpu_layer.weights, cpu_layer.weights, in_size * out_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu_layer.biases, cpu_layer.biases, out_size * sizeof(float), cudaMemcpyHostToDevice));
    
    // Allocate device memory for input and output
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc((void **)&d_input, in_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_output, out_size * sizeof(float)));
    
    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input, in_size * sizeof(float), cudaMemcpyHostToDevice));
    
    // Initialize cuBLAS
    init_cublas();
    
    // Create events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Warmup runs
    for (int i = 0; i < 5; i++) {
        linear(&cpu_layer, h_input, h_output_cpu);
        linear_cuda_cublas(&gpu_layer, d_input, d_output);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    // Benchmark CPU implementation
    float cpu_time;
    cudaEventRecord(start);
    linear(&cpu_layer, h_input, h_output_cpu);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cpu_time, start, stop);
    
    // Benchmark cuBLAS implementation
    float gpu_time;
    cudaEventRecord(start);
    linear_cuda_cublas(&gpu_layer, d_input, d_output);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);
    
    // Copy results back to host for validation
    CUDA_CHECK(cudaMemcpy(h_output_cublas, d_output, out_size * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Validate results
    int result = compare_arrays(h_output_cpu, h_output_cublas, out_size);
    
    // Calculate speedup
    float speedup = cpu_time / gpu_time;
    
    // Print results
    printf("CPU_Time: %.4f\n", cpu_time);
    printf("GPU_Time: %.4f\n", gpu_time);
    printf("Speedup: %.4f\n", speedup);
    printf("Valid: %d\n", result);
    
    if (result) {
        printf("Linear Test Passed: CPU and CuBLAS results match.\n");
    } else {
        printf("Linear Test Failed: CPU and CuBLAS results do not match.\n");
    }
    
    printf("END_RESULTS\n");
    
    // Cleanup
    free(h_input);
    free(h_output_cpu);
    free(h_output_cublas);
    free(cpu_layer.weights);
    free(cpu_layer.biases);
    cudaFree(gpu_layer.weights);
    cudaFree(gpu_layer.biases);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    // Clean up cuBLAS
    cleanup_cublas();
    
    return 0;
}
#endif