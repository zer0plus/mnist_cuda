#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "../kernels.cuh"
#include "../consts.cuh"

#define HIDDEN_LAYER_TEST_SIZE (HIDDEN_LAYER_SIZE * 1024 * 1024)
#define GRAD_LAYER_TEST_SIZE (HIDDEN_LAYER_SIZE * 1024 * 1024)

void relu_cpu(float *out, int size) {
    for (int i = 0; i < size; i++) {
        out[i] = out[i] > 0 ? out[i] : 0;
    }
}

void relu_derivative_cpu(float *grad, float *out, int size) {
    for (int i = 0; i < size; i++) {
        grad[i] *= out[i] > 0 ? 1 : 0;
    }
}

void initialize_data(float *data, int size) {
    for (int i = 0; i < size; i++) {
        data[i] = (float)(rand() % 20 - 10) / 2.0f; // Random values between -5 and 5
    }
}

int compare_arrays(float *a, float *b, int size) {
    for (int i = 0; i < size; i++) {
        if (abs(a[i] - b[i]) > 1e-5) {
            printf("Mismatch at index %d: %f != %f\n", i, a[i], b[i]);
            return 0;
        }
    }
    return 1;
}


#ifdef RUN_RELU_TEST
int main() {
    printf("\nReLU Test with data size: %d \n", HIDDEN_LAYER_TEST_SIZE);
    int block_size = 256;
    int num_blocks = (HIDDEN_LAYER_TEST_SIZE + block_size - 1) / block_size;
    float *h_hidden_cpu = (float *)malloc(HIDDEN_LAYER_TEST_SIZE * sizeof(float));
    float *h_hidden_cuda = (float *)malloc(HIDDEN_LAYER_TEST_SIZE * sizeof(float));
    float *h_hidden_derivative_cpu = (float *)malloc(HIDDEN_LAYER_TEST_SIZE * sizeof(float));
    float *h_hidden_derivative_cuda = (float *)malloc(HIDDEN_LAYER_TEST_SIZE * sizeof(float));
    float *h_grad_cpu = (float *)malloc(GRAD_LAYER_TEST_SIZE * sizeof(float));
    float *h_grad_cuda = (float *)malloc(GRAD_LAYER_TEST_SIZE * sizeof(float));
    
    float *d_hidden_relu;
    float *d_hidden_derivative_relu;
    float *d_grad_relu;

    srand(42);
    initialize_data(h_hidden_cpu, HIDDEN_LAYER_TEST_SIZE);
    initialize_data(h_hidden_derivative_cpu, HIDDEN_LAYER_TEST_SIZE);
    initialize_data(h_grad_cpu, GRAD_LAYER_TEST_SIZE);

    // Allocate and copy data to GPU
    cudaMalloc((void **)&d_hidden_relu, HIDDEN_LAYER_TEST_SIZE * sizeof(float));
    cudaMalloc((void **)&d_hidden_derivative_relu, HIDDEN_LAYER_TEST_SIZE * sizeof(float));
    cudaMalloc((void **)&d_grad_relu, GRAD_LAYER_TEST_SIZE * sizeof(float));
    cudaMemcpy(d_hidden_relu, h_hidden_cpu, HIDDEN_LAYER_TEST_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_hidden_derivative_relu, h_hidden_derivative_cpu, HIDDEN_LAYER_TEST_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_grad_relu, h_grad_cpu, GRAD_LAYER_TEST_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    // relu cpu
    cudaEvent_t start_cpu, stop_cpu, start_deriv_cpu, stop_deriv_cpu;
    cudaEventCreate(&start_cpu);
    cudaEventCreate(&stop_cpu);
    cudaEventRecord(start_cpu);
    relu_cpu(h_hidden_cpu, HIDDEN_LAYER_TEST_SIZE);
    cudaEventRecord(stop_cpu);
    cudaEventSynchronize(stop_cpu);
    float cpu_time = 0;
    cudaEventElapsedTime(&cpu_time, start_cpu, stop_cpu);

    // relu deriv cpu
    cudaEventCreate(&start_deriv_cpu);
    cudaEventCreate(&stop_deriv_cpu);
    cudaEventRecord(start_deriv_cpu);
    relu_derivative_cpu(h_hidden_derivative_cpu, h_grad_cpu, HIDDEN_LAYER_TEST_SIZE);
    cudaEventRecord(stop_deriv_cpu);
    cudaEventSynchronize(stop_deriv_cpu);
    float cpu_deriv_time = 0;
    cudaEventElapsedTime(&cpu_deriv_time, start_deriv_cpu, stop_deriv_cpu);

    // relu gpu
    cudaEvent_t start_gpu, stop_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);
    cudaEventRecord(start_gpu);
    const int vector_size = 4;
    const int vector_block_size = 256;
    const int vector_grid_size = ((HIDDEN_LAYER_TEST_SIZE + vector_size - 1) / vector_size + vector_block_size - 1) / vector_block_size;
    optimized_relu_kernel<<<vector_grid_size, vector_block_size>>>(d_hidden_relu, HIDDEN_LAYER_TEST_SIZE);
    // relu_kernel<<<num_blocks, block_size>>>(d_hidden_relu, HIDDEN_LAYER_TEST_SIZE);
    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);
    float gpu_time = 0;
    cudaEventElapsedTime(&gpu_time, start_gpu, stop_gpu);

    // relu deriv gpu
    cudaEvent_t start_deriv_gpu, stop_deriv_gpu;
    cudaEventCreate(&start_deriv_gpu);
    cudaEventCreate(&stop_deriv_gpu);
    cudaEventRecord(start_deriv_gpu);
    optimized_relu_derivative_kernel<<<vector_grid_size, vector_block_size>>>(d_hidden_derivative_relu, d_grad_relu, HIDDEN_LAYER_TEST_SIZE);
    // relu_derivative_kernel<<<num_blocks, block_size>>>(d_hidden_derivative_relu, d_grad_relu, HIDDEN_LAYER_TEST_SIZE);
    cudaEventRecord(stop_deriv_gpu);
    cudaEventSynchronize(stop_deriv_gpu);
    float gpu_deriv_time = 0;
    cudaEventElapsedTime(&gpu_deriv_time, start_deriv_gpu, stop_deriv_gpu);

    // Copy results back to host
    cudaMemcpy(h_hidden_cuda, d_hidden_relu, HIDDEN_LAYER_TEST_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_hidden_derivative_cuda, d_hidden_derivative_relu, HIDDEN_LAYER_TEST_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    // Compare results
    int result = compare_arrays(h_hidden_cpu, h_hidden_cuda, HIDDEN_LAYER_TEST_SIZE);
    if (result) {
        printf("\nReLU Test Passed: CPU and CUDA results match.\n");
    } else {
        printf("\nReLU Test Failed: CPU and CUDA results do not match.\n");
    }

    printf("CPU time: %f ms\n", cpu_time);
    printf("GPU time: %f ms\n", gpu_time);
    printf("Speedup: %fx\n", cpu_time / gpu_time);
    
    int result2 = compare_arrays(h_hidden_derivative_cpu, h_hidden_derivative_cuda, HIDDEN_LAYER_TEST_SIZE);
    if (result2) {
        printf("\nReLU Derivative Test Passed: CPU and CUDA results match.\n");
    } else {
        printf("\nReLU Derivative Test Failed: CPU and CUDA results do not match.\n");
    }
    printf("CPU time: %f ms\n", cpu_deriv_time);
    printf("GPU time: %f ms\n", gpu_deriv_time);
    printf("Speedup: %fx\n", cpu_deriv_time / gpu_deriv_time);

    // Cleanup
    free(h_hidden_cpu);
    free(h_hidden_cuda);
    free(h_hidden_derivative_cpu);
    free(h_hidden_derivative_cuda);
    free(h_grad_cpu);
    free(h_grad_cuda);
    cudaFree(d_hidden_relu);
    cudaFree(d_hidden_derivative_relu);
    cudaFree(d_grad_relu);
    cudaEventDestroy(start_cpu);
    cudaEventDestroy(start_deriv_cpu);
    cudaEventDestroy(stop_cpu);
    cudaEventDestroy(stop_deriv_cpu);
    cudaEventDestroy(start_gpu);
    cudaEventDestroy(start_deriv_gpu);
    cudaEventDestroy(stop_gpu);
    cudaEventDestroy(stop_deriv_gpu);

    return 0;
}
#endif