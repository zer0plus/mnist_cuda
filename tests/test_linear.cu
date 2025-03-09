#include "../kernels.cuh"
#include "../mnist.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cublas_v2.h>

#define HIDDEN_LAYER_TEST_SIZE (HIDDEN_LAYER_SIZE * 100)
#define OUT_LAYER_TEST_SIZE (OUTPUT_LAYER_SIZE * 100)

// Debug code to print matrices
void print_matrices(float *inp, float *weights, float *out, int in_size, int out_size) {
    printf("Input vector:\n");
    for (int i = 0; i < min(10, in_size); i++) {
        printf("%.4f ", inp[i]);
    }
    printf("\n\n");
    
    printf("Weights matrix (row-major layout):\n");
    for (int i = 0; i < min(5, in_size); i++) {
        for (int j = 0; j < min(5, out_size); j++) {
            printf("%.4f ", weights[i * out_size + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void print_ele(float* cpu_ele, float* gpu_ele, size_t size) {
    // printf("CPU ele: ");
    // for (int i = 0; i < 20; i++) {
    //     printf("%f ", cpu_ele[i]);
    // }
    // printf("\n");
    
    // printf("GPU ele: ");
    // for (int i = 0; i < 20; i++) {
    //     printf("%f ", gpu_ele[i]);
    // }
    // printf("\n");

    for (int i = 0; i < 100; i++) {
        if (fabs(cpu_ele[i] - gpu_ele[i]) > 1e-5) {
            printf("Mismatch at index %d: CPU = %f, GPU = %f\n", i, cpu_ele[i], gpu_ele[i]);
        }
    }
    // printf("Last element at index %zd: CPU = %f, GPU = %f\n", size-1, cpu_ele[size-1], gpu_ele[size-1]);
    printf("Last 10 elements check\n");

    for (int i = size-1; size-11 < i; i--) {
        if (fabs(cpu_ele[i] - gpu_ele[i]) > 1e-5) {
            printf("Mismatch at index %d: CPU = %f, GPU = %f\n", i, cpu_ele[i], gpu_ele[i]);
        }
    }
}

int compare_arrays(float *a, float *b, size_t size) {
    for (size_t i = 0; i < size; i++) {
        if (abs(a[i] - b[i]) > 1e-4) {
            return 0;
        }
    }
    return 1;
}

void initialize_data(float *data, size_t size) {
    for (size_t i = 0; i < size; i++) {
        data[i] = (float)rand() / RAND_MAX; // Random values between 0 and 1
    }
}


#ifdef RUN_LINEAR_TEST
int main() {
    // create dummy linear layers with different in and out test sizes
    // init cpu layer with data and copy it to gpu to make data same
    init_cublas();

    srand(time(NULL));
    printf("\nLinear Test with data size: %d \n", HIDDEN_LAYER_TEST_SIZE);
    float *h_out_cpu, *h_hidden_cpu, *h_out_cuda, *h_hidden_cuda, *h_input;
    CUDA_CHECK(cudaMallocHost((void **) &h_input, HIDDEN_LAYER_TEST_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMallocHost((void **) &h_out_cuda, OUT_LAYER_TEST_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMallocHost((void **) &h_out_cpu, OUT_LAYER_TEST_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMallocHost((void **) &h_hidden_cuda, HIDDEN_LAYER_TEST_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMallocHost((void **) &h_hidden_cpu, HIDDEN_LAYER_TEST_SIZE * sizeof(float)));

    float *d_input, *d_out, *d_hidden;
    CUDA_CHECK(cudaMalloc((void **) &d_input, HIDDEN_LAYER_TEST_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **) &d_out, OUT_LAYER_TEST_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **) &d_hidden, HIDDEN_LAYER_TEST_SIZE * sizeof(float)));

    GenericLayer cpu_hidden;
    GenericLayer gpu_hidden;

    initialize_data(h_input, HIDDEN_LAYER_TEST_SIZE);
    init_layer(&cpu_hidden, HIDDEN_LAYER_TEST_SIZE, OUT_LAYER_TEST_SIZE);
    gpu_hidden.in_size = cpu_hidden.in_size;
    gpu_hidden.out_size = cpu_hidden.out_size;
    size_t num_layer_elements = cpu_hidden.in_size * cpu_hidden.out_size;
    
    CUDA_CHECK(cudaMemcpy(d_input, h_input, HIDDEN_LAYER_TEST_SIZE * sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc((void **) &gpu_hidden.weights, num_layer_elements * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **) &gpu_hidden.biases, OUT_LAYER_TEST_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(gpu_hidden.weights, cpu_hidden.weights, num_layer_elements * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu_hidden.biases, cpu_hidden.biases, OUT_LAYER_TEST_SIZE * sizeof(float), cudaMemcpyHostToDevice));

    // Print CPU hidden layer weights and biases
    printf("CPU Hidden Layer Weights:\n");
    for (int i = 0; i < 20; i++) {
        printf("%.4f ", cpu_hidden.weights[i]);
    }
    printf("\n");
    printf("CPU Hidden Layer Biases:\n");
    for (int i = 0; i < 10; i++) {
        printf("%.4f ", cpu_hidden.biases[i]);
    }
    printf("\n");

    // Print GPU hidden layer weights and biases
    print_device_tensor("GPU Hidden Layer Weights", gpu_hidden.weights, num_layer_elements, 20);
    print_device_tensor("GPU Hidden Layer Biases", gpu_hidden.biases, OUT_LAYER_TEST_SIZE, 10);

    cudaEvent_t start_cpu, stop_cpu;
    float cpu_time;
    cudaEventCreate(&start_cpu);
    cudaEventCreate(&stop_cpu);
    cudaEventRecord(start_cpu);
    linear(&cpu_hidden, h_input, h_out_cpu);
    cudaEventRecord(stop_cpu);
    cudaEventSynchronize(stop_cpu);
    cudaEventElapsedTime(&cpu_time, start_cpu, stop_cpu);

    print_matrices(h_input, cpu_hidden.weights, h_out_cpu, HIDDEN_LAYER_TEST_SIZE, OUT_LAYER_TEST_SIZE);

    cudaEvent_t start_gpu, stop_gpu;
    float gpu_time;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);
    cudaEventRecord(start_gpu);
    linear_cuda_cublas(&gpu_hidden, d_input, d_out);
    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);
    cudaEventElapsedTime(&gpu_time, start_gpu, stop_gpu);

    CUDA_CHECK(cudaMemcpy(h_out_cuda, d_out, OUT_LAYER_TEST_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    int result = compare_arrays(h_out_cpu, h_out_cuda, OUT_LAYER_TEST_SIZE);
    if (result) {
        printf("\nLinear Test Passed: CPU and CUDA results match.\n");
    } else {
        printf("\nLinear Test Failed: CPU and CUDA results do not match.\n");
    }
   
       // Before running the linear operation
    print_device_tensor("GPU Input", d_input, HIDDEN_LAYER_TEST_SIZE, 20);
    // After running the linear operation
    print_device_tensor("GPU Output", d_out, OUT_LAYER_TEST_SIZE, 20);

    print_ele(h_out_cpu, h_out_cuda, OUT_LAYER_TEST_SIZE);
    printf(" CPU time: %f ms\n", cpu_time);
    printf(" GPU time: %f ms\n", gpu_time);
    printf(" Speedup: %fx\n", cpu_time / gpu_time);

    cudaFreeHost(h_input);
    cudaFreeHost(h_out_cuda);
    cudaFreeHost(h_out_cpu);
    cudaFreeHost(h_hidden_cuda);
    cudaFreeHost(h_hidden_cpu);
    cudaFree(d_input);
    cudaFree(d_out);
    cudaFree(d_hidden);
}
#endif