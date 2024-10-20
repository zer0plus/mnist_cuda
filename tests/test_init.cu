#include "../kernels.cuh"
// #include "../consts.cuh"
#include "../mnist.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <curand.h>
#include <curand_kernel.h>

// #define HIDDEN_LAYER_SIZE (HIDDEN_LAYER_SIZE)
// #define INPUT_LAYER_SIZE (INPUT_LAYER_SIZE)

void print_ele(float* cpu_ele, float* gpu_ele, size_t size) {
    printf("CPU ele: ");
    for (int i = 0; i < 20; i++) {
        printf("%f ", cpu_ele[i]);
    }
    printf("\n");
    
    printf("GPU ele: ");
    for (int i = 0; i < 20; i++) {
        printf("%f ", gpu_ele[i]);
    }
    printf("\n");

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


void print_raw_bytes(const char* label, void* data, size_t size) {
    unsigned char* bytes = (unsigned char*)data;
    printf("%s: ", label);
    for (size_t i = 0; i < size; i++) {
        printf("%02x ", bytes[i]);
    }
    printf("\n");
}


int compare_arrays(float *a, float *b, size_t size) {
    for (size_t i = 0; i < size; i++) {
        if (fabs(a[i] - b[i]) > 1e-5) {
            printf("Mismatch at index %zu: CPU = %f, GPU = %f\n", i, a[i], b[i]);
            printf("Previous 5 elements:\n");
            for (size_t j = (i > 5 ? i - 5 : 0); j < i; j++) {
                printf("Index %zu: CPU = %f, GPU = %f\n", j, a[j], b[j]);
            }
            printf("Next 5 elements:\n");
            for (size_t j = i + 1; j < (i + 6 < size ? i + 6 : size); j++) {
                printf("Index %zu: CPU = %f, GPU = %f\n", j, a[j], b[j]);
            }
            return 0;
        }
    }
    return 1;
}


#ifdef RUN_INIT_TEST
int main() {
    Network mnist_net_cpu;
    Network mnist_net_gpu;
    GenericLayer h_out_cuda;
    srand(42);

    // cpu
    cudaEvent_t start_cpu, stop_cpu;
    float cpu_time;
    cudaEventCreate(&start_cpu);
    cudaEventCreate(&stop_cpu);
    cudaEventRecord(start_cpu);
    init_layer(&mnist_net_cpu.hidden, INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE); 
    cudaEventRecord(stop_cpu);
    cudaEventSynchronize(stop_cpu);
    cudaEventElapsedTime(&cpu_time, start_cpu, stop_cpu);

    // gpu
    cudaEvent_t start_gpu, stop_gpu;
    float gpu_time;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);
    cudaEventRecord(start_gpu);
    init_layer_cuda(&mnist_net_gpu.hidden, INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE);
    cudaDeviceSynchronize();
    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);
    cudaEventElapsedTime(&gpu_time, start_gpu, stop_gpu);
    
    size_t weights_size = (size_t)INPUT_LAYER_SIZE * HIDDEN_LAYER_SIZE;
    size_t biases_size = (size_t)HIDDEN_LAYER_SIZE;
    h_out_cuda.weights = (float *)malloc(weights_size * sizeof(float));
    h_out_cuda.biases = (float *)malloc(biases_size * sizeof(float));
    
    cudaMemcpy(h_out_cuda.weights, mnist_net_gpu.hidden.weights, (INPUT_LAYER_SIZE * HIDDEN_LAYER_SIZE) * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_out_cuda.biases, mnist_net_gpu.hidden.biases, HIDDEN_LAYER_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("TEST WEUGHT SIZE: %zu \n", weights_size);

    int result_weights = compare_arrays(mnist_net_cpu.hidden.weights, h_out_cuda.weights, weights_size);
    if (result_weights) {
        printf("\nINIT Weights Test Passed: CPU and CUDA results match.\n");
    } else {
        printf("\nINIT Weights Test Failed: CPU and CUDA results do not match.\n");
    }
    print_ele(mnist_net_cpu.hidden.weights, h_out_cuda.weights, weights_size);

    int result_biases = compare_arrays(mnist_net_cpu.hidden.biases, h_out_cuda.biases, biases_size);
    if (result_biases) {
        printf("\nINIT Biases Test Passed: CPU and CUDA results match.\n");
    } else {
        printf("\nINIT Biases Test Failed: CPU and CUDA results do not match.\n");
    }
    print_ele(mnist_net_cpu.hidden.biases, h_out_cuda.biases, biases_size);

    printf(" CPU time: %f ms\n", cpu_time);
    printf(" GPU time: %f ms\n", gpu_time);
    printf(" Speedup: %fx\n", cpu_time / gpu_time);

    free(mnist_net_cpu.hidden.weights);
    free(mnist_net_cpu.hidden.biases);
    cudaFree(mnist_net_gpu.hidden.weights);
    cudaFree(mnist_net_gpu.hidden.biases);
    cudaEventDestroy(start_cpu);
    cudaEventDestroy(stop_cpu);
    cudaEventDestroy(start_gpu);
    cudaEventDestroy(stop_gpu);
}
#endif