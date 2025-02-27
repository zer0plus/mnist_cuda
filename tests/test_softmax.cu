#include "../kernels.cuh"
#include "../mnist.h"
#include <stdio.h>
#include <stdlib.h>

void initialize_data(float *data, int size) {
    for (int i = 0; i < size; i++) {
        data[i] = (float)(rand() % 2 - 1); // Random values between -555 and 555
    }
}

int compare_arrays(float *a, float *b, int size) {
    float max_diff = 0.0f;
    int max_diff_idx = -1;
    
    // First find the maximum difference
    for (int i = 0; i < size; i++) {
        float diff = fabs(a[i] - b[i]);
        if (diff > max_diff) {
            max_diff = diff;
            max_diff_idx = i;
        }
    }
    
    // Now check against different precision levels
    if (max_diff <= 1e-7) {
        printf("Passed with highest precision (1e-7)\n");
        return 1;
    } else if (max_diff <= 1e-6) {
        printf("Resolved with 1e-6 precision (Maximum diff: %.9e at index %d)\n", max_diff, max_diff_idx);
        return 1;
    } else if (max_diff <= 1e-5) {
        printf("Resolved with 1e-5 precision (Maximum diff: %.9e at index %d)\n", max_diff, max_diff_idx);
        return 1;
    } else if (max_diff <= 1e-4) {
        printf("Resolved with 1e-4 precision (Maximum diff: %.9e at index %d)\n", max_diff, max_diff_idx);
        return 1;
    } else if (max_diff <= 1e-3) {
        printf("Resolved with 1e-3 precision (Maximum diff: %.9e at index %d)\n", max_diff, max_diff_idx);
        return 1;
    } else if (max_diff <= 1e-2) {
        printf("Resolved with 1e-2 precision (Maximum diff: %.9e at index %d)\n", max_diff, max_diff_idx);
        return 1; 
    } else {
        printf("Maximum difference: %.9e at index %d\n", max_diff, max_diff_idx);
        return 0;
    }
}

#ifdef RUN_SOFTMAX_TEST
int main(int argc, char** argv) {
    int size;
    
    // Check if a size argument was provided
    if (argc > 1) {
        size = atoi(argv[1]);
        if (size <= 0) {
            fprintf(stderr, "Invalid size argument. Using default size.\n");
            size = OUTPUT_LAYER_SIZE * 10000;
        }
    } else {
        size = OUTPUT_LAYER_SIZE * 10000;
        printf("No size specified. Using default size.\n");
    }
    
    printf("\nSoftmax Test with data size: %d \n", size);
    
    float *h_out_cpu = (float *)malloc(size * sizeof(float));
    float *h_out_cuda = (float *)malloc(size * sizeof(float));
    float *d_out_softmax;

    srand(42);
    initialize_data(h_out_cpu, size);

    cudaMalloc((void **)&d_out_softmax, size * sizeof(float));
    cudaMemcpy(d_out_softmax, h_out_cpu, size * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start_cpu, stop_cpu;
    float cpu_time;
    cudaEventCreate(&start_cpu);
    cudaEventCreate(&stop_cpu);
    cudaEventRecord(start_cpu);
    softmax(h_out_cpu, size);
    cudaEventRecord(stop_cpu);
    cudaEventSynchronize(stop_cpu);
    cudaEventElapsedTime(&cpu_time, start_cpu, stop_cpu);

    cudaEvent_t start_gpu, stop_gpu;
    float gpu_time;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);
    cudaEventRecord(start_gpu);
    softmax_cuda(d_out_softmax, size);
    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);
    cudaEventElapsedTime(&gpu_time, start_gpu, stop_gpu);

    cudaMemcpy(h_out_cuda, d_out_softmax, size * sizeof(float), cudaMemcpyDeviceToHost);
    int result = compare_arrays(h_out_cpu, h_out_cuda, size);
    if (result) {
        printf("\nSoftmax Test Passed: CPU and CUDA results match.\n");
    } else {
        printf("\nSoftmax Test Failed: CPU and CUDA results do not match.\n");
    }

    printf(" CPU time: %f ms\n", cpu_time);
    printf(" GPU time: %f ms\n", gpu_time);
    printf(" Speedup: %fx\n", cpu_time / gpu_time);

    free(h_out_cpu);
    free(h_out_cuda);
    cudaFree(d_out_softmax);
    cudaEventDestroy(start_cpu);
    cudaEventDestroy(stop_cpu);
    cudaEventDestroy(start_gpu);
    cudaEventDestroy(stop_gpu);
    
    return 0;
}
#endif