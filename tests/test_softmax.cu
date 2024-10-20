#include "../kernels.cuh"
#include "../mnist.h"

#define OUT_LAYER_TEST_SIZE (OUTPUT_LAYER_SIZE * 1024 * 1024)

void initialize_data(float *data, int size) {
    for (int i = 0; i < size; i++) {
        data[i] = (float)(rand() % 20 - 10) / 2.0f; // Random values between -5 and 5
    }
}

// void softmax(float *inp, int size) {
//     float max = inp[0], sum = 0;

//     for (int i = 1; i < size; i++) {
//         if (inp[i] > max) {max = inp[i];}
//     }
//     for (int i = 0; i < size; i++) {
//         inp[i] = expf(inp[i] - max);
//         sum += inp[i];
//     }
//     for (int i = 0; i < size; i++) {
//         inp[i] = inp[i] / sum;
//     }
// }

int compare_arrays(float *a, float *b, int size) {
    for (int i = 0; i < size; i++) {
        if (abs(a[i] - b[i]) > 1e-5) {
            return 0;
        }
    }
    return 1;
}

#ifdef RUN_SOFTMAX_TEST
int main() {
    printf("\nSoftmax Test with data size: %d \n", OUT_LAYER_TEST_SIZE);
    // int block_size=256;
    // int num_blocks = (OUT_LAYER_TEST_SIZE + block_size - 1) / block_size;
    float *h_out_cpu = (float *)malloc(OUT_LAYER_TEST_SIZE * sizeof(float));
    float *h_out_cuda = (float *)malloc(OUT_LAYER_TEST_SIZE * sizeof(float));
    float *d_out_softmax;

    srand(42);
    initialize_data(h_out_cpu, OUT_LAYER_TEST_SIZE);

    cudaMalloc((void **)&d_out_softmax, OUT_LAYER_TEST_SIZE * sizeof(float));
    cudaMemcpy(d_out_softmax, h_out_cpu, OUT_LAYER_TEST_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start_cpu, stop_cpu;
    float cpu_time;
    cudaEventCreate(&start_cpu);
    cudaEventCreate(&stop_cpu);
    cudaEventRecord(start_cpu);
    softmax(h_out_cpu, OUT_LAYER_TEST_SIZE);
    cudaEventRecord(stop_cpu);
    cudaEventSynchronize(stop_cpu);
    cudaEventElapsedTime(&cpu_time, start_cpu, stop_cpu);

    cudaEvent_t start_gpu, stop_gpu;
    float gpu_time;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);
    cudaEventRecord(start_gpu);
    softmax_cuda(d_out_softmax, OUT_LAYER_TEST_SIZE);
    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);
    cudaEventElapsedTime(&gpu_time, start_gpu, stop_gpu);

    cudaMemcpy(h_out_cuda, d_out_softmax, OUT_LAYER_TEST_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    int result = compare_arrays(h_out_cpu, h_out_cuda, OUT_LAYER_TEST_SIZE);
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
}
#endif