#include "../kernels.cuh"
#include "../mnist.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <curand.h>
#include <curand_kernel.h>

#define HIDDEN_LAYER_TEST_SIZE (HIDDEN_LAYER_SIZE * 1024 * 1024)
#define OUT_LAYER_TEST_SIZE (OUTPUT_LAYER_SIZE * 1024 * 1024)

void initialize_data(float *data, int size) {
    for (int i = 0; i < size; i++) {
        data[i] = (float)(rand() % 20 - 10) / 2.0f; // Random values between -5 and 5
    }
}

int compare_arrays(float *a, float *b, int size) {
    for (int i = 0; i < size; i++) {
        if (abs(a[i] - b[i]) > 1e-5) {
            return 0;
        }
    }
    return 1;
}

// #ifdef RUN_LINEAR_TEST
int main() {
    printf("\nLinear Test with data size: %d \n", HIDDEN_LAYER_TEST_SIZE);
    float *h_out_cpu = (float *)malloc(OUT_LAYER_TEST_SIZE * sizeof(float));
    float *h_hidden_cpu = (float *)malloc(HIDDEN_LAYER_TEST_SIZE * sizeof(float));
    float *h_out_cuda = (float *)malloc(OUT_LAYER_TEST_SIZE * sizeof(float));
    float *h_hidden_cuda = (float *)malloc(HIDDEN_LAYER_TEST_SIZE * sizeof(float));
    InputData_CPU cpu_data = {0};

    read_mnist_imgs(TESTING_IMG_PATH, &cpu_data.imgs, &cpu_data.num_imgs);

    cudaEvent_t start_cpu, stop_cpu;
    float cpu_time;



}
#endif