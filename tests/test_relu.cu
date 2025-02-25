#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "../kernels.cuh"
#include "../consts.cuh"
#include <cudnn.h>

void relu_cudnn(float* data, int size, cudnnHandle_t cudnnHandle) {
    cudnnActivationDescriptor_t activationDesc;
    cudnnCreateActivationDescriptor(&activationDesc);
    cudnnSetActivationDescriptor(activationDesc, 
                                CUDNN_ACTIVATION_RELU,
                                CUDNN_PROPAGATE_NAN, 
                                0.0);

    cudnnTensorDescriptor_t tensorDesc;
    cudnnCreateTensorDescriptor(&tensorDesc);
    cudnnSetTensor4dDescriptor(tensorDesc,
                            CUDNN_TENSOR_NCHW,
                            CUDNN_DATA_FLOAT,
                            1, 1, 1, size);

    float alpha = 1.0f;
    float beta = 0.0f;
    
    cudnnActivationForward(cudnnHandle,
                        activationDesc,
                        &alpha,
                        tensorDesc,
                        data,
                        &beta,
                        tensorDesc,
                        data);

    cudnnDestroyActivationDescriptor(activationDesc);
    cudnnDestroyTensorDescriptor(tensorDesc);
}

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
        data[i] = (float)(rand() % 20 - 10) / 2.0f;
    }
}

int compare_arrays(float *a, float *b, int size) {
    for (int i = 0; i < size; i++) {
        if (fabs(a[i] - b[i]) > 1e-5) {
            return 0;
        }
    }
    return 1;
}

void run_test(int size) {
    cudnnHandle_t cudnnHandle;
    cudnnCreate(&cudnnHandle);

    // Allocate host memory
    float *h_input = (float*)malloc(size * sizeof(float));
    float *h_output_cpu = (float*)malloc(size * sizeof(float));
    float *h_output_gpu = (float*)malloc(size * sizeof(float));
    float *h_grad_cpu = (float*)malloc(size * sizeof(float));
    float *h_grad_gpu = (float*)malloc(size * sizeof(float));
    float *h_grad_deriv_cpu = (float*)malloc(size * sizeof(float));
    float *h_grad_deriv_gpu = (float*)malloc(size * sizeof(float));

    // Initialize data
    initialize_data(h_input, size);
    initialize_data(h_grad_cpu, size);
    memcpy(h_grad_deriv_cpu, h_grad_cpu, size * sizeof(float));

    // Allocate device memory
    float *d_input, *d_output, *d_grad;
    cudaMalloc((void**)&d_input, size * sizeof(float));
    cudaMalloc((void**)&d_output, size * sizeof(float));
    cudaMalloc((void**)&d_grad, size * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_input, h_input, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_grad, h_grad_cpu, size * sizeof(float), cudaMemcpyHostToDevice);

    // CPU Forward Pass
    memcpy(h_output_cpu, h_input, size * sizeof(float));
    cudaEvent_t start, stop;
    float cpu_time, gpu_time;
    
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    relu_cpu(h_output_cpu, size);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cpu_time, start, stop);

    // GPU Forward Pass
    cudaMemcpy(d_output, h_input, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaEventRecord(start);
    relu_cuda(d_output, size);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);
    cudaMemcpy(h_output_gpu, d_output, size * sizeof(float), cudaMemcpyDeviceToHost);

    // Forward Pass Validation
    int forward_valid = compare_arrays(h_output_cpu, h_output_gpu, size);

    // CPU Derivative Pass
    float cpu_deriv_time;
    cudaEventRecord(start);
    relu_derivative_cpu(h_grad_deriv_cpu, h_output_cpu, size);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cpu_deriv_time, start, stop);

    // GPU Derivative Pass
    float gpu_deriv_time;
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    
    cudaEventRecord(start);
    optimized_relu_derivative_kernel<<<num_blocks, block_size>>>(d_grad, d_output, size);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_deriv_time, start, stop);
    cudaMemcpy(h_grad_deriv_gpu, d_grad, size * sizeof(float), cudaMemcpyDeviceToHost);

    // Derivative Validation
    int deriv_valid = compare_arrays(h_grad_deriv_cpu, h_grad_deriv_gpu, size);

    // Structured Output
    printf("RELU_RESULTS:\n");
    printf("InputSize: %d\n", size);
    printf("Forward_CPU_Time: %.4f\n", cpu_time);
    printf("Forward_GPU_Time: %.4f\n", gpu_time);
    printf("Forward_Speedup: %.2f\n", cpu_time / gpu_time);
    printf("Forward_Valid: %d\n", forward_valid);
    printf("Derivative_CPU_Time: %.4f\n", cpu_deriv_time);
    printf("Derivative_GPU_Time: %.4f\n", gpu_deriv_time);
    printf("Derivative_Speedup: %.2f\n", cpu_deriv_time / gpu_deriv_time);
    printf("Derivative_Valid: %d\n", deriv_valid);
    printf("END_RESULTS\n");

    // Cleanup
    free(h_input);
    free(h_output_cpu);
    free(h_output_gpu);
    free(h_grad_cpu);
    free(h_grad_gpu);
    free(h_grad_deriv_cpu);
    free(h_grad_deriv_gpu);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_grad);
    cudnnDestroy(cudnnHandle);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <input_size>\n", argv[0]);
        return 1;
    }

    int size = atoi(argv[1]);
    if (size <= 0) {
        fprintf(stderr, "Invalid input size: %d\n", size);
        return 1;
    }

    run_test(size);
    return 0;
}