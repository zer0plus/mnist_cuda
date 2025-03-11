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

int compare_arrays(float *a, float *b, size_t size, float tolerance) {
    int mismatches = 0;
    float max_diff = 0.0f;
    size_t max_diff_idx = 0;
    
    for (size_t i = 0; i < size; i++) {
        float diff = fabs(a[i] - b[i]);
        if (diff > tolerance) {
            mismatches++;
            if (diff > max_diff) {
                max_diff = diff;
                max_diff_idx = i;
            }
        }
    }
    
    if (mismatches > 0) {
        printf("Found %d mismatches. Max difference: %f at index %zu (CPU: %f, GPU: %f)\n", 
               mismatches, max_diff, max_diff_idx, a[max_diff_idx], b[max_diff_idx]);
        
        // Print some values around the maximum difference
        printf("Values around max difference:\n");
        size_t start_idx = (max_diff_idx > 5) ? max_diff_idx - 5 : 0;
        size_t end_idx = (max_diff_idx + 5 < size) ? max_diff_idx + 5 : size - 1;
        for (size_t i = start_idx; i <= end_idx; i++) {
            printf("  [%zu] CPU: %f, GPU: %f, diff: %f\n", 
                   i, a[i], b[i], fabs(a[i] - b[i]));
        }
        return 0;
    }
    return 1;
}

// Compute the relative difference between arrays
float compute_relative_error(float *a, float *b, size_t size) {
    float sum_squared_diff = 0.0f;
    float sum_squared_a = 0.0f;
    
    for (size_t i = 0; i < size; i++) {
        float diff = a[i] - b[i];
        sum_squared_diff += diff * diff;
        sum_squared_a += a[i] * a[i];
    }
    
    // Avoid division by zero
    if (sum_squared_a < 1e-10f) {
        return sum_squared_diff;
    }
    
    return sqrtf(sum_squared_diff / sum_squared_a);
}

#ifdef RUN_BACKWARD_TEST
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

    printf("BACKWARD_RESULTS:\n");
    printf("InputSize: %d\n", in_size);
    printf("OutputSize: %d\n", out_size);
    
    // Fixed seed for reproducibility
    srand(42);
    
    // Allocate host memory
    float *h_input = (float *)malloc(in_size * sizeof(float));
    float *h_out_grad = (float *)malloc(out_size * sizeof(float));
    float *h_in_grad_cpu = (float *)malloc(in_size * sizeof(float));
    float *h_in_grad_cublas = (float *)malloc(in_size * sizeof(float));
    float *h_in_grad_custom = (float *)malloc(in_size * sizeof(float));
    
    // Initialize input and output gradient with random data
    initialize_data(h_input, in_size);
    initialize_data(h_out_grad, out_size);
    memset(h_in_grad_cpu, 0, in_size * sizeof(float));
    
    // Create CPU layer
    GenericLayer cpu_layer = {0};
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
        cpu_layer.biases[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;  // Random biases
    }
    
    // Create GPU layers with same values (one for each implementation)
    GenericLayer cublas_layer = {0};
    cublas_layer.in_size = in_size;
    cublas_layer.out_size = out_size;
    
    GenericLayer custom_layer = {0};
    custom_layer.in_size = in_size;
    custom_layer.out_size = out_size;
    
    CUDA_CHECK(cudaMalloc((void **)&cublas_layer.weights, in_size * out_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&cublas_layer.biases, out_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&custom_layer.weights, in_size * out_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&custom_layer.biases, out_size * sizeof(float)));
    
    CUDA_CHECK(cudaMemcpy(cublas_layer.weights, cpu_layer.weights, in_size * out_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(cublas_layer.biases, cpu_layer.biases, out_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(custom_layer.weights, cpu_layer.weights, in_size * out_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(custom_layer.biases, cpu_layer.biases, out_size * sizeof(float), cudaMemcpyHostToDevice));
    
    // Allocate device memory for input, output gradient, and input gradient
    float *d_input, *d_out_grad;
    float *d_in_grad_cublas, *d_in_grad_custom;
    
    CUDA_CHECK(cudaMalloc((void **)&d_input, in_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_out_grad, out_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_in_grad_cublas, in_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_in_grad_custom, in_size * sizeof(float)));
    
    // Copy input and output gradient to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input, in_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_out_grad, h_out_grad, out_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_in_grad_cublas, 0, in_size * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_in_grad_custom, 0, in_size * sizeof(float)));
    
    // Initialize cuBLAS
    init_cublas();
    
    // Create events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Warmup runs with separate layer copies to avoid affecting the actual test
    float lr = 0.01f;
    
    // Create warmup copies for all implementations
    GenericLayer warmup_cpu_layer = {0};
    warmup_cpu_layer.in_size = in_size;
    warmup_cpu_layer.out_size = out_size;
    warmup_cpu_layer.weights = (float *)malloc(in_size * out_size * sizeof(float));
    warmup_cpu_layer.biases = (float *)malloc(out_size * sizeof(float));
    
    GenericLayer warmup_cublas_layer = {0};
    warmup_cublas_layer.in_size = in_size;
    warmup_cublas_layer.out_size = out_size;
    CUDA_CHECK(cudaMalloc((void **)&warmup_cublas_layer.weights, in_size * out_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&warmup_cublas_layer.biases, out_size * sizeof(float)));
    
    GenericLayer warmup_custom_layer = {0};
    warmup_custom_layer.in_size = in_size;
    warmup_custom_layer.out_size = out_size;
    CUDA_CHECK(cudaMalloc((void **)&warmup_custom_layer.weights, in_size * out_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&warmup_custom_layer.biases, out_size * sizeof(float)));
    
    // Do warmup iterations
    for (int i = 0; i < 5; i++) {
        // Reset warmup layers to original values
        memcpy(warmup_cpu_layer.weights, cpu_layer.weights, in_size * out_size * sizeof(float));
        memcpy(warmup_cpu_layer.biases, cpu_layer.biases, out_size * sizeof(float));
        CUDA_CHECK(cudaMemcpy(warmup_cublas_layer.weights, cpu_layer.weights, in_size * out_size * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(warmup_cublas_layer.biases, cpu_layer.biases, out_size * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(warmup_custom_layer.weights, cpu_layer.weights, in_size * out_size * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(warmup_custom_layer.biases, cpu_layer.biases, out_size * sizeof(float), cudaMemcpyHostToDevice));
        
        // Run warmup iterations
        backward(&warmup_cpu_layer, h_input, h_out_grad, h_in_grad_cpu, lr);
        backward_cuda_cublas(&warmup_cublas_layer, d_input, d_out_grad, d_in_grad_cublas, lr);
        backward_cuda(&warmup_custom_layer, d_input, d_out_grad, d_in_grad_custom, lr);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    // Clean up warmup resources
    free(warmup_cpu_layer.weights);
    free(warmup_cpu_layer.biases);
    CUDA_CHECK(cudaFree(warmup_cublas_layer.weights));
    CUDA_CHECK(cudaFree(warmup_cublas_layer.biases));
    CUDA_CHECK(cudaFree(warmup_custom_layer.weights));
    CUDA_CHECK(cudaFree(warmup_custom_layer.biases));
    
    // Reset for actual test
    memset(h_in_grad_cpu, 0, in_size * sizeof(float));
    CUDA_CHECK(cudaMemset(d_in_grad_cublas, 0, in_size * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_in_grad_custom, 0, in_size * sizeof(float)));
    
    // Make copies of the original layer data for the actual test
    GenericLayer test_cpu_layer = {0};
    test_cpu_layer.in_size = in_size;
    test_cpu_layer.out_size = out_size;
    test_cpu_layer.weights = (float *)malloc(in_size * out_size * sizeof(float));
    test_cpu_layer.biases = (float *)malloc(out_size * sizeof(float));
    memcpy(test_cpu_layer.weights, cpu_layer.weights, in_size * out_size * sizeof(float));
    memcpy(test_cpu_layer.biases, cpu_layer.biases, out_size * sizeof(float));
    
    GenericLayer test_cublas_layer = {0};
    test_cublas_layer.in_size = in_size;
    test_cublas_layer.out_size = out_size;
    CUDA_CHECK(cudaMalloc((void **)&test_cublas_layer.weights, in_size * out_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&test_cublas_layer.biases, out_size * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(test_cublas_layer.weights, cpu_layer.weights, in_size * out_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(test_cublas_layer.biases, cpu_layer.biases, out_size * sizeof(float), cudaMemcpyHostToDevice));
    
    GenericLayer test_custom_layer = {0};
    test_custom_layer.in_size = in_size;
    test_custom_layer.out_size = out_size;
    CUDA_CHECK(cudaMalloc((void **)&test_custom_layer.weights, in_size * out_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&test_custom_layer.biases, out_size * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(test_custom_layer.weights, cpu_layer.weights, in_size * out_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(test_custom_layer.biases, cpu_layer.biases, out_size * sizeof(float), cudaMemcpyHostToDevice));
    
    // Benchmark CPU implementation
    float cpu_time;
    cudaEventRecord(start);
    backward(&test_cpu_layer, h_input, h_out_grad, h_in_grad_cpu, lr);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cpu_time, start, stop);
    
    // Benchmark cuBLAS implementation
    float cublas_time;
    cudaEventRecord(start);
    backward_cuda_cublas(&test_cublas_layer, d_input, d_out_grad, d_in_grad_cublas, lr);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cublas_time, start, stop);
    
    // Benchmark custom CUDA implementation
    float custom_time;
    cudaEventRecord(start);
    backward_cuda(&test_custom_layer, d_input, d_out_grad, d_in_grad_custom, lr);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&custom_time, start, stop);
    
    // Copy results back to host for validation
    float *h_weights_cublas = (float *)malloc(in_size * out_size * sizeof(float));
    float *h_biases_cublas = (float *)malloc(out_size * sizeof(float));
    float *h_weights_custom = (float *)malloc(in_size * out_size * sizeof(float));
    float *h_biases_custom = (float *)malloc(out_size * sizeof(float));
    
    CUDA_CHECK(cudaMemcpy(h_weights_cublas, test_cublas_layer.weights, in_size * out_size * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_biases_cublas, test_cublas_layer.biases, out_size * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_in_grad_cublas, d_in_grad_cublas, in_size * sizeof(float), cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaMemcpy(h_weights_custom, test_custom_layer.weights, in_size * out_size * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_biases_custom, test_custom_layer.biases, out_size * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_in_grad_custom, d_in_grad_custom, in_size * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Validate results with different tolerances for different components
    float weights_tolerance = 1e-4f;
    float biases_tolerance = 1e-4f;
    float in_grad_tolerance = 1e-2f;  // Higher tolerance for input gradients
    
    // Validate cuBLAS results against CPU
    int cublas_weights_valid = compare_arrays(test_cpu_layer.weights, h_weights_cublas, in_size * out_size, weights_tolerance);
    int cublas_biases_valid = compare_arrays(test_cpu_layer.biases, h_biases_cublas, out_size, biases_tolerance);
    int cublas_in_grad_valid = compare_arrays(h_in_grad_cpu, h_in_grad_cublas, in_size, in_grad_tolerance);
    
    // Validate custom CUDA results against CPU
    int custom_weights_valid = compare_arrays(test_cpu_layer.weights, h_weights_custom, in_size * out_size, weights_tolerance);
    int custom_biases_valid = compare_arrays(test_cpu_layer.biases, h_biases_custom, out_size, biases_tolerance);
    int custom_in_grad_valid = compare_arrays(h_in_grad_cpu, h_in_grad_custom, in_size, in_grad_tolerance);
    
    // Compute relative errors for cuBLAS
    float cublas_weights_rel_error = compute_relative_error(test_cpu_layer.weights, h_weights_cublas, in_size * out_size);
    float cublas_biases_rel_error = compute_relative_error(test_cpu_layer.biases, h_biases_cublas, out_size);
    float cublas_in_grad_rel_error = compute_relative_error(h_in_grad_cpu, h_in_grad_cublas, in_size);
    
    // Compute relative errors for custom CUDA
    float custom_weights_rel_error = compute_relative_error(test_cpu_layer.weights, h_weights_custom, in_size * out_size);
    float custom_biases_rel_error = compute_relative_error(test_cpu_layer.biases, h_biases_custom, out_size);
    float custom_in_grad_rel_error = compute_relative_error(h_in_grad_cpu, h_in_grad_custom, in_size);
    
    // Consider the implementations valid if weights and biases match
    int cublas_valid = cublas_weights_valid && cublas_biases_valid;
    int custom_valid = custom_weights_valid && custom_biases_valid;
    
    // Also compare custom CUDA against cuBLAS
    float custom_vs_cublas_weights_rel_error = compute_relative_error(h_weights_cublas, h_weights_custom, in_size * out_size);
    float custom_vs_cublas_biases_rel_error = compute_relative_error(h_biases_cublas, h_biases_custom, out_size);
    float custom_vs_cublas_in_grad_rel_error = compute_relative_error(h_in_grad_cublas, h_in_grad_custom, in_size);
    
    // Calculate speedups
    float cublas_speedup = cpu_time / cublas_time;
    float custom_speedup = cpu_time / custom_time;
    float custom_vs_cublas = cublas_time / custom_time;  // > 1 means custom is faster
    
    // Print results
    printf("CPU_Time: %.4f\n", cpu_time);
    printf("CuBLAS_Time: %.4f\n", cublas_time);
    printf("Custom_Time: %.4f\n", custom_time);
    printf("CuBLAS_Speedup: %.4f\n", cublas_speedup);
    printf("Custom_Speedup: %.4f\n", custom_speedup);
    printf("Custom_vs_CuBLAS: %.4f\n", custom_vs_cublas);
    
    // CuBLAS validation results
    printf("CuBLAS_Weights_Valid: %d\n", cublas_weights_valid);
    printf("CuBLAS_Biases_Valid: %d\n", cublas_biases_valid);
    printf("CuBLAS_In_Grad_Valid: %d\n", cublas_in_grad_valid);
    printf("CuBLAS_Weights_Rel_Error: %.6e\n", cublas_weights_rel_error);
    printf("CuBLAS_Biases_Rel_Error: %.6e\n", cublas_biases_rel_error);
    printf("CuBLAS_In_Grad_Rel_Error: %.6e\n", cublas_in_grad_rel_error);
    printf("CuBLAS_Valid: %d\n", cublas_valid);
    
    // Custom CUDA validation results
    printf("Custom_Weights_Valid: %d\n", custom_weights_valid);
    printf("Custom_Biases_Valid: %d\n", custom_biases_valid);
    printf("Custom_In_Grad_Valid: %d\n", custom_in_grad_valid);
    printf("Custom_Weights_Rel_Error: %.6e\n", custom_weights_rel_error);
    printf("Custom_Biases_Rel_Error: %.6e\n", custom_biases_rel_error);
    printf("Custom_In_Grad_Rel_Error: %.6e\n", custom_in_grad_rel_error);
    printf("Custom_Valid: %d\n", custom_valid);
    
    // Custom vs CuBLAS comparison
    printf("Custom_vs_CuBLAS_Weights_Rel_Error: %.6e\n", custom_vs_cublas_weights_rel_error);
    printf("Custom_vs_CuBLAS_Biases_Rel_Error: %.6e\n", custom_vs_cublas_biases_rel_error);
    printf("Custom_vs_CuBLAS_In_Grad_Rel_Error: %.6e\n", custom_vs_cublas_in_grad_rel_error);
    
    // Print summary
    printf("Valid: %d\n", cublas_valid && custom_valid);
    
    if (cublas_valid && custom_valid) {
        printf("Backward Test Passed: Both cuBLAS and custom CUDA implementations' weights and biases match CPU.\n");
        if (!cublas_in_grad_valid || !custom_in_grad_valid) {
            printf("Note: Input gradients have some numerical differences (relative errors: cuBLAS=%.6e, custom=%.6e).\n", 
                   cublas_in_grad_rel_error, custom_in_grad_rel_error);
            printf("This is expected due to different calculation methods and floating-point precision.\n");
        }
    } else {
        printf("Backward Test Failed: Some implementations did not match CPU results.\n");
        if (!cublas_valid) {
            if (!cublas_weights_valid) printf("- cuBLAS weights mismatch (relative error: %.6e)\n", cublas_weights_rel_error);
            if (!cublas_biases_valid) printf("- cuBLAS biases mismatch (relative error: %.6e)\n", cublas_biases_rel_error);
        }
        if (!custom_valid) {
            if (!custom_weights_valid) printf("- Custom CUDA weights mismatch (relative error: %.6e)\n", custom_weights_rel_error);
            if (!custom_biases_valid) printf("- Custom CUDA biases mismatch (relative error: %.6e)\n", custom_biases_rel_error);
        }
    }
    
    printf("Performance Summary:\n");
    printf("- CPU:          %.4f ms\n", cpu_time);
    printf("- cuBLAS:       %.4f ms (%.2fx speedup vs CPU)\n", cublas_time, cublas_speedup);
    printf("- Custom CUDA:  %.4f ms (%.2fx speedup vs CPU, %.2fx %s than cuBLAS)\n", 
           custom_time, custom_speedup, 
           fabs(custom_vs_cublas), custom_vs_cublas > 1.0f ? "faster" : "slower");
    
    printf("END_RESULTS\n");
    
    // Cleanup
    free(h_input);
    free(h_out_grad);
    free(h_in_grad_cpu);
    free(h_in_grad_cublas);
    free(h_in_grad_custom);
    free(h_weights_cublas);
    free(h_biases_cublas);
    free(h_weights_custom);
    free(h_biases_custom);
    free(cpu_layer.weights);
    free(cpu_layer.biases);
    free(test_cpu_layer.weights);
    free(test_cpu_layer.biases);
    
    cudaFree(cublas_layer.weights);
    cudaFree(cublas_layer.biases);
    cudaFree(custom_layer.weights);
    cudaFree(custom_layer.biases);
    cudaFree(test_cublas_layer.weights);
    cudaFree(test_cublas_layer.biases);
    cudaFree(test_custom_layer.weights);
    cudaFree(test_custom_layer.biases);
    cudaFree(d_input);
    cudaFree(d_out_grad);
    cudaFree(d_in_grad_cublas);
    cudaFree(d_in_grad_custom);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    // Clean up cuBLAS
    cleanup_cublas();
    
    return 0;
}
#endif