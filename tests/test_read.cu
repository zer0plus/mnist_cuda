#include "../kernels.cuh"
#include "../mnist.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Normalize CPU images to match the GPU implementation (uint8 -> float32 / 255.0f)
void normalize_cpu_images(unsigned char *raw_imgs, float *normalized_imgs, int num_imgs, int img_size) {
    for (int i = 0; i < num_imgs * img_size; i++) {
        normalized_imgs[i] = raw_imgs[i] / 255.0f;
    }
}

int compare_float_arrays(float *a, float *b, int size) {
    float max_diff = 0.0f;
    int max_diff_idx = -1;
    
    // Find the maximum difference
    for (int i = 0; i < size; i++) {
        float diff = fabs(a[i] - b[i]);
        if (diff > max_diff) {
            max_diff = diff;
            max_diff_idx = i;
        }
    }
    
    // Check against precision level
    if (max_diff <= 1e-6) {
        printf("Arrays match with precision of 1e-6 or better\n");
        return 1;
    } else {
        printf("Maximum difference: %.9e at index %d (CPU: %f, GPU: %f)\n", 
               max_diff, max_diff_idx, a[max_diff_idx], b[max_diff_idx]);
        return 0;
    }
}

int compare_label_arrays(unsigned char *a, unsigned char *b, int size) {
    for (int i = 0; i < size; i++) {
        if (a[i] != b[i]) {
            printf("Label mismatch at index %d: CPU = %d, GPU = %d\n", i, a[i], b[i]);
            return 0;
        }
    }
    return 1;
}

#ifdef RUN_READ_TEST
int main(int argc, char** argv) {
    printf("READ_RESULTS:\n");
    
    // Path variables for dataset files - adjusted for running from tests directory
    const char* img_path = "../data/train-images.idx3-ubyte";
    const char* label_path = "../data/train-labels.idx1-ubyte";
    
    // Print paths for verification
    printf("Image file: %s\n", img_path);
    printf("Label file: %s\n", label_path);
    
    // Check if files exist before proceeding
    FILE* test_img = fopen(img_path, "rb");
    if (!test_img) {
        printf("Error: Cannot open image file at %s\n", img_path);
        // Try alternative path
        img_path = "../data/MNIST/raw/train-images-idx3-ubyte";
        label_path = "../data/MNIST/raw/train-labels-idx1-ubyte";
        
        test_img = fopen(img_path, "rb");
        if (!test_img) {
            printf("Error: Cannot open image file at alternative path %s\n", img_path);
            printf("END_RESULTS\n");
            return 1;
        }
        printf("Using alternative paths: %s and %s\n", img_path, label_path);
    }
    fclose(test_img);
    
    // ==================== IMAGES BENCHMARK ====================
    
    // Setup for CPU image reading
    InputData_CPU cpu_img_data = {0};
    int cpu_num_imgs = 0;
    
    // Create CUDA events for timing
    cudaEvent_t start_cpu_img, stop_cpu_img;
    cudaEventCreate(&start_cpu_img);
    cudaEventCreate(&stop_cpu_img);
    
    // Benchmark CPU image reading
    float cpu_img_time;
    cudaEventRecord(start_cpu_img);
    read_mnist_imgs(img_path, &cpu_img_data.imgs, &cpu_num_imgs);
    cudaEventRecord(stop_cpu_img);
    cudaEventSynchronize(stop_cpu_img);
    cudaEventElapsedTime(&cpu_img_time, start_cpu_img, stop_cpu_img);
    
    printf("CPU_Images_Read_Time: %f\n", cpu_img_time);
    printf("Images_Count: %d\n", cpu_num_imgs);
    printf("Image_Size: %d\n", INPUT_LAYER_SIZE);
    
    // Setup for GPU image reading
    InputData_GPU gpu_img_data = {0};
    int gpu_num_imgs = 0;
    
    // Benchmark GPU image reading (includes normalization)
    cudaEvent_t start_gpu_img, stop_gpu_img;
    cudaEventCreate(&start_gpu_img);
    cudaEventCreate(&stop_gpu_img);
    
    float gpu_img_time;
    cudaEventRecord(start_gpu_img);
    read_mnist_imgs_cuda(img_path, &gpu_img_data.imgs, &gpu_num_imgs);
    cudaEventRecord(stop_gpu_img);
    cudaEventSynchronize(stop_gpu_img);
    cudaEventElapsedTime(&gpu_img_time, start_gpu_img, stop_gpu_img);
    
    printf("GPU_Images_Read_Time: %f\n", gpu_img_time);
    
    // Benchmark CPU normalization separately
    float* cpu_normalized_imgs = (float*)malloc(cpu_num_imgs * INPUT_LAYER_SIZE * sizeof(float));
    
    cudaEvent_t start_cpu_norm, stop_cpu_norm;
    cudaEventCreate(&start_cpu_norm);
    cudaEventCreate(&stop_cpu_norm);
    
    float cpu_norm_time;
    cudaEventRecord(start_cpu_norm);
    normalize_cpu_images(cpu_img_data.imgs, cpu_normalized_imgs, cpu_num_imgs, INPUT_LAYER_SIZE);
    cudaEventRecord(stop_cpu_norm);
    cudaEventSynchronize(stop_cpu_norm);
    cudaEventElapsedTime(&cpu_norm_time, start_cpu_norm, stop_cpu_norm);
    
    printf("CPU_Images_Normalize_Time: %f\n", cpu_norm_time);
    
    // Calculate total CPU time including normalization
    float cpu_total_img_time = cpu_img_time + cpu_norm_time;
    printf("CPU_Images_Total_Time: %f\n", cpu_total_img_time);
    
    // Calculate speedup
    float img_speedup = cpu_total_img_time / gpu_img_time;
    printf("Images_Speedup: %f\n", img_speedup);
    
    // Validate results
    float* gpu_imgs_host = (float*)malloc(gpu_num_imgs * INPUT_LAYER_SIZE * sizeof(float));
    cudaMemcpy(gpu_imgs_host, gpu_img_data.imgs, gpu_num_imgs * INPUT_LAYER_SIZE * sizeof(float),
               cudaMemcpyDeviceToHost);
    
    int img_result = compare_float_arrays(cpu_normalized_imgs, gpu_imgs_host, 
                                          cpu_num_imgs * INPUT_LAYER_SIZE);
    printf("Images_Valid: %d\n", img_result);
    
    // ==================== LABELS BENCHMARK ====================
    
    // Setup for CPU label reading
    InputData_CPU cpu_label_data = {0};
    int cpu_num_labels = 0;
    
    // Benchmark CPU label reading
    cudaEvent_t start_cpu_label, stop_cpu_label;
    cudaEventCreate(&start_cpu_label);
    cudaEventCreate(&stop_cpu_label);
    
    float cpu_label_time;
    cudaEventRecord(start_cpu_label);
    read_mnist_labels(label_path, &cpu_label_data.labels, &cpu_num_labels);
    cudaEventRecord(stop_cpu_label);
    cudaEventSynchronize(stop_cpu_label);
    cudaEventElapsedTime(&cpu_label_time, start_cpu_label, stop_cpu_label);
    
    printf("CPU_Labels_Read_Time: %f\n", cpu_label_time);
    printf("Labels_Count: %d\n", cpu_num_labels);
    
    // Setup for GPU label reading
    InputData_GPU gpu_label_data = {0};
    int gpu_num_labels = 0;
    
    // Benchmark GPU label reading
    cudaEvent_t start_gpu_label, stop_gpu_label;
    cudaEventCreate(&start_gpu_label);
    cudaEventCreate(&stop_gpu_label);
    
    float gpu_label_time;
    cudaEventRecord(start_gpu_label);
    read_mnist_labels_cuda(label_path, &gpu_label_data.labels, &gpu_num_labels);
    cudaEventRecord(stop_gpu_label);
    cudaEventSynchronize(stop_gpu_label);
    cudaEventElapsedTime(&gpu_label_time, start_gpu_label, stop_gpu_label);
    
    printf("GPU_Labels_Read_Time: %f\n", gpu_label_time);
    
    // Calculate speedup
    float label_speedup = cpu_label_time / gpu_label_time;
    printf("Labels_Speedup: %f\n", label_speedup);
    
    // Validate results
    unsigned char* gpu_labels_host = (unsigned char*)malloc(gpu_num_labels * sizeof(unsigned char));
    cudaMemcpy(gpu_labels_host, gpu_label_data.labels, gpu_num_labels * sizeof(unsigned char),
               cudaMemcpyDeviceToHost);
    
    int label_result = compare_label_arrays(cpu_label_data.labels, gpu_labels_host, cpu_num_labels);
    printf("Labels_Valid: %d\n", label_result);
    
    // ==================== OVERALL RESULTS ====================
    
    // Calculate overall time and speedup
    float cpu_total_time = cpu_total_img_time + cpu_label_time;
    float gpu_total_time = gpu_img_time + gpu_label_time;
    float total_speedup = cpu_total_time / gpu_total_time;
    
    printf("CPU_Total_Time: %f\n", cpu_total_time);
    printf("GPU_Total_Time: %f\n", gpu_total_time);
    printf("Total_Speedup: %f\n", total_speedup);
    printf("Overall_Valid: %d\n", img_result && label_result);
    
    // Print summary
    if (img_result && label_result) {
        printf("\nRead Test Passed: CPU and CUDA results match.\n");
        printf("Images: CPU: %.2f ms, GPU: %.2f ms, Speedup: %.2fx\n", 
               cpu_total_img_time, gpu_img_time, img_speedup);
        printf("Labels: CPU: %.2f ms, GPU: %.2f ms, Speedup: %.2fx\n", 
               cpu_label_time, gpu_label_time, label_speedup);
        printf("Total:  CPU: %.2f ms, GPU: %.2f ms, Speedup: %.2fx\n", 
               cpu_total_time, gpu_total_time, total_speedup);
    } else {
        printf("\nRead Test Failed: CPU and CUDA results do not match.\n");
    }
    
    printf("END_RESULTS\n");
    
    // Clean up
    free(cpu_normalized_imgs);
    free(gpu_imgs_host);
    free(gpu_labels_host);
    free(cpu_img_data.imgs);
    free(cpu_label_data.labels);
    cudaFree(gpu_img_data.imgs);
    cudaFree(gpu_label_data.labels);
    
    cudaEventDestroy(start_cpu_img);
    cudaEventDestroy(stop_cpu_img);
    cudaEventDestroy(start_gpu_img);
    cudaEventDestroy(stop_gpu_img);
    cudaEventDestroy(start_cpu_norm);
    cudaEventDestroy(stop_cpu_norm);
    cudaEventDestroy(start_cpu_label);
    cudaEventDestroy(stop_cpu_label);
    cudaEventDestroy(start_gpu_label);
    cudaEventDestroy(stop_gpu_label);
    
    return 0;
}
#endif