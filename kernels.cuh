// kernel.cuh
#ifndef KERNELS_H
#define KERNELS_H
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <curand.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include "consts.cuh"

void init_cublas();
void cleanup_cublas();
//Tested
void init_layer_cuda(GenericLayer *layer, int in_size, int out_size);
__global__ void init_curand_states(curandState *states, unsigned long seed, size_t total_weights);
__global__ void init_weights_kernel(float *weights, size_t total_weights, float scale, curandState *states);

__global__ void relu_kernel(float *out, int size);
__global__ void relu_derivative_kernel(float *grad, float *out, int size);
void softmax_cuda(float *d_inp, int size);
// __global__ void exp_subtract_sum_kernel(float *inp, int size, float *max_val, float *sum);
// __global__ void divide_kernel(float *inp, int size, float *sum);
void read_mnist_imgs_cuda(const char *filename, float **d_imgs, int *num_imgs);
void read_mnist_labels_cuda(const char *filename, unsigned char **d_labels, int *num_labels);
__global__ void normalize_imgs_kernel(unsigned char* raw_imgs, float* d_normalized_imgs, int total_pixels);
void linear_cuda(GenericLayer *layer, float *inp, float *out);
void linear_cuda_cublas(GenericLayer *layer, float *inp, float *out);
__global__ void copy_biases_kernel(float *biases, float *out, int out_size);
__global__ void dot_product_kernel(float *weights, float *inp, float *partial_sums, int in_size, int out_size);
__global__ void matrix_multiply_kernel(float *weights, float *inp, float *out, float *partial_sums, int in_size, int out_size);

//testing
void backward_cuda(GenericLayer *layer, float *inp, float *out_grad, float *in_grad, float lr);
void backward_cuda_cublas(GenericLayer *layer, float *inp, float *out_grad, float *in_grad, float lr);
__global__ void update_out_grad_kernel(float *out_grad, float *output, unsigned char *d_labels, int d_label_idx);

// To Test
__global__ void swap_pixels_kernel(float *imgs, int i, int j, int img_size);
__global__ void shuffle_kernel(float *imgs, unsigned char *labels, int *shuffled_indices, int num_imgs, int img_size);
__global__ void shuffle_naive_kernel(float *imgs, unsigned char *labels, int *shuffled_indices, int num_imgs, int img_size);
__global__ void optimized_relu_kernel(float* __restrict__ data, const int size);
__global__ void optimized_relu_derivative_kernel(float *grad, float *input, int size);
__global__ void optimized_softmax_kernel(float* __restrict__ inp, const int size);
void relu_cuda(float *d_data, int size);

void print_device_tensor(const char* tensor_name, float* d_ptr, int shape_size, int num_elements_to_print);

#endif