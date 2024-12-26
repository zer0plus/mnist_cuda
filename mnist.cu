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
#include <cuda.h>
#include <cublas_v2.h>

// The MNIST image/label file structure is as follows:
//     [offset] [type] [value] [description]
//     0000 32 bit int 0x00000803 magic number
//     0004 32 bit int 60000 number of images
//     0008 32 bit int 28 number of rows
//     0012 32 bit int 28 number of columns
//     0016 unsigned byte ?? pixel
//     0017 unsigned byte ?? pixel
//     ........
//     xxxx unsigned byte ?? pixel


void print_device_tensor(const char* tensor_name, float* d_ptr, int shape_size, int num_elements_to_print) {
    CUdeviceptr base;
    size_t actual_size;
    cuMemGetAddressRange(&base, &actual_size, (CUdeviceptr)d_ptr);
    size_t actual_elements = actual_size / sizeof(float);
    
    // Ensure we don't try to print more elements than exist
    num_elements_to_print = (num_elements_to_print > shape_size) ? shape_size : num_elements_to_print;
    
    // Allocate host memory for the elements we want to print
    float* h_data = (float*)malloc(num_elements_to_print * sizeof(float));
    cudaMemcpy(h_data, d_ptr, num_elements_to_print * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Print tensor information with both logical shape and actual elements
    printf("%s: dtype=float32, shape=(%d,), allocated_elements=%zu, size_in_bytes=%zu\n", 
        tensor_name, shape_size, actual_elements, actual_size);
    
    // Print elements
    printf("First %d elements: [", num_elements_to_print);
    for (int i = 0; i < num_elements_to_print; i++) {
        printf("%.4f", h_data[i]);
        if (i < num_elements_to_print - 1) {
            printf(", ");
        }
    }
    printf("]%s\n", shape_size > num_elements_to_print ? ", ...]" : "]");
    
    free(h_data);
}

__global__ void init_curand_states(curandState *states, unsigned long seed, size_t total_weights) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_weights) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

__global__ void init_weights_kernel(float *weights, size_t total_weights, float scale, curandState *states) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_weights) {
        curandState localState = states[idx];
        weights[idx] = (curand_uniform(&localState) - 0.5f) * 2 * scale;
        states[idx] = localState;
    }
}

void init_layer_cuda(GenericLayer *layer, int in_size, int out_size) {
    size_t total_weights = in_size * out_size;
    float scale = sqrtf(2.0f / in_size);
    printf("CUDA - Total weights: %zu, Scale: %f\n", total_weights, scale);

    layer->in_size = in_size;
    layer->out_size = out_size;

    CUDA_CHECK(cudaMalloc((void **)&layer->weights, total_weights * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&layer->biases, out_size * sizeof(float)));

    size_t block_size = 256;
    size_t num_blocks = (total_weights + block_size - 1) / block_size;

    curandState *d_states;
    CUDA_CHECK(cudaMalloc(&d_states, total_weights * sizeof(curandState)));

    unsigned long seed = time(NULL);
    printf("Launching kernels with %zu blocks of %zu threads\n", num_blocks, block_size);

    init_curand_states<<<num_blocks, block_size>>>(d_states, seed, total_weights);
    CUDA_CHECK(cudaGetLastError());
    
    init_weights_kernel<<<num_blocks, block_size>>>(layer->weights, total_weights, scale, d_states);
    
    CUDA_CHECK(cudaGetLastError());    
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemset(layer->biases, 0, out_size * sizeof(float)));
    CUDA_CHECK(cudaFree(d_states));
}

__global__ void normalize_imgs_kernel(unsigned char* raw_imgs, float* d_normalized_imgs, int total_pixels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_pixels) {
        d_normalized_imgs[idx] = raw_imgs[idx] / 255.0f;
    }
}

void read_mnist_imgs_cuda(const char *filename, float **d_imgs, int *num_imgs) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Error: Unable to open file %s\n", filename);
        exit(1);
    }

    int tmp, rows, cols;
    size_t read_elements;

    read_elements = fread(&tmp, sizeof(int), 1, file);
    if (read_elements != 1) {
        fprintf(stderr, "Error: Failed to read magic number\n");
        fclose(file);
        exit(1);
    }

    read_elements = fread(num_imgs, sizeof(int), 1, file);
    if (read_elements != 1) {
        fprintf(stderr, "Error: Failed to read number of images\n");
        fclose(file);
        exit(1);
    }
    *num_imgs = __builtin_bswap32(*num_imgs);

    read_elements = fread(&rows, sizeof(int), 1, file);
    if (read_elements != 1) {
        fprintf(stderr, "Error: Failed to read number of rows\n");
        fclose(file);
        exit(1);
    }
    rows = __builtin_bswap32(rows);

    read_elements = fread(&cols, sizeof(int), 1, file);
    if (read_elements != 1) {
        fprintf(stderr, "Error: Failed to read number of columns\n");
        fclose(file);
        exit(1);
    }
    cols = __builtin_bswap32(cols);

    int total_pixels = (*num_imgs) * IMAGE_H * IMAGE_W;
    unsigned char *h_imgs = (unsigned char *)malloc(total_pixels);

    read_elements = fread(h_imgs, sizeof(unsigned char), total_pixels, file);
    if (read_elements != total_pixels) {
        fprintf(stderr, "Error: Failed to read image data\n");
        free(h_imgs);
        fclose(file);
        exit(1);
    }

    fclose(file);

    // float *d_imgs;
    unsigned char *d_raw_imgs;
    CUDA_CHECK(cudaMalloc((void **)&d_raw_imgs, total_pixels * sizeof(unsigned char)));

    CUDA_CHECK(cudaMalloc((void **)d_imgs, total_pixels * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_raw_imgs, h_imgs, total_pixels * sizeof(unsigned char), cudaMemcpyHostToDevice));
    
    int block_size = 256;
    int num_blocks = (total_pixels + block_size - 1) / block_size;
    normalize_imgs_kernel<<<num_blocks, block_size>>>(d_raw_imgs, *d_imgs, total_pixels);
    CUDA_CHECK(cudaGetLastError());

    free(h_imgs);
    CUDA_CHECK(cudaFree(d_raw_imgs));
}

void read_mnist_labels_cuda(const char *filename, unsigned char **d_labels, int *num_labels) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Error: Unable to open file %s\n", filename);
        exit(1);
    }

    int tmp;
    size_t read_elements;

    read_elements = fread(&tmp, sizeof(int), 1, file);
    if (read_elements != 1) {
        fprintf(stderr, "Error: Failed to read magic number\n");
        fclose(file);
        exit(1);
    }

    read_elements = fread(num_labels, sizeof(int), 1, file);
    if (read_elements != 1) {
        fprintf(stderr, "Error: Failed to read number of labels\n");
        fclose(file);
        exit(1);
    }
    *num_labels = __builtin_bswap32(*num_labels);

    unsigned char *h_labels = (unsigned char *)malloc(*num_labels);

    read_elements = fread(h_labels, sizeof(unsigned char), (*num_labels), file);
    if (read_elements != (*num_labels)) {
        fprintf(stderr, "Error: Failed to read label data\n");
        free(h_labels);
        fclose(file);
        exit(1);
    }
    fclose(file);

    CUDA_CHECK(cudaMalloc((void **)d_labels, *num_labels * sizeof(unsigned char)));
    CUDA_CHECK(cudaMemcpy(*d_labels, h_labels, *num_labels * sizeof(unsigned char), cudaMemcpyHostToDevice));

    free(h_labels);
}

__global__ void swap_pixels_kernel(float *imgs, int i, int j, int img_size) {
    int pixel_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (pixel_idx < img_size) {
        float tmp = imgs[i * img_size + pixel_idx];
        imgs[i * img_size + pixel_idx] = imgs[j * img_size + pixel_idx];
        imgs[j * img_size + pixel_idx] = tmp;
    }
}

__global__ void shuffle_kernel(float *imgs, unsigned char *labels, int *shuffled_indices, int num_imgs, int img_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_imgs) {
        int j = shuffled_indices[idx];
        
        // Swap labels
        unsigned char tmp_label = labels[idx];
        labels[idx] = labels[j];
        labels[j] = tmp_label;
        
        int block_size = 256;
        int num_blocks = (img_size + block_size - 1) / block_size;

        swap_pixels_kernel<<<num_blocks, block_size>>>(imgs, idx, j, img_size);
        // CUDA_CHECK(cudaGetLastError());
    }
}

__global__ void shuffle_naive_kernel(float *imgs, unsigned char *labels, int *shuffled_indices, int num_imgs, int img_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_imgs) {
        int j = shuffled_indices[idx];
        
        // Swap labels
        unsigned char tmp_label = labels[idx];
        labels[idx] = labels[j];
        labels[j] = tmp_label;
        
        // Swap pixels
        for (int pixel_idx = 0; pixel_idx < img_size; pixel_idx++) {
            float tmp_pixel = imgs[idx * img_size + pixel_idx];
            imgs[idx * img_size + pixel_idx] = imgs[j * img_size + pixel_idx];
            imgs[j * img_size + pixel_idx] = tmp_pixel;
        }
    }
}

void shuffle_data_cuda(float *imgs, unsigned char *labels, int num_imgs, int img_size) {
    int *h_indices = (int *)malloc(num_imgs * sizeof(int));
    for (int i = 0; i < num_imgs; i++) {
        h_indices[i] = i;
    }
    for (int i = num_imgs-1; i > 0; i--) {
        int j = rand() % (i+1);
        int temp = h_indices[i];
        h_indices[i] = h_indices[j];
        h_indices[j] = temp;
    }

    int *d_indices;
    CUDA_CHECK(cudaMalloc(&d_indices, num_imgs * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_indices, h_indices, num_imgs * sizeof(int), cudaMemcpyHostToDevice));
    
    int block_size = 256;
    int num_blocks = (num_imgs + block_size - 1) / block_size;
    // shuffle_kernel<<<num_blocks, block_size>>>(imgs, labels, d_indices, num_imgs, img_size);
    shuffle_naive_kernel<<<num_blocks, block_size>>>(imgs, labels, d_indices, num_imgs, img_size);
    CUDA_CHECK(cudaGetLastError());

    // Synchronize to ensure all pixel swaps are complete
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(d_indices));
    free(h_indices);
}

__global__ void copy_biases_kernel(float *biases, float *out, int out_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < out_size) {
        out[idx] = biases[idx];
    }
}

__global__ void dot_product_kernel(float *weights, float *inp, float *partial_sums, int in_size, int out_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx < in_size && idy < out_size) {
        atomicAdd(&partial_sums[idy], inp[idx] * weights[idx * out_size + idy]);
    }
}

__global__ void matrix_multiply_kernel(float *weights, float *inp, float *out, float *partial_sums, int in_size, int out_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < out_size) {
        out[idx] += partial_sums[idx];
    }
}

void linear_cuda(GenericLayer *layer, float *inp, float *out) {
    int block_size = 256;
    int num_blocks = (layer->out_size + block_size - 1) / block_size;
    copy_biases_kernel<<<num_blocks, block_size>>>(layer->biases, out, layer->out_size);
    CUDA_CHECK(cudaGetLastError());

    float *d_partial_sums;
    CUDA_CHECK(cudaMalloc(&d_partial_sums, layer->out_size * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_partial_sums, 0, layer->out_size * sizeof(float)));

    dim3 block_size_dot(16, 16);
    dim3 grid_size_dot((layer->in_size + block_size_dot.x - 1) / block_size_dot.x,
                    (layer->out_size + block_size_dot.y - 1) / block_size_dot.y);

    dot_product_kernel<<<grid_size_dot, block_size_dot>>>(layer->weights, inp, d_partial_sums, layer->in_size, layer->out_size);
    CUDA_CHECK(cudaGetLastError());

    int num_blocks_multiply = (layer->out_size + block_size - 1) / block_size;
    matrix_multiply_kernel<<<num_blocks_multiply, block_size>>>(layer->weights, inp, out, d_partial_sums, layer->in_size, layer->out_size);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaFree(d_partial_sums));
}

static cublasHandle_t cublas_handle = NULL;
void init_cublas() {
    if (cublas_handle == NULL) {
        CUBLAS_CHECK(cublasCreate(&cublas_handle));
        CUBLAS_CHECK(cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_HOST));
    }
}

void cleanup_cublas() {
    if (cublas_handle != NULL) {
        CUBLAS_CHECK(cublasDestroy(cublas_handle));
        cublas_handle = NULL;
    }
}

void linear_cuda_cublas(GenericLayer *layer, float *inp, float *out) {
    int block_size = 256;
    int num_blocks = (layer->out_size + block_size - 1) / block_size;

    copy_biases_kernel<<<num_blocks, block_size>>>(layer->biases, out, layer->out_size);
    CUDA_CHECK(cudaGetLastError());

    // Perform matrix multiplication: out = weights * inp + out
    float alpha = 1.0f;
    float beta = 1.0f;
    
    CUBLAS_CHECK(cublasSgemv(cublas_handle, CUBLAS_OP_N,    // No transpose needed
                          layer->out_size,                  // Number of rows in weights
                          layer->in_size,                   // Number of columns in weights
                          &alpha,
                          layer->weights,                   // Input matrix (weights)
                          layer->out_size,                  // Leading dimension of weights
                          inp,                              // Input vector
                          1,                                // Stride of input vector
                          &beta,                            // Scale factor for biases
                          out,                              // Output vector (contains biases)
                          1));                              // Stride of output vector
}


__device__ static float atomicMax(float *address, float val) {
    int *address_as_i = (int *)address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
                    __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}


__global__ void optimized_softmax_kernel(float* __restrict__ inp, const int size) {
    // Calculate elements per block instead of using full size
    const int elements_per_block = BLOCK_SIZE * VECTOR_SIZE;
    extern __shared__ float smem[];
    float* row_shared = smem;
    float* reduction_shared = &smem[elements_per_block];
    
    const int tid = threadIdx.x;
    const int lane_id = tid % WARP_SIZE;
    const int global_id = blockIdx.x * blockDim.x + tid;
    
    // Calculate elements per thread based on block size
    const int elements_per_thread = (elements_per_block + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int vector_elements = elements_per_thread / VECTOR_SIZE;
    float reg[128];
    
    // Initialize reduction value
    if (tid == 0) {
        reduction_shared[0] = -INFINITY;
    }
    
    float local_max = -INFINITY;
    
    // Load data using float4 and find local max
    if (global_id * VECTOR_SIZE < size) {
        for (int i = 0; i < vector_elements; i++) {
            const int idx = global_id * VECTOR_SIZE + i * BLOCK_SIZE * VECTOR_SIZE;
            if (idx < size) {
                float4* vector_ptr = reinterpret_cast<float4*>(inp + idx);
                float4 elements = *vector_ptr;
                
                reg[i*4] = elements.x;
                reg[i*4 + 1] = elements.y;
                reg[i*4 + 2] = elements.z;
                reg[i*4 + 3] = elements.w;
                
                local_max = fmaxf(local_max, elements.x);
                local_max = fmaxf(local_max, elements.y);
                local_max = fmaxf(local_max, elements.z);
                local_max = fmaxf(local_max, elements.w);
            }
        }
    }
    
    // Warp-level reduction for max
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        local_max = fmaxf(local_max, __shfl_down_sync(FULL_MASK, local_max, offset));
    }
    
    // First thread in each warp updates shared memory
    if (lane_id == 0) {
        atomicMax(reduction_shared, local_max);
    }
    __syncthreads();
    
    // Get the global maximum
    const float max_val = reduction_shared[0];
    
    // Reset for sum reduction
    if (tid == 0) {
        reduction_shared[0] = 0.0f;
    }
    __syncthreads();
    
    // Compute exp and local sum
    float local_sum = 0.0f;
    if (global_id * VECTOR_SIZE < size) {
        #pragma unroll
        for (int i = 0; i < vector_elements; i++) {
            const int idx = global_id * VECTOR_SIZE + i * BLOCK_SIZE * VECTOR_SIZE;
            if (idx < size) {
                float4 output;
                output.x = expf(reg[i*4] - max_val);
                output.y = expf(reg[i*4 + 1] - max_val);
                output.z = expf(reg[i*4 + 2] - max_val);
                output.w = expf(reg[i*4 + 3] - max_val);
                
                reinterpret_cast<float4*>(row_shared + idx)[0] = output;
                
                local_sum += output.x + output.y + output.z + output.w;
            }
        }
    }
    
    // Warp-level reduction for sum
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(FULL_MASK, local_sum, offset);
    }
    
    if (lane_id == 0) {
        atomicAdd(reduction_shared, local_sum);
    }
    __syncthreads();
    
    // Final division and store
    const float final_sum = reduction_shared[0];
    if (global_id * VECTOR_SIZE < size) {
        #pragma unroll
        for (int i = 0; i < vector_elements; i++) {
            const int idx = global_id * VECTOR_SIZE + i * BLOCK_SIZE * VECTOR_SIZE;
            if (idx < size) {
                float4 output;
                output.x = row_shared[idx] / final_sum;
                output.y = row_shared[idx + 1] / final_sum;
                output.z = row_shared[idx + 2] / final_sum;
                output.w = row_shared[idx + 3] / final_sum;
                
                reinterpret_cast<float4*>(inp + idx)[0] = output;
            }
        }
    }
}


void softmax_cuda(float *d_inp, int size) {
    const int block_size = BLOCK_SIZE;
    const int elements_per_block = block_size * VECTOR_SIZE;
    const int shared_mem_size = (elements_per_block + 1) * sizeof(float);
    const int num_blocks = (size + elements_per_block - 1) / elements_per_block;
    
    optimized_softmax_kernel<<<num_blocks, block_size, shared_mem_size>>>(d_inp, size);
    CUDA_CHECK(cudaGetLastError());
}


__global__ void relu_kernel(float *out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        out[idx] = out[idx] > 0 ? out[idx] : 0;
    }
}


__global__ void optimized_relu_kernel(float* __restrict__ data, const int size) {
    const int tid = threadIdx.x;
    const int global_id = blockIdx.x * blockDim.x + tid;
    const int vector_idx = global_id * VECTOR_SIZE;
    
    // Process 4 elements per thread using float4
    if (vector_idx + VECTOR_SIZE <= size) {
        float4* vector_ptr = reinterpret_cast<float4*>(data + vector_idx);
        float4 elements = *vector_ptr;
        
        // Process in registers
        elements.x = (elements.x > 0.0f) ? elements.x : 0.0f;
        elements.y = (elements.y > 0.0f) ? elements.y : 0.0f;
        elements.z = (elements.z > 0.0f) ? elements.z : 0.0f;
        elements.w = (elements.w > 0.0f) ? elements.w : 0.0f;
        
        // Write back
        *vector_ptr = elements;
    }
    // Handle remaining elements
    else if (vector_idx < size) {
        for (int i = vector_idx; i < min(vector_idx + VECTOR_SIZE, size); i++) {
            data[i] = (data[i] > 0.0f) ? data[i] : 0.0f;
        }
    }
}


void relu_cuda(float *d_data, int size) {
    const int block_size = BLOCK_SIZE;
    const int num_blocks = (size + (block_size * VECTOR_SIZE) - 1) / (block_size * VECTOR_SIZE);
    
    optimized_relu_kernel<<<num_blocks, block_size>>>(d_data, size);
    CUDA_CHECK(cudaGetLastError());
}


__global__ void relu_derivative_kernel(float *grad, float *out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        grad[idx] *= out[idx] > 0 ? 1 : 0;
    }
}

__global__ void optimized_relu_derivative_kernel(float *grad, float *input, int size) {
    // Calculate global thread index
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int vector_size = 4;
    const int vector_idx = tid * vector_size;
    
    // Process 4 elements per thread if within bounds
    if (vector_idx < size - 3) {
        float4* grad_ptr = reinterpret_cast<float4*>(grad + vector_idx);
        float4* input_ptr = reinterpret_cast<float4*>(input + vector_idx);
        
        float4 grad_elements = *grad_ptr;
        float4 input_elements = *input_ptr;
        
        // Apply ReLU derivative
        grad_elements.x *= (input_elements.x > 0.0f) ? 1.0f : 0.0f;
        grad_elements.y *= (input_elements.y > 0.0f) ? 1.0f : 0.0f;
        grad_elements.z *= (input_elements.z > 0.0f) ? 1.0f : 0.0f;
        grad_elements.w *= (input_elements.w > 0.0f) ? 1.0f : 0.0f;
        
        *grad_ptr = grad_elements;
    }
    // Handle remaining elements individually
    else if (tid < size) {
        grad[tid] *= (input[tid] > 0.0f) ? 1.0f : 0.0f;
    }
}

__global__ void weight_gradient_kernel(float *weights, float *inp, 
    float *out_grad, int in_size, int out_size, float lr) {
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    __shared__ float shared_out_grad[TILE_SIZE];
    __shared__ float shared_inp[TILE_SIZE];
    
    if (threadIdx.x < blockDim.x && col < out_size) {
        shared_out_grad[threadIdx.x] = out_grad[col];
    }
    if (threadIdx.y < blockDim.y && row < in_size) {
        shared_inp[threadIdx.y] = inp[row];
    }
    __syncthreads();
    
    if (row < in_size && col < out_size) {
        // Coalesced access: threads in the same warp access consecutive memory locations
        weights[row * out_size + col] -= lr * (shared_out_grad[threadIdx.x] * shared_inp[threadIdx.y]);
    }
}

__global__ void input_gradient_kernel(float *weights, float *out_grad, 
    float *in_grad, int in_size, int out_size) {
    
    int row = blockIdx.x * blockDim.x + threadIdx.x;  // input dimension
    extern __shared__ float shared_out_grad[];
    
    // Cooperatively load output gradients
    if (threadIdx.x < out_size) {
        shared_out_grad[threadIdx.x] = out_grad[threadIdx.x];
    }
    __syncthreads();
    
    if (row < in_size) {
        float grad_sum = 0.0f;
        float *weight_row = &weights[row * out_size];
        
        #pragma unroll
        for (int col = 0; col < out_size; col++) {
            grad_sum += shared_out_grad[col] * weight_row[col];
        }
        atomicAdd(&in_grad[row], grad_sum);
    }
}

__global__ void bias_update_kernel(float *biases, float *out_grad, 
    int out_size, float lr) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < out_size) {
        biases[idx] -= lr * out_grad[idx];
    }
}

void backward_cuda(GenericLayer *layer, float *inp, float *out_grad, 
    float *in_grad, float lr) {
    
    dim3 block(16, 16);
    dim3 grid(
        (layer->out_size + block.x - 1) / block.x,
        (layer->in_size + block.y - 1) / block.y
    );
    size_t shared_mem_size_weight = (block.x + block.y) * sizeof(float);

    weight_gradient_kernel<<<grid, block, shared_mem_size_weight>>>(
    layer->weights, inp, out_grad, layer->in_size, layer->out_size, lr);

    int block_size = 256;
    int num_blocks_in = (layer->in_size + block_size - 1) / block_size;
    size_t shared_mem_size = layer->out_size * sizeof(float);
    
    if (in_grad) {
        input_gradient_kernel<<<num_blocks_in, block_size, shared_mem_size>>>(
            layer->weights, out_grad, in_grad, layer->in_size, layer->out_size);
    }
    
    int num_blocks_out = (layer->out_size + block_size - 1) / block_size;
    bias_update_kernel<<<num_blocks_out, block_size>>>(
        layer->biases, out_grad, layer->out_size, lr);
    
    CUDA_CHECK(cudaGetLastError());
}

void backward_cuda_cublas(GenericLayer *layer, float *inp, float *out_grad, float *in_grad, float lr) {
    static cublasHandle_t cublas_handle = NULL;
    if (cublas_handle == NULL) {
        CUBLAS_CHECK(cublasCreate(&cublas_handle));
        CUBLAS_CHECK(cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_HOST));
    }

    // Weight Gradient Update
    // weights[in_size][out_size] -= lr * (inp[1][in_size]^T * out_grad[1][out_size])
    float alpha = -lr;  // Negative lr because we want to subtract
    float beta = 1.0f;  // Use 1.0 to update existing weights
    
    CUBLAS_CHECK(cublasSgemm(cublas_handle, 
                            CUBLAS_OP_N, CUBLAS_OP_T,
                            layer->out_size,    // M: number of rows of output matrix
                            layer->in_size,     // N: number of columns of output matrix
                            1,                  // K: number of columns in inp and rows in out_grad
                            &alpha,             
                            out_grad,           // Matrix A: out_grad
                            layer->out_size,    // Leading dimension of A
                            inp,                // Matrix B: inp
                            layer->in_size,     // Leading dimension of B
                            &beta,              
                            layer->weights,     // Matrix C: weights
                            layer->out_size));  // Leading dimension of C

    // Input Gradient Computation (if needed)
    if (in_grad) {
        alpha = 1.0f;
        beta = 0.0f;
        CUBLAS_CHECK(cublasSgemv(cublas_handle, 
                                CUBLAS_OP_T,           // Transpose weights
                                layer->out_size,       // Number of rows of weights
                                layer->in_size,        // Number of columns of weights
                                &alpha,
                                layer->weights,        // Weight matrix
                                layer->out_size,       // Leading dimension of weights
                                out_grad,              // Vector to multiply with
                                1,                     // Stride of out_grad
                                &beta,
                                in_grad,               // Result vector
                                1));                   // Stride of in_grad
    }

    // Bias Update (keep as is)
    int block_size = 256;
    int num_blocks_out = (layer->out_size + block_size - 1) / block_size;
    bias_update_kernel<<<num_blocks_out, block_size>>>(
        layer->biases, out_grad, layer->out_size, lr);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void update_out_grad_kernel(float *out_grad, float *output, unsigned char *d_labels, int d_label_idx) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < OUTPUT_LAYER_SIZE) {
        out_grad[idx] = output[idx] - (idx == d_labels[d_label_idx]);
    }
}

void train_mnist_cuda(Network *net, float *inp, unsigned char *d_labels, int d_label_idx, float lr, float* final_out) {
    static float *d_final_output, *d_hidden_out, *d_out_grad, *d_hidden_grad;
    static bool first_run = true;

    if (first_run) {
        CUDA_CHECK(cudaMalloc(&d_final_output, OUTPUT_LAYER_SIZE * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_hidden_out, HIDDEN_LAYER_SIZE * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_out_grad, OUTPUT_LAYER_SIZE * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_hidden_grad, HIDDEN_LAYER_SIZE * sizeof(float)));
        first_run = false;
    }

    // ZERO GRAD
    CUDA_CHECK(cudaMemset(d_out_grad, 0, OUTPUT_LAYER_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_hidden_grad, 0, HIDDEN_LAYER_SIZE * sizeof(float)));

    // Inp to Hidden layer fwd pass
    linear_cuda(&net->hidden, inp, d_hidden_out);

    int block_size = 256;
    int num_blocks_hidden = (HIDDEN_LAYER_SIZE + block_size - 1) / block_size;
    // relu_kernel<<<num_blocks_hidden, block_size>>>(d_hidden_out, HIDDEN_LAYER_SIZE);
    int vector_block_size = 256;
    int vector_grid_size = (HIDDEN_LAYER_SIZE/4 + vector_block_size - 1) / vector_block_size;
    // optimized_relu_kernel<<<vector_grid_size, vector_block_size>>>(d_hidden_out, HIDDEN_LAYER_SIZE);
    relu_cuda(d_hidden_out, HIDDEN_LAYER_SIZE);
    CUDA_CHECK(cudaGetLastError());

    // Hidden to out layer fwd pass
    linear_cuda(&net->output, d_hidden_out, d_final_output);
    softmax_cuda(d_final_output, OUTPUT_LAYER_SIZE);

    // Compute out gradient
    int num_blocks_out = (OUTPUT_LAYER_SIZE + block_size - 1) / block_size;
    update_out_grad_kernel<<<num_blocks_out, block_size>>>(d_out_grad, d_final_output, d_labels, d_label_idx);
    CUDA_CHECK(cudaGetLastError());

    // output to hidden layer bwd pass
    backward_cuda(&net->output, d_hidden_out, d_out_grad, d_hidden_grad, lr);

    // Backprop through ReLU(derivative) Activation
    optimized_relu_derivative_kernel<<<vector_grid_size, vector_block_size>>>(d_hidden_grad, d_hidden_out, HIDDEN_LAYER_SIZE);
    // relu_derivative_kernel<<<num_blocks_hidden, block_size>>>(d_hidden_grad, d_hidden_out, HIDDEN_LAYER_SIZE);
    CUDA_CHECK(cudaGetLastError());

    // hidden to output layer bwd pass
    backward_cuda(&net->hidden, inp, d_hidden_grad, NULL, lr);
    
    // float *final_output = (float *)malloc(OUTPUT_LAYER_SIZE * sizeof(float));
    CUDA_CHECK(cudaMemcpy(final_out, d_final_output, OUTPUT_LAYER_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
}


int forward(Network *net, float *inp, float *final_out) {
    static float *d_hidden_out, *d_final_output;
    static bool first_run = true;

    if (first_run) {
        CUDA_CHECK(cudaMalloc(&d_final_output, OUTPUT_LAYER_SIZE * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_hidden_out, HIDDEN_LAYER_SIZE * sizeof(float)));
        first_run = false;
    }
    linear_cuda(&net->hidden, inp, d_hidden_out);

    int block_size = 256;
    int num_blocks = (HIDDEN_LAYER_SIZE + block_size - 1) / block_size;
    // relu_kernel<<<num_blocks, block_size>>>(d_hidden_out, HIDDEN_LAYER_SIZE);
    int vector_block_size = 256;
    int vector_grid_size = (HIDDEN_LAYER_SIZE/4 + vector_block_size - 1) / vector_block_size;
    optimized_relu_kernel<<<vector_grid_size, vector_block_size>>>(d_hidden_out, HIDDEN_LAYER_SIZE);
    CUDA_CHECK(cudaGetLastError());
    
    linear_cuda(&net->output, d_hidden_out, d_final_output);
    softmax_cuda(d_final_output, OUTPUT_LAYER_SIZE);

    // float* final_out = (float *)malloc(OUTPUT_LAYER_SIZE * sizeof(float));
    CUDA_CHECK(cudaMemcpy(final_out, d_final_output, OUTPUT_LAYER_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    
    int ans = 0;
    // gettin the max probability class from the softmax as ans
    for (int i = 0; i < OUTPUT_LAYER_SIZE; i++) {
        if (final_out[i] > final_out[ans]) {
            ans = i;
        }
    }
    return ans;
}


#ifdef RUN_MNIST_CUDA
int main() {
    Network mnist_net;
    InputData_GPU data = {0};
    float lr = LEARNING_RATE;
    clock_t start, end;
    double gpu_time_used;

    init_cublas();

    // Hardware Debug info //
    // cudaDeviceProp prop;
    // cudaGetDeviceProperties(&prop, 0);  // Assume using device 0
    // printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
    // printf("Shared memory per block: %zu bytes\n", prop.sharedMemPerBlock);
    // printf("Warp size: %d\n", prop.warpSize);

    srand(time(NULL));

    init_layer_cuda(&mnist_net.hidden, INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE);
    init_layer_cuda(&mnist_net.output, HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE);

    // imgs & labels are directly loaded to device
    read_mnist_imgs_cuda(TRAIN_IMG_PATH, &data.imgs, &data.num_imgs);
    read_mnist_labels_cuda(TRAIN_LABEL_PATH, &data.labels, &data.num_imgs);

    // shuffle_data_cuda(data.imgs, data.labels, data.num_imgs, IMAGE_SIZE);

    int train_size = (int)(data.num_imgs * TRAIN_SPLIT);
    int test_size = data.num_imgs - train_size;
    unsigned char* h_labels = (unsigned char*) malloc(data.num_imgs * sizeof(unsigned char));
    // float* h_imgs = (float*) malloc(data.num_imgs * INPUT_LAYER_SIZE * sizeof(float));
    float *final_out = (float *)malloc(OUTPUT_LAYER_SIZE * sizeof(float));
    float total_loss = 0;

    CUDA_CHECK(cudaMemcpy(h_labels, data.labels, data.num_imgs * sizeof(unsigned char), cudaMemcpyDeviceToHost));
    // CUDA_CHECK(cudaMemcpy(h_imgs, data.imgs, data.num_imgs * INPUT_LAYER_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        start = clock();
        total_loss = 0;
        for (int i = 0; i < train_size; i++) {
            float *current_img = data.imgs + (i * IMAGE_SIZE);
            if (i == 0 and epoch == 0) {
                printf("Image at iteration 0:\n");
                float host_img[IMAGE_SIZE];
                CUDA_CHECK(cudaMemcpy(host_img, current_img, IMAGE_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
                for (int row = 0; row < 28; row++) {
                    for (int col = 0; col < 28; col++) {
                        if (host_img[row * 28 + col] > 0.0f) {
                            printf("X");
                        } else {
                            printf(" ");
                        }
                    }
                    printf("\n");
                }
                printf("\n");
            }
            train_mnist_cuda(&mnist_net, current_img, data.labels, i, lr, final_out);
            // printf("CURR LABEL: %u\n", h_labels[i]);
            total_loss -= logf(fmaxf(final_out[h_labels[i]], 1e-10f));
        }

        int correct = 0;
        for (int i = train_size; i < data.num_imgs; i++) {
            float *test_img = data.imgs + (i * IMAGE_SIZE);
            if (forward(&mnist_net, test_img, final_out) == h_labels[i]) {
                correct++;
            }
        }
        end = clock();
        gpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

        printf("Epoch %d, Accuracy: %.2f%%, Avg Loss: %.4f, Time: %.2f seconds\n", 
            epoch + 1, (float)correct / test_size * 100, total_loss / train_size, gpu_time_used);
    }
    cleanup_cublas();
    CUDA_CHECK(cudaFree(mnist_net.hidden.weights));
    CUDA_CHECK(cudaFree(mnist_net.hidden.biases));
    CUDA_CHECK(cudaFree(mnist_net.output.weights));
    CUDA_CHECK(cudaFree(mnist_net.output.biases));
    CUDA_CHECK(cudaFree(data.imgs));
    CUDA_CHECK(cudaFree(data.labels));
    free(final_out);
    return 0;
}
#endif