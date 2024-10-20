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



// Kernel to initialize random number generators
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
        // weights[idx] = (5.0f - 0.5f) * 2 * scale;
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

    // unsigned long seed = time(NULL);
    unsigned long seed = 42;
    printf("Launching kernels with %zu blocks of %zu threads\n", num_blocks, block_size);


    init_curand_states<<<num_blocks, block_size>>>(d_states, seed, total_weights);

    init_weights_kernel<<<num_blocks, block_size>>>(layer->weights, total_weights, scale, d_states);

    cudaDeviceSynchronize();

    // // After initializing weights in init_layer_cuda
    // float *test_weights = (float *)malloc(total_weights * sizeof(float));
    // CUDA_CHECK(cudaMemcpy(test_weights, layer->weights, total_weights * sizeof(float), cudaMemcpyDeviceToHost));
    // printf("First 10 CUDA weights: ");
    // for (int i = 0; i < 10; i++) {
    //     printf("%f ", test_weights[i]);
    // }
    // printf("\n");
    // free(test_weights);

    cudaMemset(layer->biases, 0, out_size * sizeof(float));

    cudaFree(d_states);
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

    // Allocate device memory for raw and normalized images
    // float *d_imgs;
    unsigned char *d_raw_imgs;
    cudaError_t err = cudaMalloc((void **)&d_raw_imgs, total_pixels * sizeof(unsigned char));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory for raw images (error code %d)!\n", err);
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void **)d_imgs, total_pixels * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory for normalized images (error code %d)!\n", err);
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_raw_imgs, h_imgs, total_pixels * sizeof(unsigned char), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy image data from host to device (error code %d)!\n", err);
        exit(EXIT_FAILURE);
    }

    int block_size = 256;
    int num_blocks = (total_pixels + block_size - 1) / block_size;
    normalize_imgs_kernel<<<num_blocks, block_size>>>(d_raw_imgs, *d_imgs, total_pixels);

    free(h_imgs);
    cudaFree(d_raw_imgs);
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

    cudaMalloc((void **)d_labels, *num_labels * sizeof(unsigned char));
    cudaError_t err = cudaMemcpy(*d_labels, h_labels, *num_labels * sizeof(unsigned char), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("CUDA err: %s\n", cudaGetErrorString(err));
        exit(-1);
    }
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
    cudaMalloc(&d_indices, num_imgs * sizeof(int));
    cudaError_t err = cudaMemcpy(d_indices, h_indices, num_imgs * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("CUDA err: %s\n", cudaGetErrorString(err));
        exit(-1);
    }


    int block_size = 256;
    int num_blocks = (num_imgs + block_size - 1) / block_size;
    
    // shuffle_kernel<<<num_blocks, block_size>>>(imgs, labels, d_indices, num_imgs, img_size);
    shuffle_naive_kernel<<<num_blocks, block_size>>>(imgs, labels, d_indices, num_imgs, img_size);
    // Synchronize to ensure all pixel swaps are complete
    cudaDeviceSynchronize();

    cudaFree(d_indices);
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
    int idy = blockIdx.y;
    
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

    cudaDeviceSynchronize();

    float *d_partial_sums;
    cudaMalloc(&d_partial_sums, layer->out_size * sizeof(float));
    cudaMemset(d_partial_sums, 0, layer->out_size * sizeof(float));

    dim3 block_size_dot(16, 16);
    dim3 grid_size_dot((layer->in_size + block_size_dot.x - 1) / block_size_dot.x,
                    (layer->out_size + block_size_dot.y - 1) / block_size_dot.y);
    dot_product_kernel<<<grid_size_dot, block_size_dot>>>(layer->weights, inp, d_partial_sums, layer->in_size, layer->out_size);

    int num_blocks_multiply = (layer->out_size + block_size - 1) / block_size;
    matrix_multiply_kernel<<<num_blocks_multiply, block_size>>>(layer->weights, inp, out, d_partial_sums, layer->in_size, layer->out_size);

    cudaDeviceSynchronize();

    cudaFree(d_partial_sums);
}

__global__ void exp_subtract_sum_kernel(float *inp, int size, float *max_val, float *sum) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        inp[idx] = expf(inp[idx] - *max_val);
        atomicAdd(sum, inp[idx]);
    }
}

__global__ void divide_kernel(float *inp, int size, float *sum) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        inp[idx] /= *sum;
    }
}


void softmax_cuda(float *d_inp, int size) {
    float *d_max, *d_sum;
    cudaError_t err;

    err = cudaMalloc(&d_max, sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate d_max: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc(&d_sum, sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate d_sum: %s\n", cudaGetErrorString(err));
        cudaFree(d_max);
        exit(EXIT_FAILURE);
    }

    // Find max value
    thrust::device_ptr<float> dev_ptr(d_inp);
    thrust::device_ptr<float> max_ptr = thrust::max_element(dev_ptr, dev_ptr + size);
    err = cudaMemcpy(d_max, thrust::raw_pointer_cast(max_ptr), sizeof(float), cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy max value: %s\n", cudaGetErrorString(err));
        cudaFree(d_max);
        cudaFree(d_sum);
        exit(EXIT_FAILURE);
    }

    err = cudaMemset(d_sum, 0, sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to set d_sum to zero: %s\n", cudaGetErrorString(err));
        cudaFree(d_max);
        cudaFree(d_sum);
        exit(EXIT_FAILURE);
    }

    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;

    exp_subtract_sum_kernel<<<num_blocks, block_size>>>(d_inp, size, d_max, d_sum);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch exp_subtract_sum_kernel: %s\n", cudaGetErrorString(err));
        cudaFree(d_max);
        cudaFree(d_sum);
        exit(EXIT_FAILURE);
    }
    
    divide_kernel<<<num_blocks, block_size>>>(d_inp, size, d_sum);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch divide_kernel: %s\n", cudaGetErrorString(err));
        cudaFree(d_max);
        cudaFree(d_sum);
        exit(EXIT_FAILURE);
    }
    
    cudaFree(d_sum);
    cudaFree(d_max);
}

__global__ void relu_kernel(float *out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        out[idx] = out[idx] > 0 ? out[idx] : 0;
    }
}

__global__ void relu_derivative_kernel(float *grad, float *out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        grad[idx] *= out[idx] > 0 ? 1 : 0;
    }
}

__global__ void backward_kernel(float *weights, float *biases, float *inp, float *out_grad, float *in_grad, int in_size, int out_size, float lr) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < in_size) {
        float *weight_row_start = &weights[idx * out_size];
        float input_i = inp[idx];
        
        if (in_grad) {
            float temp_grad = 0.0f;
            for (int j = 0; j < out_size; j++) {
                temp_grad += out_grad[j] * weight_row_start[j];
                weight_row_start[j] -= lr * (out_grad[j] * input_i);
            }
            atomicAdd(&in_grad[idx], temp_grad);
        }

        else {
            for (int j = 0; j < out_size; j++) {
                weight_row_start[j] -= lr * (out_grad[j] * input_i);
            }
        }
    }
    
    if (idx < out_size) {
        biases[idx] -= lr * out_grad[idx];
    }
}

void backward_cuda(GenericLayer *layer, float *inp, float *out_grad, float *in_grad, float lr) {
    int block_size = 256;
    int num_blocks_in = (layer->in_size + block_size - 1) / block_size;
    int num_blocks_out = (layer->out_size + block_size - 1) / block_size;
    
    int num_blocks = max(num_blocks_in, num_blocks_out);
    
    backward_kernel<<<num_blocks, block_size>>>(layer->weights, layer->biases, inp, out_grad, in_grad, layer->in_size, layer->out_size, lr);
    
    cudaDeviceSynchronize();
}


__global__ void update_out_grad_kernel(float *out_grad, float *output, unsigned char *d_labels, int d_label_idx) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < OUTPUT_LAYER_SIZE) {
        out_grad[idx] = output[idx] - (idx == d_labels[d_label_idx]);
    }
}

float* train_mnist_cuda(Network *net, float *inp, unsigned char *d_labels, int d_label_idx, float lr) {
    static float *d_final_output, *d_hidden_out, *d_out_grad, *d_hidden_grad;
    static bool first_run = true;

    if (first_run) {
        cudaMalloc(&d_final_output, OUTPUT_LAYER_SIZE * sizeof(float));
        cudaMalloc(&d_hidden_out, HIDDEN_LAYER_SIZE * sizeof(float));
        cudaMalloc(&d_out_grad, OUTPUT_LAYER_SIZE * sizeof(float));
        cudaMalloc(&d_hidden_grad, HIDDEN_LAYER_SIZE * sizeof(float));
        first_run = false;
    }

    cudaMemset(d_out_grad, 0, OUTPUT_LAYER_SIZE * sizeof(float));
    cudaMemset(d_hidden_grad, 0, HIDDEN_LAYER_SIZE * sizeof(float));

    // Inp to Hidden layer fwd pass
    linear_cuda(&net->hidden, inp, d_hidden_out);
    int block_size = 256;
    int num_blocks_hidden = (HIDDEN_LAYER_SIZE + block_size - 1) / block_size;
    relu_kernel<<<num_blocks_hidden, block_size>>>(d_hidden_out, HIDDEN_LAYER_SIZE);
    cudaDeviceSynchronize();

    // Hidden to out layer fwd pass
    linear_cuda(&net->output, d_hidden_out, d_final_output);
    softmax_cuda(d_final_output, OUTPUT_LAYER_SIZE);

    // Compute out gradient
    int num_blocks_out = (OUTPUT_LAYER_SIZE + block_size - 1) / block_size;
    update_out_grad_kernel<<<num_blocks_out, block_size>>>(d_out_grad, d_final_output, d_labels, d_label_idx);

    // output to hidden layer bwd pass
    backward_cuda(&net->output, d_hidden_out, d_out_grad, d_hidden_grad, lr);

    // Backprop through ReLU(derivative) Activation
    relu_derivative_kernel<<<num_blocks_hidden, block_size>>>(d_hidden_grad, d_hidden_out, HIDDEN_LAYER_SIZE);
    cudaDeviceSynchronize();

    // hidden to output layer bwd pass
    backward_cuda(&net->hidden, inp, d_hidden_grad, NULL, lr);

    float *final_output = (float *)malloc(OUTPUT_LAYER_SIZE * sizeof(float));
    cudaError_t err = cudaMemcpy(final_output, d_final_output, OUTPUT_LAYER_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("CUDA err: %s\n", cudaGetErrorString(err));
        exit(-1);
    }
    return final_output;
}

// forward only 
int forward(Network *net, float *inp) {
    static float *d_hidden_out, *d_final_output;
    static bool first_run = true;

    if (first_run) {
        cudaMalloc(&d_final_output, OUTPUT_LAYER_SIZE * sizeof(float));
        cudaMalloc(&d_hidden_out, HIDDEN_LAYER_SIZE * sizeof(float));
        first_run = false;
    }
    linear_cuda(&net->hidden, inp, d_hidden_out);
    int block_size = 256;
    int num_blocks = (HIDDEN_LAYER_SIZE + block_size - 1) / block_size;
    relu_kernel<<<num_blocks, block_size>>>(d_hidden_out, HIDDEN_LAYER_SIZE);
    
    linear_cuda(&net->output, d_hidden_out, d_final_output);
    softmax_cuda(d_final_output, OUTPUT_LAYER_SIZE);

    float* final_out = (float *)malloc(OUTPUT_LAYER_SIZE * sizeof(float));
    cudaError_t err = cudaMemcpy(final_out, d_final_output, OUTPUT_LAYER_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("CUDA err: %s\n", cudaGetErrorString(err));
        exit(-1);
    }
    int ans = 0;
    // gettin the max probability class from the softmax as ans
    for (int i = 0; i < OUTPUT_LAYER_SIZE; i++) {
        if (final_out[i] > final_out[ans]) {
            ans = i;
        }
    }
    free(final_out);
    return ans;
}

#ifdef RUN_MNIST_CUDA
int main() {
    Network mnist_net;
    InputData_GPU data = {0};
    float lr = LEARNING_RATE;
    clock_t start, end;
    double cpu_time_used;

    // srand(time(NULL));
    srand(42);

    init_layer_cuda(&mnist_net.hidden, INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE);
    init_layer_cuda(&mnist_net.output, HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE);

    // imgs are directly loaded to device
    read_mnist_imgs_cuda(TRAIN_IMG_PATH, &data.imgs, &data.num_imgs);
    // labels stay on host for now
    read_mnist_labels_cuda(TRAIN_LABEL_PATH, &data.labels, &data.num_imgs);

    // shuffle_data_cuda(data.imgs, data.labels, data.num_imgs, IMAGE_SIZE);
    printf("SHUFFLE DONE\n");
    int train_size = (int)(data.num_imgs * TRAIN_SPLIT);
    int test_size = data.num_imgs - train_size;
    unsigned char* h_labels = (unsigned char*) malloc(data.num_imgs * sizeof(unsigned char));
    float *final_out;

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        start = clock();
        float total_loss = 0;

        cudaError_t err = cudaMemcpy(h_labels, data.labels, data.num_imgs * sizeof(unsigned char), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            printf("CUDA err: %s\n", cudaGetErrorString(err));
            exit(-1);
        }
        
        for (int i = 0; i < train_size; i++) {
            float *current_img = data.imgs + i * IMAGE_SIZE;
            final_out = train_mnist_cuda(&mnist_net, current_img, data.labels, i, lr);
            total_loss += logf(final_out[h_labels[i]] + 1e-10f);
        }

        int correct = 0;
        for (int i = train_size; i < data.num_imgs; i++) {
            float *test_img = data.imgs + i * IMAGE_SIZE;
            if (forward(&mnist_net, test_img) == h_labels[i]) {
                correct++;
            }
        }
        end = clock();
        cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

        printf("Epoch %d, Accuracy: %.2f%%, Avg Loss: %.4f, Time: %.2f seconds\n", 
            epoch + 1, (float)correct / test_size * 100, total_loss / train_size, cpu_time_used);
    }


    cudaFree(mnist_net.hidden.weights);
    cudaFree(mnist_net.hidden.biases);
    cudaFree(mnist_net.output.weights);
    cudaFree(mnist_net.output.biases);
    cudaFree(data.imgs);
    cudaFree(data.labels);
    free(final_out);
    return 0;
}
#endif