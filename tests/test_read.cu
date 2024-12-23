#include "../kernels.cuh"
#include "../mnist.h"

#define TESTING_IMG_PATH "../data/train-images.idx3-ubyte"
#define TESTING_LABEL_PATH "../data/train-labels.idx1-ubyte"

int compare_arrays(float *a, float *b, int size) {
    for (int i = 0; i < size; i++) {
        if (abs(a[i] - b[i]) > 1e-5) {
            return 0;
        }
    }
    return 1;
}

int compare_labels(unsigned char *a, unsigned char *b, int size) {
    for (int i = 0; i < size; i++) {
        if (abs(a[i] - b[i]) > 1e-5) {
            return 0;
        }
    }
    return 1;
}

void test_backward() {
    // Setup
    GenericLayer layer = {0};
    layer.in_size = INPUT_LAYER_SIZE;
    layer.out_size = HIDDEN_LAYER_SIZE;
    layer.weights = (float *)malloc(INPUT_LAYER_SIZE * HIDDEN_LAYER_SIZE * sizeof(float));
    layer.biases = (float *)malloc(HIDDEN_LAYER_SIZE * sizeof(float));

    GenericLayer d_layer = {0};
    d_layer.in_size = INPUT_LAYER_SIZE;
    d_layer.out_size = HIDDEN_LAYER_SIZE;
    CUDA_CHECK(cudaMalloc((void **)&d_layer.weights, INPUT_LAYER_SIZE * HIDDEN_LAYER_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_layer.biases, HIDDEN_LAYER_SIZE * sizeof(float)));

    // Initialize weights and biases with some values
    for (int i = 0; i < INPUT_LAYER_SIZE * HIDDEN_LAYER_SIZE; i++) {
        layer.weights[i] = (float)rand() / RAND_MAX;
    }
    for (int i = 0; i < HIDDEN_LAYER_SIZE; i++) {
        layer.biases[i] = (float)rand() / RAND_MAX;
    }

    // Copy to GPU
    CUDA_CHECK(cudaMemcpy(d_layer.weights, layer.weights, INPUT_LAYER_SIZE * HIDDEN_LAYER_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_layer.biases, layer.biases, HIDDEN_LAYER_SIZE * sizeof(float), cudaMemcpyHostToDevice));

    // Create input, output gradient, and input gradient
    float *inp = (float *)malloc(INPUT_LAYER_SIZE * sizeof(float));
    float *out_grad = (float *)malloc(HIDDEN_LAYER_SIZE * sizeof(float));
    float *in_grad = (float *)malloc(INPUT_LAYER_SIZE * sizeof(float));

    float *d_inp, *d_out_grad, *d_in_grad;
    CUDA_CHECK(cudaMalloc((void **)&d_inp, INPUT_LAYER_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_out_grad, HIDDEN_LAYER_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_in_grad, INPUT_LAYER_SIZE * sizeof(float)));

    // Initialize input and output gradient with some values
    for (int i = 0; i < INPUT_LAYER_SIZE; i++) {
        inp[i] = (float)rand() / RAND_MAX;
    }
    for (int i = 0; i < HIDDEN_LAYER_SIZE; i++) {
        out_grad[i] = (float)rand() / RAND_MAX;
    }

    // Copy to GPU
    CUDA_CHECK(cudaMemcpy(d_inp, inp, INPUT_LAYER_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_out_grad, out_grad, HIDDEN_LAYER_SIZE * sizeof(float), cudaMemcpyHostToDevice));

    // Run CPU version
    float lr = 0.01;
    backward(&layer, inp, out_grad, in_grad, lr);

    // Run GPU version
    backward_cuda(&d_layer, d_inp, d_out_grad, d_in_grad, lr);

    // Copy results back to CPU
    float *gpu_weights = (float *)malloc(INPUT_LAYER_SIZE * HIDDEN_LAYER_SIZE * sizeof(float));
    float *gpu_biases = (float *)malloc(HIDDEN_LAYER_SIZE * sizeof(float));
    float *gpu_in_grad = (float *)malloc(INPUT_LAYER_SIZE * sizeof(float));

    CUDA_CHECK(cudaMemcpy(gpu_weights, d_layer.weights, INPUT_LAYER_SIZE * HIDDEN_LAYER_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gpu_biases, d_layer.biases, HIDDEN_LAYER_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gpu_in_grad, d_in_grad, INPUT_LAYER_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

    // Compare results
    int result_weights = compare_arrays(layer.weights, gpu_weights, INPUT_LAYER_SIZE * HIDDEN_LAYER_SIZE);
    int result_biases = compare_arrays(layer.biases, gpu_biases, HIDDEN_LAYER_SIZE);
    int result_in_grad = compare_arrays(in_grad, gpu_in_grad, INPUT_LAYER_SIZE);

    if (result_weights && result_biases && result_in_grad) {
        printf("\nBackward Test Passed: CPU and CUDA results match.\n");
    } else {
        printf("\nBackward Test Failed: CPU and CUDA results do not match.\n");
    }

    // Clean up
    free(layer.weights);
    free(layer.biases);
    free(inp);
    free(out_grad);
    free(in_grad);
    free(gpu_weights);
    free(gpu_biases);
    free(gpu_in_grad);
    cudaFree(d_layer.weights);
    cudaFree(d_layer.biases);
    cudaFree(d_inp);
    cudaFree(d_out_grad);
    cudaFree(d_in_grad);
}

void test_update_out_grad() {
    // Setup
    float *output = (float *)malloc(OUTPUT_LAYER_SIZE * sizeof(float));
    float *out_grad = (float *)malloc(OUTPUT_LAYER_SIZE * sizeof(float));
    unsigned char *labels = (unsigned char *)malloc(sizeof(unsigned char));
    int label_idx = 0;

    float *d_output, *d_out_grad;
    unsigned char *d_labels;
    CUDA_CHECK(cudaMalloc((void **)&d_output, OUTPUT_LAYER_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_out_grad, OUTPUT_LAYER_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_labels, sizeof(unsigned char)));

    // Initialize output with some values and set a label
    for (int i = 0; i < OUTPUT_LAYER_SIZE; i++) {
        output[i] = (float)rand() / RAND_MAX;
    }
    labels[0] = rand() % OUTPUT_LAYER_SIZE;

    // Copy to GPU
    CUDA_CHECK(cudaMemcpy(d_output, output, OUTPUT_LAYER_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_labels, labels, sizeof(unsigned char), cudaMemcpyHostToDevice));

    // Run CPU version
    for (int i = 0; i < OUTPUT_LAYER_SIZE; i++) {
        out_grad[i] = output[i] - (i == labels[label_idx]);
    }

    // Run GPU version
    int block_size = 256;
    int num_blocks = (OUTPUT_LAYER_SIZE + block_size - 1) / block_size;
    update_out_grad_kernel<<<num_blocks, block_size>>>(d_out_grad, d_output, d_labels, label_idx);
    cudaDeviceSynchronize();

    // Copy results back to CPU
    float *gpu_out_grad = (float *)malloc(OUTPUT_LAYER_SIZE * sizeof(float));
    CUDA_CHECK(cudaMemcpy(gpu_out_grad, d_out_grad, OUTPUT_LAYER_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

    // Compare results
    int result = compare_arrays(out_grad, gpu_out_grad, OUTPUT_LAYER_SIZE);

    if (result) {
        printf("\nUpdate Out Grad Test Passed: CPU and CUDA results match.\n");
    } else {
        printf("\nUpdate Out Grad Test Failed: CPU and CUDA results do not match.\n");
    }

    // Clean up
    free(output);
    free(out_grad);
    free(labels);
    free(gpu_out_grad);
    cudaFree(d_output);
    cudaFree(d_out_grad);
    cudaFree(d_labels);
}


#ifdef RUN_READ_TEST
int main() {
    printf("\n READ Test with each img size: %d \n", INPUT_LAYER_SIZE);
    InputData_CPU cpu_data = {0};
    InputData_GPU gpu_data = {0};

    cudaEvent_t cpu_start, cpu_stop;
    float cpu_time;
    cudaEventCreate(&cpu_start);
    cudaEventCreate(&cpu_stop);
    cudaEventRecord(cpu_start);
    read_mnist_imgs(TESTING_IMG_PATH, &cpu_data.imgs, &cpu_data.num_imgs);
    read_mnist_labels(TESTING_LABEL_PATH, &cpu_data.labels, &cpu_data.num_imgs);
    // printf("cpu_data.num_imgs: %d\n", cpu_data.num_imgs);
    float *h_imgs_cpu = (float *) malloc(cpu_data.num_imgs * INPUT_LAYER_SIZE * sizeof(float));
    for (int i = 0; i < cpu_data.num_imgs; i++) {
        for (int pixel = 0; pixel < INPUT_LAYER_SIZE; pixel++) {
            h_imgs_cpu[i * INPUT_LAYER_SIZE + pixel] = cpu_data.imgs[i* INPUT_LAYER_SIZE + pixel] / 255.0f;
        }
    }
    cudaEventRecord(cpu_stop);
    cudaEventSynchronize(cpu_stop);
    cudaEventElapsedTime(&cpu_time, cpu_start, cpu_stop);

    float *h_imgs_gpu = (float *) malloc(cpu_data.num_imgs * INPUT_LAYER_SIZE * sizeof(float));
    unsigned char *h_labels_gpu = (unsigned char *) malloc(cpu_data.num_imgs * sizeof(unsigned char));
    cudaEvent_t gpu_start, gpu_stop;
    float gpu_time;
    cudaEventCreate(&gpu_start);
    cudaEventCreate(&gpu_stop);
    cudaEventRecord(gpu_start);
    read_mnist_imgs_cuda(TESTING_IMG_PATH, &gpu_data.imgs, &gpu_data.num_imgs);
    read_mnist_labels_cuda(TESTING_LABEL_PATH, &gpu_data.labels, &gpu_data.num_imgs);
    cudaEventRecord(gpu_stop);
    cudaEventSynchronize(gpu_stop);
    cudaEventElapsedTime(&gpu_time, gpu_start, gpu_stop);
    cudaMemcpy(h_imgs_gpu, gpu_data.imgs, cpu_data.num_imgs * INPUT_LAYER_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_labels_gpu, gpu_data.labels, cpu_data.num_imgs * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    int result = compare_arrays(h_imgs_cpu, h_imgs_gpu, cpu_data.num_imgs * INPUT_LAYER_SIZE);
    if (result) {
        printf("\nRead/Normalize Test Passed: CPU and CUDA results match.\n");
    } else {
        printf("\nRead/Normalize Test Failed: CPU and CUDA results do not match.\n");
    }

    int result_labels = compare_labels(cpu_data.labels, h_labels_gpu, cpu_data.num_imgs);
    if (result_labels) {
        printf("\nRead Labels Test Passed: CPU and CUDA results match.\n");
    } else {
        printf("\nRead Labels Test Failed: CPU and CUDA results do not match.\n");
    }
    printf(" CPU time: %f ms\n", cpu_time);
    printf(" GPU time: %f ms\n", gpu_time);
    printf(" Speedup: %fx\n", cpu_time / gpu_time);


    // GenericLayer hidden = {0};
    // hidden.in_size = INPUT_LAYER_SIZE;
    // hidden.out_size = HIDDEN_LAYER_SIZE;
    // hidden.weights = (float *) malloc(INPUT_LAYER_SIZE * HIDDEN_LAYER_SIZE * sizeof(float));
    // hidden.biases = (float *) calloc(HIDDEN_LAYER_SIZE, sizeof(float));

    // GenericLayer d_hidden = {0};
    // d_hidden.in_size = INPUT_LAYER_SIZE;
    // d_hidden.out_size = HIDDEN_LAYER_SIZE;
    // cudaMalloc((void **) &d_hidden.weights, INPUT_LAYER_SIZE * HIDDEN_LAYER_SIZE * sizeof(float));
    // CUDA_CHECK(cudaMalloc((void **) &d_hidden.biases, HIDDEN_LAYER_SIZE * sizeof(float)));
    // cudaMemset(d_hidden.biases, 0, HIDDEN_LAYER_SIZE * sizeof(float));

    // float *h_hidden_out = (float *) malloc(cpu_data.num_imgs * HIDDEN_LAYER_SIZE * sizeof(float));
    // cudaEvent_t cpu_linear_start, cpu_linear_stop;
    // float cpu_linear_time;
    // cudaEventCreate(&cpu_linear_start);
    // cudaEventCreate(&cpu_linear_stop);
    // cudaEventRecord(cpu_linear_start);
    // linear(&hidden, h_imgs_cpu, h_hidden_out);
    // cudaEventRecord(cpu_linear_stop);
    // cudaEventSynchronize(cpu_linear_stop);
    // cudaEventElapsedTime(&cpu_linear_time, cpu_linear_start, cpu_linear_stop);

    // float *d_hidden_out;
    // CUDA_CHECK(cudaMalloc((void **)&d_hidden_out, cpu_data.num_imgs * HIDDEN_LAYER_SIZE * sizeof(float)));

    // cudaEvent_t gpu_linear_start, gpu_linear_stop;
    // float gpu_linear_time;
    // cudaEventCreate(&gpu_linear_start);
    // cudaEventCreate(&gpu_linear_stop);
    // cudaEventRecord(gpu_linear_start);
    // linear_cuda(&d_hidden, gpu_data.imgs, d_hidden_out);
    // cudaEventRecord(gpu_linear_stop);
    // cudaEventSynchronize(gpu_linear_stop);
    // cudaEventElapsedTime(&gpu_linear_time, gpu_linear_start, gpu_linear_stop);

    // float *h_hidden_out_cuda = (float*) malloc(cpu_data.num_imgs * HIDDEN_LAYER_SIZE * sizeof(float));
    // cudaMemcpy(h_hidden_out_cuda, d_hidden_out, cpu_data.num_imgs * HIDDEN_LAYER_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    // int result_linear = compare_arrays(h_hidden_out, h_hidden_out_cuda, cpu_data.num_imgs * HIDDEN_LAYER_SIZE);
    // if (result_linear) {
    //     printf("\nLinear Test Passed: CPU and CUDA results match.\n");
    // } else {
    //     printf("\nLinear Test Failed: CPU and CUDA results do not match.\n");
    // }
    // printf(" CPU time: %f ms\n", cpu_linear_time);
    // printf(" GPU time: %f ms\n", gpu_linear_time);
    // printf(" Speedup: %fx\n", cpu_linear_time / gpu_linear_time);

    // test_backward();
    // test_update_out_grad();
    

    cudaFree(gpu_data.imgs);
    cudaFree(gpu_data.labels);
    // cudaFree(d_hidden.weights);
    // cudaFree(d_hidden.biases);
    free(cpu_data.imgs);
    free(cpu_data.labels);
    free(h_imgs_gpu);
    free(h_imgs_cpu);
    free(h_labels_gpu);
    // free(h_hidden_out);
    // free(h_hidden_out_cuda);
    // free(hidden.weights);
    // free(hidden.biases);
}
#endif