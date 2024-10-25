#ifndef CONSTS_H
#define CONSTS_H

#define IMAGE_H 28
#define IMAGE_W 28
#define IMAGE_SIZE (IMAGE_H * IMAGE_W)
#define INPUT_LAYER_SIZE 784
#define HIDDEN_LAYER_SIZE 256
#define OUTPUT_LAYER_SIZE 10
#define LEARNING_RATE 0.0005f
#define EPOCHS 20
#define BATCH_SIZE 64
#define TRAIN_SPLIT 0.8
#define PRINT_INTERVAL 1000
#define TILE_SIZE 32

#define TRAIN_IMG_PATH "./data/train-images.idx3-ubyte"
#define TRAIN_LABEL_PATH "./data/train-labels.idx1-ubyte"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            cudaDeviceReset(); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

typedef struct {
    float *weights, *biases;
    int in_size, out_size;
} GenericLayer;

typedef struct {
    GenericLayer hidden, output;
} Network;

typedef struct {
    unsigned char *labels, *imgs;
    int num_imgs;
} InputData_CPU;

typedef struct {
    unsigned char *labels;
    float *imgs;
    int num_imgs;
} InputData_GPU;

#endif