#ifndef MNIST_H
#define MNIST_H
#include "consts.cuh"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

void softmax(float *inp, int size);
void linear(GenericLayer *layer, float *inp, float *out);
void read_mnist_imgs(const char *filename, unsigned char **imgs, int *num_imgs);
void read_mnist_labels(const char *filename, unsigned char **labels, int *num_labels);
void init_layer(GenericLayer *layer, size_t in_size, size_t out_size);
void backward(GenericLayer *layer, float *inp, float *out_grad, float *in_grad, float lr);


#ifdef __cplusplus
}
#endif

#endif