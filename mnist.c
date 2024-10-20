#include "mnist.h"

// https://x.com/konradgajdus/status/1837196363735482396
// https://github.com/konrad-gajdus/miniMNIST-c/blob/main/nn.c


void read_mnist_imgs(const char *filename, unsigned char **imgs, int *num_imgs) {
    FILE *file = fopen(filename, "rb");
    if (!file) exit(1);

    int tmp, rows, cols;
    size_t bits_read;
    bits_read = fread(&tmp, sizeof(int), 1, file); 
    if (bits_read != 1) {fprintf(stderr, "Error reading img data magic number: expected %d items, read %zu items\n", 
                            1, bits_read); fclose(file); exit(1);}

    bits_read = fread(num_imgs, sizeof(int), 1, file);
    if (bits_read != 1) {fprintf(stderr, "Error reading num_imgs: expected %d items, read %zu items\n", 
                            1, bits_read); fclose(file); exit(1);}

    *num_imgs = __builtin_bswap32(*num_imgs); //__builtin_bswap32: reverses order of bytes while keeping the bits in each byte the same

    bits_read = fread(&rows, sizeof(int), 1, file);
    if (bits_read != 1) {fprintf(stderr, "Error reading img rows: expected %d items, read %zu items\n", 
                            1, bits_read); fclose(file); exit(1);}

    bits_read = fread(&cols, sizeof(int), 1, file);
    if (bits_read != 1) {fprintf(stderr, "Error reading img cols: expected %d items, read %zu items\n", 
                            1, bits_read); fclose(file); exit(1);}

    rows = __builtin_bswap32(rows);
    cols = __builtin_bswap32(cols);

    *imgs = malloc((*num_imgs) * IMAGE_H * IMAGE_W);

    bits_read = fread(*imgs, sizeof(unsigned char), (*num_imgs) * IMAGE_H * IMAGE_W, file);
    if (bits_read != (*num_imgs) * IMAGE_H * IMAGE_W) {fprintf(stderr, "Error reading img data: expected %d items, read %zu items\n", 
                                                            ((*num_imgs) * IMAGE_H * IMAGE_W), bits_read); fclose(file); exit(1);}

    fclose(file);
}


void read_mnist_labels(const char *filename, unsigned char **labels, int *num_labels) {
    FILE *file = fopen(filename, "rb");
    if (!file) exit(1);

    int tmp;
    size_t bits_read;

    bits_read = fread(&tmp, sizeof(int), 1, file);
    if (bits_read != 1) {fprintf(stderr, "Error reading labels magic num: expected %d items, read %zu items\n", 
                            1, bits_read); fclose(file); exit(1);}

    bits_read = fread(num_labels, sizeof(int), 1, file);
    if (bits_read != 1) {fprintf(stderr, "Error reading num_labels: expected %d items, read %zu items\n", 
                            1, bits_read); fclose(file); exit(1);}

    *num_labels = __builtin_bswap32(*num_labels);

    *labels = malloc(*num_labels);
    bits_read = fread(*labels, sizeof(unsigned char), (*num_labels), file);
    if (bits_read != (*num_labels)) {fprintf(stderr, "Error reading labels: expected %d items, read %zu items\n", 
                                        *num_labels, bits_read); fclose(file); exit(1);}

    fclose(file);
}


void softmax(float *inp, int size) {
    float max = inp[0], sum = 0;

    for (int i = 1; i < size; i++) {
        if (inp[i] > max) {max = inp[i];}
    }
    for (int i = 0; i < size; i++) {
        inp[i] = expf(inp[i] - max);
        sum += inp[i];
    }
    for (int i = 0; i < size; i++) {
        inp[i] = inp[i] / sum;
    }
}


void shuffle_data(unsigned char *imgs, unsigned char *labels, int n) {
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i+1);
        for (int k = 0; k < INPUT_LAYER_SIZE; k++) {
            float tmp = imgs[i * INPUT_LAYER_SIZE + k];
            imgs[i * INPUT_LAYER_SIZE + k] = imgs[j * INPUT_LAYER_SIZE + k];
            imgs[j * INPUT_LAYER_SIZE + k] = tmp;
        }
        unsigned char tmp = labels[i];
        labels[i] = labels[j];
        labels[j] = tmp;
    }
}


void linear(GenericLayer *layer, float *inp, float *out) {
    for (int i = 0; i < layer->out_size; i++) {
        //init biases of layer to be calcd
        out[i] = layer->biases[i];
    }
    for (int x = 0; x < layer->in_size; x++) {
        float input_x = inp[x];
        //getting the start of each weight row, so each index doesn't need to be recalculated
        float *weight_row_start = &layer->weights[x * layer->out_size];
        for (int j = 0; j < layer->out_size; j++) {
            out[j] += input_x * weight_row_start[j];
        }
    }
}


void backward(GenericLayer *layer, float *inp, float *out_grad, float *in_grad, float lr) {
    for (int i = 0; i < layer->in_size; i++) {
        float *weight_row_start = &layer->weights[i * layer->out_size];
        float input_i = inp[i];
        if (in_grad) {
            in_grad[i] = 0.0f;
            for (int j = 0; j < layer->out_size; j++) {
                in_grad[i] += out_grad[j] * weight_row_start[j];
                weight_row_start[j] -= lr * (out_grad[j] * input_i);
            }
        }
        else {
            for (int j = 0; j < layer->out_size; j++) {
                weight_row_start[j] -= lr * (out_grad[j] * input_i);
            }
        }
        
    }
    for (int i = 0; i < layer->out_size; i++) {
        layer->biases[i] -= lr * out_grad[i];
    }
}


void init_layer(GenericLayer *layer, int in_size, int out_size) {
    size_t n = in_size * out_size;
    //to normally distribute our weights with rand values
    float scale = sqrtf(2.0f / in_size);
    printf("CPU - Total weights : %zu, Scale: %f\n", n, scale);

    layer->in_size = in_size;
    layer->out_size = out_size;
    layer->weights = (float *)malloc(n * sizeof(float));
    // layer->weights = (float *)calloc(n, sizeof(float));
    layer->biases = (float *)calloc(out_size, sizeof(float)); //minim-MNIST only has biases for opt neurons

    // 'He' initialization is used to set the weights
    for (int i = 0; i < n; i++) { // sets weights to a random value and scales using inp size
        // layer->weights[i] = (5.0f - 0.5f) * 2 * scale;
        layer->weights[i] = ((float)rand() / RAND_MAX - 0.5f) * 2 * scale;
    }
}


float* train_mnist(Network *net, float *inp, int label, float lr) {
    static float final_output[OUTPUT_LAYER_SIZE];
    float hidden_out[HIDDEN_LAYER_SIZE];
    float out_grad[OUTPUT_LAYER_SIZE] = {0}, hidden_grad[HIDDEN_LAYER_SIZE] = {0};

    // Inp to Hidden layer fwd pass
    linear(&net->hidden, inp, hidden_out);
    // ReLU activation
    for (int i = 0; i < HIDDEN_LAYER_SIZE; i++) {
        hidden_out[i] = hidden_out[i] > 0 ? hidden_out[i] : 0;
    }

    // Hidden to out layer fwd pass
    linear(&net->output, hidden_out, final_output);
    softmax(final_output, OUTPUT_LAYER_SIZE);

    // Compute out gradient
    for (int i = 0; i < OUTPUT_LAYER_SIZE; i++) { // goes thru all numericals
        out_grad[i] = final_output[i] - (i == label);
    }
    
    // output to hidden layer bwd pass
    backward(&net->output, hidden_out, out_grad, hidden_grad, lr);

    // Backprop through ReLU(derivative) Activation
    for (int i = 0; i < HIDDEN_LAYER_SIZE; i++) {
        hidden_grad[i] *= hidden_out[i] > 0 ? 1 : 0; 
    }

    //hidden to output layer bwd pass
    backward(&net->hidden, inp, hidden_grad, NULL, lr);

    return final_output;
}

// forward only loop aKa inference
int inference(Network *net, float *inp) {
    float hidden_out[HIDDEN_LAYER_SIZE], final_out[OUTPUT_LAYER_SIZE];

    linear(&net->hidden, inp, hidden_out);
    // ReLU
    for (int i = 0; i < HIDDEN_LAYER_SIZE; i++){
        hidden_out[i] = hidden_out[i] > 0 ? hidden_out[i] : 0;
    }
    linear(&net->output, hidden_out, final_out);
    softmax(final_out, OUTPUT_LAYER_SIZE);

    int ans = 0;
    // gettin the max probability class from the softmax as ans
    for (int i = 0; i < OUTPUT_LAYER_SIZE; i++) {
        if (final_out[i] > final_out[ans]) {
            ans = i;
        }
    }
    return ans;
}

#ifdef RUN_MNIST_CPU
int main() {
    Network mnist_net;
    InputData_CPU data = {0};
    float lr = LEARNING_RATE, img[INPUT_LAYER_SIZE];
    clock_t start, end;
    double cpu_time_used;

    srand(42);
    // srand(time(NULL));

    init_layer(&mnist_net.hidden, INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE);
    init_layer(&mnist_net.output, HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE);

    read_mnist_imgs(TRAIN_IMG_PATH, &data.imgs, &data.num_imgs);
    read_mnist_labels(TRAIN_LABEL_PATH, &data.labels, &data.num_imgs);

    shuffle_data(data.imgs, data.labels, data.num_imgs);

    int train_size = (int)(data.num_imgs * TRAIN_SPLIT);
    int test_size = data.num_imgs - train_size;

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        start = clock();
        float total_loss = 0;
        for (int i = 0; i < train_size; i++) {
            for (int k = 0; k < INPUT_LAYER_SIZE; k++) {
                img[k] = data.imgs[i* INPUT_LAYER_SIZE + k] / 255.0f;
            }
            float *final_out = train_mnist(&mnist_net, img, data.labels[i], lr);
            // what loss func
            total_loss += logf(final_out[data.labels[i]] + 1e-10f);
        }

        int correct = 0;
        for (int i = train_size; i < data.num_imgs; i++) {
            for (int k = 0; k <INPUT_LAYER_SIZE; k++) {
                img[k] = data.imgs[i*INPUT_LAYER_SIZE+k] / 255.0f;
            }
            if (inference(&mnist_net, img) == data.labels[i]) {
                correct++;
            }
        }
        end = clock();
        cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

        printf("Epoch %d, Accuracy: %.2f%%, Avg Loss: %.4f, Time: %.2f seconds\n", 
               epoch + 1, (float)correct / test_size * 100, total_loss / train_size, cpu_time_used);
    }

    free(mnist_net.hidden.weights);
    free(mnist_net.hidden.biases);
    free(mnist_net.output.weights);
    free(mnist_net.output.biases);
    free(data.imgs);
    free(data.labels);

    return 0;
}
#endif