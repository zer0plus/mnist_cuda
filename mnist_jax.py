import jax
import jax.numpy as jnp
from jax import grad, jit, random
import struct
import numpy as np
import time

# Constants
INPUT_SIZE = 784
HIDDEN_SIZE = 256
OUTPUT_SIZE = 10
LEARNING_RATE = 0.0005
EPOCHS = 20
TRAIN_SPLIT = 0.8

def read_mnist_images(filename):
    with open(filename, 'rb') as f:
        magic = struct.unpack(">I", f.read(4))[0]
        num_imgs = struct.unpack(">I", f.read(4))[0]
        rows = struct.unpack(">I", f.read(4))[0]
        cols = struct.unpack(">I", f.read(4))[0]
        images = np.frombuffer(f.read(), dtype=np.uint8)
        return images.reshape(-1, INPUT_SIZE).astype(np.float32) / 255.0

def read_mnist_labels(filename):
    with open(filename, 'rb') as f:
        magic = struct.unpack(">I", f.read(4))[0]
        num_labels = struct.unpack(">I", f.read(4))[0]
        return np.frombuffer(f.read(), dtype=np.uint8)

def init_params(key):
    # He initialization
    key1, key2 = random.split(key)
    scale1 = jnp.sqrt(2.0 / INPUT_SIZE)
    scale2 = jnp.sqrt(2.0 / HIDDEN_SIZE)
    return {
        'hidden': {
            'w': random.normal(key1, (INPUT_SIZE, HIDDEN_SIZE)) * scale1,
            'b': jnp.zeros(HIDDEN_SIZE)
        },
        'output': {
            'w': random.normal(key2, (HIDDEN_SIZE, OUTPUT_SIZE)) * scale2,
            'b': jnp.zeros(OUTPUT_SIZE)
        }
    }

@jit
def forward(params, x):
    # Hidden layer with built-in relu
    h1 = jax.nn.relu(jnp.dot(x, params['hidden']['w']) + params['hidden']['b'])
    # Output layer with built-in softmax
    logits = jnp.dot(h1, params['output']['w']) + params['output']['b']
    return jax.nn.softmax(logits)

@jit
def loss_fn(params, x, y):
    pred = forward(params, x)
    return -jnp.log(pred[y] + 1e-10)

@jit
def update(params, x, y):
    grads = jax.grad(loss_fn)(params, x, y)
    # return jax.tree_map(
    #     lambda p, g: p - LEARNING_RATE * g, 
    #     params, 
    #     grads
    # )
    return jax.tree.map(
        lambda p, g: p - LEARNING_RATE * g, 
        params, 
        grads
    )

def train_mnist():
    # Load data
    train_images = read_mnist_images("./data/train-images.idx3-ubyte")
    train_labels = read_mnist_labels("./data/train-labels.idx1-ubyte")
    
    # Initialize parameters
    key = random.PRNGKey(0)
    params = init_params(key)
    
    # Training loop
    n_samples = len(train_images)
    train_size = int(n_samples * TRAIN_SPLIT)
    test_size = n_samples - train_size
    
    for epoch in range(EPOCHS):
        start_time = time.time()
        total_loss = 0.0
        
        # Training
        for i in range(train_size):
            x = train_images[i]
            y = train_labels[i]
            total_loss += loss_fn(params, x, y)
            params = update(params, x, y)
        
        # Testing
        correct = 0
        for i in range(train_size, n_samples):
            x = train_images[i]
            pred = jnp.argmax(forward(params, x))
            if pred == train_labels[i]:
                correct += 1
        
        end_time = time.time()
        epoch_time = end_time - start_time
        print(f"Epoch {epoch + 1}, Accuracy: {(correct/test_size)*100:.2f}%, Avg Loss: {total_loss/train_size:.4f}, Time: {epoch_time:.2f} seconds")
        # print(f"Epoch {epoch + 1}, Accuracy: {(correct/test_size)*100:.2f}%, Avg Loss: {total_loss/train_size:.4f}")

if __name__ == "__main__":
    train_mnist()