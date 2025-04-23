# mnist_cuda
Training/Inference implementation for MNIST from scratch using CUDA

![MNIST Digits Example](https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png)

## Project Overview

This project implements a neural network for MNIST handwritten digit classification with a focus on GPU acceleration using CUDA. The architecture is a simple feed-forward network with:

- Input layer (784 neurons - 28x28 pixel images)
- Hidden layer (256 neurons with ReLU activation)
- Output layer (10 neurons with softmax activation)

Each component has been implemented twice. Once in standard C for CPU and once in CUDA with multiple GPU optimizations to demonstrate the performance benefits of GPU acceleration for neural network training.

## Hardware Specifications

All benchmarks were performed on the following hardware:

| Component | Specification |
|-----------|---------------|
| CPU | 2 AMD EPYC 7551P vCPUs |
| RAM | 16 GB |
| GPU | NVIDIA RTX A4000 (16 GB VRAM) |

## Benchmarks & Optimizations

### 1. Linear Layer Forward Pass

The linear layer computes `output = weights × input + biases`, forming the core computation in neural networks.

**Optimizations:**
- Used cuBLAS for matrix multiplication (`cublasSgemv`)
- Employed a specialized kernel for bias addition
- Leveraged vectorized memory access patterns for coalesced reads/writes
- Optimized thread block configurations for maximum occupancy

**Results:**

![Linear Speedup](benchmarks/plots/linear_speedup.png)
![Linear Times](benchmarks/plots/linear_times.png)

**Analysis:**
- GPU acceleration provides significant speedup for larger matrices
- At matrix sizes smaller than 784×256, CPU can be faster due to device memory transfer overhead
- Maximum observed speedup: ~50x for the largest tested matrices
- Performance scales well with increasing matrix dimensions

### 2. ReLU Activation

The ReLU activation function applies a simple non-linear transform: `f(x) = max(0, x)`.

**Optimizations:**
- Vectorized operations using float4 to process 4 elements per thread
- Optimized memory access patterns for coalesced global memory operations
- Used shared memory to reduce global memory transactions
- Implemented efficient warp-level operations

**Results:**

![ReLU Forward Speedup](benchmarks/plots/forward_relu_speedup.png)
![ReLU Derivative Speedup](benchmarks/plots/derivative_relu_speedup.png)

**Analysis:**
- Forward ReLU shows poor GPU acceleration for small to medium sizes, with speedup < 1 until ~2^24 elements
- Forward pass only achieves ~3x maximum speedup at the largest tested size (2^26)
- In stark contrast, ReLU derivative shows exceptional GPU scaling, reaching ~130x speedup
- Derivative operation begins showing benefits at moderate sizes (~2^18) and scales exponentially
- The forward ReLU's limited arithmetic intensity makes it memory-bound, while the derivative implementation allows better computational parallelism

### 3. Softmax Activation

The softmax activation normalizes the output of the network into a probability distribution.

**Optimizations:**
- Used parallel reduction for max and sum calculations
- Leveraged shared memory to reduce global memory accesses
- Utilized warp-level primitives (`__shfl_down_sync`) for efficient reduction
- Implemented vectorized operations with float4 for higher throughput

**Results:**

![Softmax Speedup](benchmarks/plots/softmax_speedup.png)
![Softmax Times](benchmarks/plots/softmax_times.png)

**Analysis:**
- Softmax illustrates the classic case where operations with high arithmetic intensity but parallel structure excel on GPU architecture
- Small input sizes(< 2<sup>10</sup>) do noy show any benefit due to kernel launch overhead
- Large input sizes demonstrate speedup that is too great to measure against the CPU
- The exponential operations benefit greatly from GPU's specialized math units

### 4. Backward Pass (Gradient Computation)

The backward pass updates network weights based on computed gradients, critical for training.

**Optimizations:**
- Used cuBLAS for matrix operations in gradient computation
- Implemented separate custom kernels with tiling for weight updates
- Used shared memory for gradient accumulation
- Optimized thread block configurations for both large and small matrices

**Results:**

![Backward Speedup vs CPU](benchmarks/plots/backward_speedup_vs_cpu.png)
![Backward Custom vs cuBLAS](benchmarks/plots/backward_custom_vs_cublas.png)

**Analysis:**
- Custom CUDA implementation sometimes outperforms cuBLAS for specific small matrix sizes
- Backward pass shows greatest speedup for large input sizes
- Performance scales well with network size
- Custom kernels provide flexibility for network-specific optimizations

### 5. Data Loading

This primarily consists of loading and normalizing the MNIST image data.

**Optimizations:**
- Implemented parallel data normalization on GPU
- Combined file reading and preprocessing in GPU implementation
- Leveraged asynchronous memory operations where possible
- Used pinned memory for faster host-device transfers

**Results:**

![MNIST Read Times](benchmarks/plots/mnist_read_times.png)
![MNIST Read Speedup](benchmarks/plots/mnist_read_speedup.png)

**Analysis:**
- GPU provides significant speedup for image preprocessing
- Label reading shows minimal benefit on GPU due to small data size
- Overall data loading time was improved by ~2-3x
- The normalization step benefits most from parallelization

## General Optimization Principles

Throughout this project, several key optimization principles were applied:

1. **Memory Access Patterns**
   - Coalesced memory access for global memory operations
   - Strategic use of shared memory for data reuse
   - Memory padding to avoid bank conflicts

2. **Parallelization Strategies**
   - Tile-based algorithms for matrix operations
   - Thread coarsening for higher arithmetic intensity
   - Dynamic thread allocation based on input size

3. **Kernel Fusion**
   - Combined operations where possible to reduce kernel launches
   - Balanced between specialized kernels and fused operations

4. **Vectorization**
   - Used float4 operations for higher memory throughput
   - Applied vectorized operations in compute-intensive sections

5. **Warp-Level Primitives**
   - Leveraged warp shuffle operations for efficient communication
   - Used warp-level voting functions for divergence management

## Performance Scaling Observations

Performance scaling varies significantly based on the operation and data size:

1. **Matrix Size Impact**
   - Operations with larger matrices benefit more from GPU acceleration
   - Small matrices (<1024 elements) often see limited benefit due to overhead
   - Optimal performance requires tuning block sizes based on matrix dimensions

2. **Operation Intensity**
   - Compute-bound operations (softmax, matrix multiply) show highest speedups
   - Memory-bound operations (ReLU) show more modest improvements
   - I/O-bound operations show least improvement unless preprocessing is complex

## Getting Started

### Prerequisites
- CUDA Toolkit 11.0+
- GCC/G++ compatible with your CUDA version
- Python 3.6+ with matplotlib (for benchmarking)

## Running the MNIST Implementation

### CPU Implementation

```bash
gcc -DRUN_MNIST_CPU -O3 -march=native -ffast-math -c mnist.c -o mnist.o && \
gcc -DRUN_MNIST_CPU -O3 -march=native -ffast-math mnist.o -o mnist -lm && \
./mnist
```

### CUDA Implementation

```bash
nvcc -O3 -arch=sm_86 -rdc=true -DRUN_MNIST_CUDA -c mnist.cu -o mnist_cuda.o && \
nvcc -O3 -arch=sm_86 -rdc=true -DRUN_MNIST_CUDA mnist_cuda.o -o mnist_cuda -lcudadevrt -lcurand -lcuda -lcublas -lcudnn && \
./mnist_cuda
```

**Note:** You may need to adjust the `-arch=sm_86` parameter based on your GPU architecture.


### Building the Project for benchmarking
```bash
# Compile the CPU implementation
gcc -O3 -march=native -ffast-math -c mnist.c -o mnist.o

# Compile the CUDA implementation
nvcc -O3 -arch=sm_86 -rdc=true -c mnist.cu -o mnist_cuda.o
```

### Running Benchmarks
```bash
# Build and run a specific benchmark
cd benchmarks
python plot_linear.py  # Or any other benchmark script
```
