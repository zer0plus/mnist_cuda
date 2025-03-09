import os
import subprocess
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def build_test_executable():
    """
    Build the test_linear executable from the benchmarks directory
    """
    # Get paths
    current_dir = Path(__file__).parent
    tests_dir = current_dir.parent / "tests"
    test_file = tests_dir / "test_linear.cu"
    mnist_cuda_obj = current_dir.parent / "mnist_cuda.o"
    mnist_obj = current_dir.parent / "mnist.o"
    output_file = tests_dir / "test_linear"

    # Build command
    build_cmd = [
        "nvcc",
        "-O3",
        "-arch=sm_86",
        "-rdc=true",
        "-DRUN_LINEAR_TEST",
        "-o", str(output_file),
        str(test_file),
        str(mnist_cuda_obj),
        str(mnist_obj),
        "-lcudadevrt",
        "-lcurand",
        "-lcuda",
        "-lcublas"
    ]

    try:
        # Run build command from tests directory
        subprocess.run(build_cmd, cwd=tests_dir, check=True)
        print("Successfully built test_linear executable")
    except subprocess.CalledProcessError as e:
        print(f"Failed to build test_linear executable: {e}")
        exit(1)

def format_size(size, for_plot=False):
    """
    Format memory size in bytes to a human-readable string
    """
    # Calculate actual memory usage (size * sizeof(float))
    actual_bytes = size * 4
    
    if actual_bytes < 1024:  # Less than 1 KB
        if for_plot:
            return f"{actual_bytes} B"
        return f"{actual_bytes} bytes"
    elif actual_bytes < 1024 * 1024:  # Less than 1 MB
        kb = actual_bytes / 1024
        if for_plot:
            return f"{kb:.1f} KB"
        return f"{actual_bytes} bytes ({kb:.1f} KB)"
    elif actual_bytes < 1024 * 1024 * 1024:  # Less than 1 GB
        mb = actual_bytes / (1024 * 1024)
        if for_plot:
            return f"{mb:.1f} MB"
        return f"{actual_bytes} bytes ({mb:.1f} MB)"
    else:
        gb = actual_bytes / (1024 * 1024 * 1024)
        if for_plot:
            return f"{gb:.2f} GB"
        return f"{actual_bytes} bytes ({gb:.2f} GB)"

def format_matrix_size(size, for_plot=False):
    """Format matrix size for plots"""
    if for_plot:
        return f"{size}×{size}"
    return f"{size}×{size} ({format_size(size*size, False)} matrix)"

def run_test(size):
    """
    Run the test_linear executable with equal input and output sizes
    """
    tests_dir = Path(__file__).parent.parent / "tests"
    test_exe = tests_dir / "test_linear"

    # Calculate memory requirements
    matrix_elements = size * size
    matrix_memory_gb = (matrix_elements * 4) / (1024**3)  # in GB
    
    # Set a timeout based on matrix size to prevent tests from hanging
    timeout = min(300, max(30, int(matrix_memory_gb * 60)))  # 30s minimum, 300s maximum
    
    try:
        # Run the test with timeout
        print(f"  Running test with timeout of {timeout}s...")
        result = subprocess.run(
            [str(test_exe), str(size), str(size)],
            cwd=tests_dir,
            capture_output=True,
            text=True,
            timeout=timeout
        )

        # Parse the output
        output = result.stdout
        results = {}
        parsing_results = False
        
        for line in output.split('\n'):
            if line.strip() == "LINEAR_RESULTS:":
                parsing_results = True
                continue
            elif line.strip() == "END_RESULTS":
                parsing_results = False
                continue
            
            if parsing_results and ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                
                # Skip empty values
                if not value:
                    continue
                    
                try:
                    # Try to convert to float if there's a decimal point,
                    # otherwise try to convert to int
                    if '.' in value:
                        results[key] = float(value)
                    else:
                        results[key] = int(value)
                except ValueError:
                    print(f"Warning: Could not parse value '{value}' for key '{key}'")
                    continue

        # If we didn't get valid results, the test probably crashed
        if not results:
            print(f"Error: No results parsed from test output for size={size}")
            print("Test output:")
            print(output)
            # Return default values indicating failure
            return {
                'Size': size,
                'MatrixSize': size*size,
                'CPU_Time': 0,
                'GPU_Time': 0,
                'Speedup': 0,
                'Valid': 0
            }

        # Add matrix size
        results['Size'] = size
        results['MatrixSize'] = size*size
        
        return results

    except subprocess.TimeoutExpired:
        print(f"Error: Test timed out after {timeout}s for size={size}")
        return {
            'Size': size,
            'MatrixSize': size*size,
            'CPU_Time': 0,
            'GPU_Time': 0,
            'Speedup': 0,
            'Valid': 0,
            'TimedOut': True
        }
    except subprocess.SubprocessError as e:
        print(f"Error running test with size={size}: {e}")
        if hasattr(e, 'stdout'):
            print(f"Test output: {e.stdout}")
        if hasattr(e, 'stderr'):
            print(f"Error output: {e.stderr}")
        
        # Return default values indicating failure
        return {
            'Size': size,
            'MatrixSize': size*size,
            'CPU_Time': 0,
            'GPU_Time': 0,
            'Speedup': 0,
            'Valid': 0
        }

def main():
    # Build the executable first
    build_test_executable()
    
    # Create plots directory if it doesn't exist
    plots_dir = Path(__file__).parent / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Test with equal dimensions
    # Start with 2^13 (8192) up to 2^25 (33,554,432)
    # Use powers of 2 with some steps in between for better distribution
    sizes = []
    for i in range(13, 26):
        sizes.append(2**i)
        if i < 25:  # Add intermediate points between powers of 2
            sizes.append(int(2**(i+0.5)))  # Geometric midpoint
    
    results = []
    
    print("\nRunning tests with equal input and output dimensions...")
    for size in sizes:
        # Calculate total matrix size for info
        total_elements = size * size
        total_memory = total_elements * 4  # 4 bytes per float
        memory_gb = total_memory / (1024**3)
        
        # Skip sizes that would use more than 16GB for weight matrix
        if memory_gb > 16:
            print(f"Skipping size={size}×{size} ({format_size(total_memory)}) - exceeds memory limit")
            continue
            
        print(f"Testing size={size}×{size} ({format_size(total_memory)})")
        result = run_test(size)
        results.append(result)
        
        # Safety check: if a mid-sized test fails, skip even larger sizes
        if result['Valid'] == 0 and memory_gb > 4:
            print(f"Test failed at {size}×{size}. Skipping larger sizes to avoid crashes.")
            break
    
    # Only include valid results for plotting
    valid_results = [r for r in results if r['Valid'] == 1]
    
    if not valid_results:
        print("No valid results to plot.")
        return
    
    # Plot 1: Speedup vs Matrix Size
    plt.figure(figsize=(12, 6))
    
    # Extract data for plotting
    matrix_sizes = [r['MatrixSize'] for r in valid_results]
    speedups = [r['Speedup'] for r in valid_results]
    
    plt.plot(matrix_sizes, speedups, 'b-o', linewidth=2, markersize=8)
    
    plt.xscale('log')
    plt.grid(True, which="both", ls="--", alpha=0.7)
    plt.xlabel('Matrix Size (elements)')
    plt.ylabel('Speedup (CPU Time / GPU Time)')
    plt.title('Linear Layer Speedup vs Matrix Size')
    
    # Add dimension labels to x-axis
    for i, r in enumerate(valid_results):
        plt.annotate(f"{r['Size']}×{r['Size']}", 
                    xy=(r['MatrixSize'], speedups[i]),
                    xytext=(0, 10),
                    textcoords='offset points',
                    ha='center')
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'linear_equal_dims_speedup.png')
    plt.close()
    
    # Plot 2: Execution Times (CPU vs GPU)
    plt.figure(figsize=(12, 6))
    
    cpu_times = [r['CPU_Time'] for r in valid_results]
    gpu_times = [r['GPU_Time'] for r in valid_results]
    
    plt.plot(matrix_sizes, cpu_times, 'b-o', label='CPU Time', linewidth=2, markersize=8)
    plt.plot(matrix_sizes, gpu_times, 'r-o', label='GPU Time', linewidth=2, markersize=8)
    
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, which="both", ls="--", alpha=0.7)
    plt.xlabel('Matrix Size (elements)')
    plt.ylabel('Time (ms, log scale)')
    plt.title('Linear Layer Execution Time vs Matrix Size')
    plt.legend()
    
    # Add dimension labels to x-axis
    for i, r in enumerate(valid_results):
        plt.annotate(f"{r['Size']}×{r['Size']}", 
                    xy=(r['MatrixSize'], max(cpu_times[i], gpu_times[i])),
                    xytext=(0, 10),
                    textcoords='offset points',
                    ha='center')
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'linear_equal_dims_times.png')
    plt.close()
    
    # Plot 3: Performance trends
    plt.figure(figsize=(12, 6))
    
    sizes_array = [r['Size'] for r in valid_results]
    flops = [2 * s * s for s in sizes_array]  # Multiply-add operations
    
    cpu_gflops = [flops[i] / (cpu_times[i] * 1e6) for i in range(len(valid_results))]  # GFLOPS = flops/(time in sec * 1e9)
    gpu_gflops = [flops[i] / (gpu_times[i] * 1e6) for i in range(len(valid_results))]
    
    plt.plot(sizes_array, cpu_gflops, 'b-o', label='CPU GFLOPS', linewidth=2, markersize=8)
    plt.plot(sizes_array, gpu_gflops, 'r-o', label='GPU GFLOPS', linewidth=2, markersize=8)
    
    plt.xscale('log', base=2)
    plt.grid(True, which="both", ls="--", alpha=0.7)
    plt.xlabel('Matrix Dimension')
    plt.ylabel('GFLOPS (higher is better)')
    plt.title('Linear Layer Performance in GFLOPS')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'linear_equal_dims_gflops.png')
    plt.close()
    
    # Print summary statistics
    print("\nBenchmark Summary:")
    print(f"Total tests: {len(results)}")
    print(f"Valid tests: {len(valid_results)}")
    print(f"Invalid tests: {len(results) - len(valid_results)}")
    
    if valid_results:
        print("\nPerformance Statistics:")
        print(f"Average speedup: {sum(speedups)/len(speedups):.2f}x")
        print(f"Maximum speedup: {max(speedups):.2f}x (at {valid_results[speedups.index(max(speedups))]['Size']}×{valid_results[speedups.index(max(speedups))]['Size']})")
        
        print("\nMatrix size with best CPU performance (GFLOPS):")
        max_cpu_idx = cpu_gflops.index(max(cpu_gflops))
        print(f"  {valid_results[max_cpu_idx]['Size']}×{valid_results[max_cpu_idx]['Size']} achieved {cpu_gflops[max_cpu_idx]:.2f} GFLOPS")
        
        print("\nMatrix size with best GPU performance (GFLOPS):")
        max_gpu_idx = gpu_gflops.index(max(gpu_gflops))
        print(f"  {valid_results[max_gpu_idx]['Size']}×{valid_results[max_gpu_idx]['Size']} achieved {gpu_gflops[max_gpu_idx]:.2f} GFLOPS")
    
    print("\nAll plots have been generated in the plots directory")

if __name__ == "__main__":
    main()