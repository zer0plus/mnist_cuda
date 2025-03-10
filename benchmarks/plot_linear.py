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
    # Calculate memory usage (size * sizeof(float))
    memory_bytes = size * 4
    
    if memory_bytes < 1024:  # Less than 1 KB
        if for_plot:
            return f"{memory_bytes} B"
        return f"{memory_bytes} bytes"
    elif memory_bytes < 1024 * 1024:  # Less than 1 MB
        kb = memory_bytes / 1024
        if for_plot:
            return f"{kb:.1f} KB"
        return f"{memory_bytes} bytes ({kb:.1f} KB)"
    elif memory_bytes < 1024 * 1024 * 1024:  # Less than 1 GB
        mb = memory_bytes / (1024 * 1024)
        if for_plot:
            return f"{mb:.1f} MB"
        return f"{memory_bytes} bytes ({mb:.1f} MB)"
    else:
        gb = memory_bytes / (1024 * 1024 * 1024)
        if for_plot:
            return f"{gb:.2f} GB"
        return f"{memory_bytes} bytes ({gb:.2f} GB)"

def format_matrix_size(in_size, out_size, for_plot=False):
    """Format matrix dimensions for plots"""
    if for_plot:
        return f"{in_size}×{out_size}"
    matrix_elements = in_size * out_size
    return f"{in_size}×{out_size} ({format_size(matrix_elements, False)})"

def format_xtick(in_size, out_size):
    """
    Format x-axis tick label showing dimensions and memory size
    """
    mem_size = format_size(in_size * out_size, for_plot=True)
    return f"{in_size}×{out_size}\n{mem_size}"

def run_test(in_size, out_size):
    """
    Run the test_linear executable with the given dimensions
    """
    tests_dir = Path(__file__).parent.parent / "tests"
    test_exe = tests_dir / "test_linear"
    
    # Calculate memory requirements
    matrix_elements = in_size * out_size
    matrix_memory_gb = (matrix_elements * 4) / (1024**3)  # Convert to GB
    
    # Set a timeout based on matrix size to prevent tests from hanging
    timeout = min(300, max(30, int(matrix_memory_gb * 60)))  # 30s minimum, 300s maximum
    
    try:
        # Run the test with timeout
        print(f"Running test with dimensions {in_size}×{out_size} ({format_size(matrix_elements)})...")
        result = subprocess.run(
            [str(test_exe), str(in_size), str(out_size)],
            cwd=tests_dir,
            capture_output=True,
            text=True,
            timeout=timeout
        )

        # Parse the output
        output = result.stdout
        results = {}
        inside_results = False
        
        for line in output.split('\n'):
            if line.strip() == "LINEAR_RESULTS:":
                inside_results = True
                continue
            elif line.strip() == "END_RESULTS":
                inside_results = False
                continue
                
            if inside_results and ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                
                if not value:
                    continue
                    
                try:
                    if '.' in value:
                        results[key] = float(value)
                    else:
                        results[key] = int(value)
                except ValueError:
                    # Skip values that can't be parsed
                    continue
                    
        # Add dimensions to results
        results['InputSize'] = in_size
        results['OutputSize'] = out_size
        results['MatrixElements'] = in_size * out_size
        
        # Verify that essential values were parsed
        required_keys = ['CPU_Time', 'GPU_Time', 'Speedup', 'Valid']
        missing = [k for k in required_keys if k not in results]
        
        if missing:
            print(f"Error: Missing required values in output: {missing}")
            return None
            
        return results
        
    except subprocess.TimeoutExpired:
        print(f"Error: Test timed out after {timeout}s for dimensions {in_size}×{out_size}")
        return None
    except subprocess.SubprocessError as e:
        print(f"Error running test: {e}")
        return None

def main():
    # Build the executable
    build_test_executable()
    
    # Create plots directory if it doesn't exist
    plots_dir = Path(__file__).parent / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Define test cases
    test_cases = [
        # Small (MNIST-like)
        (784, 256),
        (256, 10),
        
        # Medium
        (1024, 512),
        (2048, 1024),
        (4096, 1024),
        
        # Large
        (8192, 2048),
        (16384, 4096),
        (32768, 8192),
        
        # Very large
        (65536, 16384),
        (131072, 32768)
    ]
    
    results = []
    
    # Run tests
    for in_size, out_size in test_cases:
        result = run_test(in_size, out_size)
        if result and result['Valid'] == 1:
            results.append(result)
        
        # Stop if we encounter a large failed test to avoid wasting time
        if (not result or result['Valid'] == 0) and in_size * out_size > 4 * 1024 * 1024:
            print(f"Stopping tests after failure with large matrix ({in_size}×{out_size})")
            break
    
    if not results:
        print("No valid results to plot.")
        return
    
    # Sort results by matrix size
    results.sort(key=lambda r: r['MatrixElements'])
    
    # Extract data for plotting
    matrix_sizes = [r['MatrixElements'] for r in results]
    speedups = [r['Speedup'] for r in results]
    cpu_times = [r['CPU_Time'] for r in results]
    gpu_times = [r['GPU_Time'] for r in results]
    
    # Create x-tick labels
    xtick_labels = [format_xtick(r['InputSize'], r['OutputSize']) for r in results]
    
    # Plot 1: Speedup vs Matrix Size
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(results)), speedups, 'r-o', linewidth=2, markersize=8)
    plt.xticks(range(len(results)), xtick_labels, rotation=45, ha='right')
    plt.axhline(y=1, color='k', linestyle='--', alpha=0.5)
    plt.grid(True, alpha=0.3)
    plt.xlabel('Matrix Dimensions (input×output / memory size)')
    plt.ylabel('Speedup (CPU Time / GPU Time)')
    plt.title('cuBLAS Linear Layer Speedup vs. CPU')
    plt.tight_layout()
    plt.savefig(plots_dir / 'linear_speedup.png')
    plt.close()
    
    # Plot 2: Execution Times (log scale)
    plt.figure(figsize=(12, 6))
    plt.semilogy(range(len(results)), cpu_times, 'g-o', label='CPU', linewidth=2, markersize=8)
    plt.semilogy(range(len(results)), gpu_times, 'r-o', label='cuBLAS', linewidth=2, markersize=8)
    plt.xticks(range(len(results)), xtick_labels, rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    plt.xlabel('Matrix Dimensions (input×output / memory size)')
    plt.ylabel('Time (ms, log scale)')
    plt.title('Linear Layer Execution Time')
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / 'linear_times.png')
    plt.close()
    
    # Print summary table
    print("\nSummary of Results:")
    print("-" * 80)
    print(f"{'Dimensions':<20} {'CPU Time (ms)':<15} {'GPU Time (ms)':<15} {'Speedup':<10} {'Valid':<8}")
    print("-" * 80)
    
    for r in results:
        dims = f"{r['InputSize']}×{r['OutputSize']}"
        print(f"{dims:<20} {r['CPU_Time']:<15.2f} {r['GPU_Time']:<15.2f} {r['Speedup']:<10.2f} {'✓' if r['Valid'] else '✗'}")
    
    print("-" * 80)
    
    # Print some statistics
    if results:
        max_speedup = max(speedups)
        max_speedup_idx = speedups.index(max_speedup)
        max_speedup_dims = (results[max_speedup_idx]['InputSize'], results[max_speedup_idx]['OutputSize'])
        
        print(f"Maximum speedup: {max_speedup:.2f}x with dimensions {max_speedup_dims[0]}×{max_speedup_dims[1]}")
        
        # Find the crossover point (where speedup > 1)
        for i, s in enumerate(speedups):
            if s > 1:
                crossover_dims = (results[i]['InputSize'], results[i]['OutputSize'])
                crossover_elements = results[i]['MatrixElements']
                print(f"GPU becomes faster than CPU at matrix size: {crossover_dims[0]}×{crossover_dims[1]} ({format_size(crossover_elements)})")
                break
    
    print("\nAll plots have been generated in the plots directory")

if __name__ == "__main__":
    main()