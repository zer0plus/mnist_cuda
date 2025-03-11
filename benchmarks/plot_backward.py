import os
import subprocess
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def build_test_executable():
    """
    Build the test_backward executable from the benchmarks directory
    """
    # Get paths
    current_dir = Path(__file__).parent
    tests_dir = current_dir.parent / "tests"
    test_file = tests_dir / "test_backward.cu"
    mnist_cuda_obj = current_dir.parent / "mnist_cuda.o"
    mnist_obj = current_dir.parent / "mnist.o"
    output_file = tests_dir / "test_backward"

    # Build command
    build_cmd = [
        "nvcc",
        "-O3",
        "-arch=sm_86",
        "-rdc=true",
        "-DRUN_BACKWARD_TEST",
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
        print("Successfully built test_backward executable")
    except subprocess.CalledProcessError as e:
        print(f"Failed to build test_backward executable: {e}")
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

def format_xtick(in_size, out_size):
    """
    Format x-axis tick label showing dimensions and memory size
    """
    mem_size = format_size(in_size * out_size, for_plot=True)
    return f"{in_size}×{out_size}\n{mem_size}"

def run_test(in_size, out_size):
    """
    Run the test_backward executable with the given dimensions
    """
    tests_dir = Path(__file__).parent.parent / "tests"
    test_exe = tests_dir / "test_backward"
    
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
            if line.strip() == "BACKWARD_RESULTS:":
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
                    if '.' in value or 'e' in value.lower():  # Handle scientific notation
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
        required_keys = [
            'CPU_Time', 'CuBLAS_Time', 'Custom_Time',
            'CuBLAS_Speedup', 'Custom_Speedup', 'Custom_vs_CuBLAS',
            'Valid'
        ]
        missing = [k for k in required_keys if k not in results]
        
        if missing:
            print(f"Error: Missing required values in output: {missing}")
            print(f"Output: {output}")
            return None
            
        return results
        
    except subprocess.TimeoutExpired:
        print(f"Error: Test timed out after {timeout}s for dimensions {in_size}×{out_size}")
        return None
    except subprocess.SubprocessError as e:
        print(f"Error running test: {e}")
        print(f"Command output: {e.output if hasattr(e, 'output') else 'No output'}")
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
        if result:
            # We'll include results with weights and biases valid (Valid=1)
            if result['Valid'] == 1:
                results.append(result)
                print(f"✓ Test passed for dimensions {in_size}×{out_size}")
                print(f"  CPU: {result['CPU_Time']:.4f} ms")
                print(f"  cuBLAS: {result['CuBLAS_Time']:.4f} ms ({result['CuBLAS_Speedup']:.2f}x speedup vs CPU)")
                print(f"  Custom: {result['Custom_Time']:.4f} ms ({result['Custom_Speedup']:.2f}x speedup vs CPU, {result['Custom_vs_CuBLAS']:.2f}x vs cuBLAS)")
            else:
                print(f"✗ Test failed for dimensions {in_size}×{out_size}")
                if 'CuBLAS_Valid' in result:
                    print(f"  cuBLAS valid: {'Yes' if result['CuBLAS_Valid'] == 1 else 'No'}")
                if 'Custom_Valid' in result:
                    print(f"  Custom valid: {'Yes' if result['Custom_Valid'] == 1 else 'No'}")
            
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
    cublas_speedups = [r['CuBLAS_Speedup'] for r in results]
    custom_speedups = [r['Custom_Speedup'] for r in results]
    custom_vs_cublas = [r['Custom_vs_CuBLAS'] for r in results]
    
    cpu_times = [r['CPU_Time'] for r in results]
    cublas_times = [r['CuBLAS_Time'] for r in results]
    custom_times = [r['Custom_Time'] for r in results]
    
    # Create x-tick labels
    xtick_labels = [format_xtick(r['InputSize'], r['OutputSize']) for r in results]
    
    # Plot 1: Speedup vs Matrix Size (CPU comparison)
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(results)), cublas_speedups, 'r-o', linewidth=2, markersize=8, label='cuBLAS vs CPU')
    plt.plot(range(len(results)), custom_speedups, 'g-o', linewidth=2, markersize=8, label='Custom vs CPU')
    plt.xticks(range(len(results)), xtick_labels, rotation=45, ha='right')
    plt.axhline(y=1, color='k', linestyle='--', alpha=0.5)
    plt.grid(True, alpha=0.3)
    plt.xlabel('Matrix Dimensions (input×output / memory size)')
    plt.ylabel('Speedup vs CPU')
    plt.title('GPU Backward Pass Speedup vs. CPU')
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / 'backward_speedup_vs_cpu.png')
    plt.close()
    
    # Plot 2: Custom vs cuBLAS
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(results)), custom_vs_cublas, 'b-o', linewidth=2, markersize=8)
    plt.xticks(range(len(results)), xtick_labels, rotation=45, ha='right')
    plt.axhline(y=1, color='k', linestyle='--', alpha=0.5)
    plt.grid(True, alpha=0.3)
    plt.xlabel('Matrix Dimensions (input×output / memory size)')
    plt.ylabel('Speedup Ratio (cuBLAS Time / Custom Time)')
    plt.title('Custom CUDA vs. cuBLAS Performance (>1 means Custom is faster)')
    plt.tight_layout()
    plt.savefig(plots_dir / 'backward_custom_vs_cublas.png')
    plt.close()
    
    # Plot 3: Execution Times (log scale)
    plt.figure(figsize=(12, 6))
    plt.semilogy(range(len(results)), cpu_times, 'k-o', label='CPU', linewidth=2, markersize=8)
    plt.semilogy(range(len(results)), cublas_times, 'r-o', label='cuBLAS', linewidth=2, markersize=8)
    plt.semilogy(range(len(results)), custom_times, 'g-o', label='Custom', linewidth=2, markersize=8)
    plt.xticks(range(len(results)), xtick_labels, rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    plt.xlabel('Matrix Dimensions (input×output / memory size)')
    plt.ylabel('Time (ms, log scale)')
    plt.title('Backward Pass Execution Time')
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / 'backward_times.png')
    plt.close()
    
    # Print summary table
    print("\nSummary of Results:")
    print("-" * 110)
    header = f"{'Dimensions':<20} {'CPU (ms)':<10} {'cuBLAS (ms)':<12} {'Custom (ms)':<12} {'cuBLAS Speedup':<15} {'Custom Speedup':<15} {'Custom vs cuBLAS':<15}"
    print(header)
    print("-" * 110)
    
    for r in results:
        dims = f"{r['InputSize']}×{r['OutputSize']}"
        row = f"{dims:<20} {r['CPU_Time']:<10.2f} {r['CuBLAS_Time']:<12.2f} {r['Custom_Time']:<12.2f} {r['CuBLAS_Speedup']:<15.2f} {r['Custom_Speedup']:<15.2f} {r['Custom_vs_CuBLAS']:<15.2f}"
        print(row)
    
    print("-" * 110)
    
    # Print some statistics
    if results:
        # cuBLAS stats
        max_cublas_speedup = max(cublas_speedups)
        max_cublas_idx = cublas_speedups.index(max_cublas_speedup)
        max_cublas_dims = (results[max_cublas_idx]['InputSize'], results[max_cublas_idx]['OutputSize'])
        
        # Custom stats
        max_custom_speedup = max(custom_speedups)
        max_custom_idx = custom_speedups.index(max_custom_speedup)
        max_custom_dims = (results[max_custom_idx]['InputSize'], results[max_custom_idx]['OutputSize'])
        
        # Custom vs cuBLAS stats
        max_custom_vs_cublas = max(custom_vs_cublas)
        min_custom_vs_cublas = min(custom_vs_cublas)
        avg_custom_vs_cublas = sum(custom_vs_cublas) / len(custom_vs_cublas)
        
        print("\nPerformance Statistics:")
        print(f"Maximum cuBLAS speedup vs CPU: {max_cublas_speedup:.2f}x with dimensions {max_cublas_dims[0]}×{max_cublas_dims[1]}")
        print(f"Maximum Custom speedup vs CPU: {max_custom_speedup:.2f}x with dimensions {max_custom_dims[0]}×{max_custom_dims[1]}")
        
        print(f"\nCustom vs cuBLAS Performance:")
        print(f"  Average: {avg_custom_vs_cublas:.2f}x ({'' if avg_custom_vs_cublas > 1 else 'Not '} faster on average)")
        print(f"  Best case: {max_custom_vs_cublas:.2f}x")
        print(f"  Worst case: {min_custom_vs_cublas:.2f}x")
        
        # Find the crossover points
        for i, s in enumerate(cublas_speedups):
            if s > 1:
                cublas_crossover_dims = (results[i]['InputSize'], results[i]['OutputSize'])
                cublas_crossover_elements = results[i]['MatrixElements']
                print(f"\ncuBLAS becomes faster than CPU at matrix size: {cublas_crossover_dims[0]}×{cublas_crossover_dims[1]} ({format_size(cublas_crossover_elements)})")
                break
                
        for i, s in enumerate(custom_speedups):
            if s > 1:
                custom_crossover_dims = (results[i]['InputSize'], results[i]['OutputSize'])
                custom_crossover_elements = results[i]['MatrixElements']
                print(f"Custom becomes faster than CPU at matrix size: {custom_crossover_dims[0]}×{custom_crossover_dims[1]} ({format_size(custom_crossover_elements)})")
                break
        
        # Find where custom becomes faster than cuBLAS (if it does)
        custom_beats_cublas = any(cvs > 1 for cvs in custom_vs_cublas)
        if custom_beats_cublas:
            for i, cvs in enumerate(custom_vs_cublas):
                if cvs > 1:
                    custom_beats_cublas_dims = (results[i]['InputSize'], results[i]['OutputSize'])
                    custom_beats_cublas_elements = results[i]['MatrixElements']
                    print(f"Custom becomes faster than cuBLAS at matrix size: {custom_beats_cublas_dims[0]}×{custom_beats_cublas_dims[1]} ({format_size(custom_beats_cublas_elements)})")
                    break
        else:
            print("Custom implementation does not exceed cuBLAS performance in any tested configuration.")
        
        # Note about gradient errors but without plotting them
        print("\nNote: Input gradient differences are collected but not plotted. These differences are expected")
        print("      due to different computation methods and floating-point precision between implementations.")
    
    print("\nPlots have been generated in the plots directory:")
    print("1. backward_speedup_vs_cpu.png - Comparing both GPU implementations to CPU")
    print("2. backward_custom_vs_cublas.png - Direct comparison of custom CUDA vs cuBLAS")
    print("3. backward_times.png - Execution times for all three implementations")

if __name__ == "__main__":
    main()