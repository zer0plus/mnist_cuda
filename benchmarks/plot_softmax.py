import os
import subprocess
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import Counter

def build_test_executable():
    """
    Build the test_softmax executable from the benchmarks directory
    """
    # Get paths
    current_dir = Path(__file__).parent
    tests_dir = current_dir.parent / "tests"
    test_file = tests_dir / "test_softmax.cu"
    mnist_cuda_obj = current_dir.parent / "mnist_cuda.o"
    mnist_obj = current_dir.parent / "mnist.o"
    output_file = tests_dir / "test_softmax"

    # Build command
    build_cmd = [
        "nvcc",
        "-O3",
        "-arch=sm_86",
        "-rdc=true",
        "-DRUN_SOFTMAX_TEST",
        "-o", str(output_file),
        str(test_file),
        str(mnist_cuda_obj),
        str(mnist_obj),
        "-lcudadevrt",
        "-lcurand",
        "-lcuda",
        "-lcublas",
        "-lcudnn"
    ]

    try:
        # Run build command from tests directory
        subprocess.run(build_cmd, cwd=tests_dir, check=True)
        print("Successfully built test_softmax executable")
    except subprocess.CalledProcessError as e:
        print(f"Failed to build test_softmax executable: {e}")
        exit(1)

def format_size(size, for_plot=False):
    """
    Format the actual allocated memory size in bytes to a human-readable string with KB or MB units
    Each element is a float (FP32, 4 bytes), so total memory is size * 4 bytes
    If for_plot is True, returns only the memory size string without byte count
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
            return f"{gb:.1f} GB"
        return f"{actual_bytes} bytes ({gb:.1f} GB)"

def format_xtick(size):
    """
    Format x-axis tick label showing both power of 2 and memory size
    """
    mem_size = format_size(size, for_plot=True)
    return f"2^{int(np.log2(size))}\n/{mem_size}"

def run_test(size):
    """
    Run the test_softmax executable with given size and parse results
    """
    tests_dir = Path(__file__).parent.parent / "tests"
    test_exe = tests_dir / "test_softmax"

    try:
        # Run the test
        result = subprocess.run(
            [str(test_exe), str(size)],
            cwd=tests_dir,
            capture_output=True,
            text=True,
            check=True
        )

        # Parse the output
        output = result.stdout
        results = {}
        
        # Look for specific metrics in the output
        for line in output.split('\n'):
            if 'CPU time:' in line:
                results['CPU_Time'] = float(line.split(':')[1].strip().split()[0])
            elif 'GPU time:' in line:
                results['GPU_Time'] = float(line.split(':')[1].strip().split()[0])
            elif 'Speedup:' in line:
                # Extract the speedup value, handling inf cases
                speedup_str = line.split(':')[1].strip().split('x')[0]
                if 'inf' in speedup_str.lower():
                    results['Speedup'] = float('inf')
                else:
                    results['Speedup'] = float(speedup_str)
            elif 'Softmax Test Passed' in line:
                results['Valid'] = 1
            elif 'Softmax Test Failed' in line:
                results['Valid'] = 0
            elif 'highest precision' in line:
                results['Precision'] = '1e-7'
            elif 'Resolved with' in line:
                precision = line.split('Resolved with ')[1].split(' precision')[0]
                results['Precision'] = precision

        # If no explicit Speedup was found but we have CPU and GPU times
        if 'Speedup' not in results and 'CPU_Time' in results and 'GPU_Time' in results:
            if results['GPU_Time'] > 0:
                results['Speedup'] = results['CPU_Time'] / results['GPU_Time']
            else:
                results['Speedup'] = float('inf')  # Avoid division by zero

        # Check if all required keys are present
        required_keys = ['CPU_Time', 'GPU_Time', 'Valid']
        
        missing_keys = [key for key in required_keys if key not in results]
        if missing_keys:
            print(f"Error: Missing required output values: {missing_keys}")
            print("Test output:")
            print(output)
            exit(1)

        # Add size to results
        results['InputSize'] = size
        
        return results

    except subprocess.CalledProcessError as e:
        print(f"Error running test with size {size}: {e}")
        print(f"Test output: {e.stdout}")
        print(f"Error output: {e.stderr}")
        exit(1)

def main():
    # Test sizes (powers of 2)
    sizes = [2**i for i in range(5, 26)]  # From 2^5 to 2^25
    
    # Build the executable first
    build_test_executable()

    # Lists to store results
    speedups = []
    cpu_times = []
    gpu_times = []
    valid_flags = []
    precision_levels = []
    sizes_tested = []

    # Run tests for each size
    for size in sizes:
        formatted_size = format_size(size)
        print(f"Running test for size {size} elements, using {formatted_size} of memory...")
        
        try:
            results = run_test(size)
            
            # Store results
            sizes_tested.append(size)
            speedups.append(results.get('Speedup', 0))
            cpu_times.append(results.get('CPU_Time', 0))
            gpu_times.append(results.get('GPU_Time', 0))
            valid_flags.append(results.get('Valid', 0))
            precision_levels.append(results.get('Precision', 'unknown'))
            
            # Report status for each test
            precision_info = f" (at {results.get('Precision', 'unknown')} precision)" if results.get('Valid', 0) == 1 and 'Precision' in results else ""
            if results.get('Valid', 0) == 1:
                print(f"✓ Test passed for size {size}{precision_info}")
            else:
                print(f"✗ Test failed for size {size}")
                
        except Exception as e:
            print(f"Error testing size {size}: {e}")
            # Add the size but mark as invalid
            sizes_tested.append(size)
            speedups.append(0)
            cpu_times.append(0)
            gpu_times.append(0)
            valid_flags.append(0)
            precision_levels.append('error')

    # Create plots directory if it doesn't exist
    plots_dir = Path(__file__).parent / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Define custom x-ticks for better visibility
    # Use powers of 5 for cleaner spacing (2^5, 2^10, 2^15, etc.)
    xticks = [2**i for i in range(5, 26, 5)]
    xtick_labels = [format_xtick(x) for x in xticks]

    # Plot 1: Softmax Speedup
    plt.figure(figsize=(12, 6))
    
    # Use a logarithmic scale for speedup if needed
    use_log_scale = any(s > 100 for s in speedups if s != float('inf'))
    
    # Filter out inf values for plotting
    plotting_sizes = []
    plotting_speedups = []
    for size, speedup, valid in zip(sizes_tested, speedups, valid_flags):
        if speedup != float('inf') and valid == 1:
            plotting_sizes.append(size)
            plotting_speedups.append(speedup)
    
    if plotting_sizes:
        plt.plot(plotting_sizes, plotting_speedups, 'b-o', label='Valid measurements')
    
    # Mark the points that had infinite speedup
    inf_sizes = [size for size, speedup, valid in zip(sizes_tested, speedups, valid_flags) 
                if speedup == float('inf') and valid == 1]
    if inf_sizes:
        max_finite_speedup = max([s for s in speedups if s != float('inf')], default=100)
        inf_y_value = max_finite_speedup * 1.1  # Place slightly above the max
        plt.plot(inf_sizes, [inf_y_value] * len(inf_sizes), 'r^', markersize=10, 
                 label='Infinite speedup')
    
    # Mark invalid tests
    invalid_sizes = [size for size, valid in zip(sizes_tested, valid_flags) if valid == 0]
    if invalid_sizes:
        plt.plot(invalid_sizes, [0] * len(invalid_sizes), 'rx', markersize=8, 
                 label='Invalid result')
    
    plt.xscale('log', base=2)
    if use_log_scale:
        plt.yscale('log')
        
    plt.xticks(xticks, xtick_labels, rotation=45, ha='right')
    plt.grid(True)
    plt.xlabel('Input Size (Elements/Allocated Memory)')
    plt.ylabel('Speedup Multiple (GPU vs CPU)' + (' [log scale]' if use_log_scale else ''))
    plt.title('Softmax Speedup: GPU vs CPU')
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / 'softmax_speedup.png')
    plt.close()

    # Plot 2: Softmax Times (CPU vs GPU)
    plt.figure(figsize=(12, 6))
    
    # Filter out zero times for log scale
    valid_indices = [i for i, v in enumerate(valid_flags) if v == 1]
    valid_sizes = [sizes_tested[i] for i in valid_indices]
    valid_cpu_times = [cpu_times[i] for i in valid_indices]
    valid_gpu_times = [gpu_times[i] for i in valid_indices]
    
    plt.plot(valid_sizes, valid_cpu_times, 'b-o', label='CPU Time')
    plt.plot(valid_sizes, valid_gpu_times, 'r-o', label='GPU Time')
    
    plt.xscale('log', base=2)
    plt.yscale('log')
    plt.xticks(xticks, xtick_labels, rotation=45, ha='right')
    plt.grid(True)
    plt.xlabel('Input Size (Elements/Allocated Memory)')
    plt.ylabel('Time (ms, log scale)')
    plt.title('Softmax Execution Time (CPU vs GPU)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / 'softmax_times.png')
    plt.close()

    # Print summary
    print("\nTest Summary:")
    print(f"Total tests: {len(sizes_tested)}")
    print(f"Passed: {sum(valid_flags)}")
    print(f"Failed: {len(valid_flags) - sum(valid_flags)}")
    
    # Print precision distribution
    if precision_levels and precision_levels[0] != 'unknown':
        precision_counts = Counter(precision_levels)
        print("\nPrecision distribution:")
        for level, count in sorted(precision_counts.items()):
            if level != 'unknown' and level != 'error':
                print(f"{level}: {count} tests")
    
    # Print speedup statistics
    valid_speedups = [s for s, v in zip(speedups, valid_flags) if v == 1 and s != float('inf')]
    if valid_speedups:
        print("\nSpeedup statistics:")
        print(f"Min: {min(valid_speedups):.2f}x")
        print(f"Max: {max(valid_speedups):.2f}x")
        print(f"Avg: {sum(valid_speedups)/len(valid_speedups):.2f}x")
    
    inf_count = sum(1 for s, v in zip(speedups, valid_flags) if s == float('inf') and v == 1)
    if inf_count:
        print(f"Infinite speedup: {inf_count} tests")
    
    print("\nAll plots have been generated in the plots directory")

if __name__ == "__main__":
    main()