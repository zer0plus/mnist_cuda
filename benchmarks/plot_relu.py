import os
import subprocess
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def build_test_executable():
    """
    Build the test_relu executable from the benchmarks directory
    """
    # Get paths
    current_dir = Path(__file__).parent
    tests_dir = current_dir.parent / "tests"
    test_file = tests_dir / "test_relu.cu"
    mnist_cuda_obj = current_dir.parent / "mnist_cuda.o"
    mnist_obj = current_dir.parent / "mnist.o"
    output_file = tests_dir / "test_relu"

    # Build command
    build_cmd = [
        "nvcc",
        "-O3",
        "-arch=sm_86",
        "-rdc=true",
        "-DRUN_RELU_TEST",
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
        print("Successfully built test_relu executable")
    except subprocess.CalledProcessError as e:
        print(f"Failed to build test_relu executable: {e}")
        exit(1)


# def format_size(size):
#     """
#     Format the actual allocated memory size in bytes to a human-readable string with KB or MB units
#     Each element is a float (FP32, 4 bytes), so total memory is size * 4 bytes
#     """
#     # Calculate actual memory usage (size * sizeof(float))
#     actual_bytes = size * 4
    
#     if actual_bytes < 1024 * 1024:  # Less than 1 MB
#         kb = actual_bytes / 1024
#         return f"{actual_bytes} ({kb:.1f} KB)"
#     else:
#         mb = actual_bytes / (1024 * 1024)
#         return f"{actual_bytes} ({mb:.1f} MB)"

def run_test(size):
    """
    Run the test_relu executable with given size and parse results
    """
    tests_dir = Path(__file__).parent.parent / "tests"
    test_exe = tests_dir / "test_relu"

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
        for line in output.split('\n'):
            if ':' in line:
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
                except ValueError as e:
                    print(f"Warning: Could not parse value '{value}' for key '{key}': {e}")
                    continue

        # Check if all required keys are present
        required_keys = [
            'Forward_Valid', 'Derivative_Valid',
            'Forward_Speedup', 'Derivative_Speedup',
            'Forward_CPU_Time', 'Forward_GPU_Time',
            'Derivative_CPU_Time', 'Derivative_GPU_Time'
        ]
        
        missing_keys = [key for key in required_keys if key not in results]
        if missing_keys:
            print(f"Error: Missing required output values: {missing_keys}")
            print("Test output:")
            print(output)
            exit(1)

        return results

    except subprocess.CalledProcessError as e:
        print(f"Error running test with size {size}: {e}")
        print(f"Test output: {e.output}")
        exit(1)

def format_size(size, for_plot=False):
    """
    Format the actual allocated memory size in bytes to a human-readable string with KB or MB units
    Each element is a float (FP32, 4 bytes), so total memory is size * 4 bytes
    If for_plot is True, returns only the memory size string without byte count
    """
    # Calculate actual memory usage (size * sizeof(float))
    actual_bytes = size * 4
    
    if actual_bytes < 1024 * 1024:  # Less than 1 MB
        kb = actual_bytes / 1024
        if for_plot:
            return f"{kb:.1f} KB"
        return f"{actual_bytes} bytes ({kb:.1f} KB)"
    else:
        mb = actual_bytes / (1024 * 1024)
        if for_plot:
            return f"{mb:.1f} MB"
        return f"{actual_bytes} bytes ({mb:.1f} MB)"

def format_xtick(size):
    """
    Format x-axis tick label showing both number of elements and allocated memory
    """
    mem_size = format_size(size, for_plot=True)
    return f"2^{int(np.log2(size))}\n/{mem_size}"


def main():
    # Test sizes (powers of 2)
    sizes = [2**i for i in range(9, 27)]  

    # Build the executable first
    build_test_executable()

    # Lists to store results
    forward_speedups = []
    derivative_speedups = []
    forward_cpu_times = []
    forward_gpu_times = []
    derivative_cpu_times = []
    derivative_gpu_times = []

    # Run tests for each size
    for size in sizes:
        formatted_size = format_size(size)
        print(f"Running test for INPUT_SIZE of {size} elements, using {formatted_size} of memory...")
        results = run_test(size)

        # Validate results
        if results['Forward_Valid'] != 1 or results['Derivative_Valid'] != 1:
            print(f"Validation failed for size {size}")
            print(f"Forward_Valid: {results['Forward_Valid']}")
            print(f"Derivative_Valid: {results['Derivative_Valid']}")
            print("Terminating script due to invalid results")
            exit(1)

        # Store results
        forward_speedups.append(results['Forward_Speedup'])
        derivative_speedups.append(results['Derivative_Speedup'])
        forward_cpu_times.append(results['Forward_CPU_Time'])
        forward_gpu_times.append(results['Forward_GPU_Time'])
        derivative_cpu_times.append(results['Derivative_CPU_Time'])
        derivative_gpu_times.append(results['Derivative_GPU_Time'])

# Create plots directory if it doesn't exist
    plots_dir = Path(__file__).parent / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Define custom x-ticks for better visibility
    xticks = [2**i for i in range(9, 27, 3)]  # Every 3rd power of 2, adjusted for new range
    xtick_labels = [format_xtick(x) for x in xticks]

    # Plot 1: Forward ReLU Speedup
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, forward_speedups, 'b-o', label='Forward ReLU Speedup')
    plt.xscale('log', base=2)
    plt.xticks(xticks, xtick_labels, rotation=45, ha='right')
    plt.grid(True)
    plt.xlabel('Input Size (Elements/Allocated Memory)')
    plt.ylabel('Speedup Multiple (GPU vs CPU)')
    plt.title('Forward ReLU Speedup: GPU vs CPU')
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / 'forward_relu_speedup.png')
    plt.close()

    # Plot 2: Derivative ReLU Speedup
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, derivative_speedups, 'r-o', label='Derivative ReLU Speedup')
    plt.xscale('log', base=2)
    plt.xticks(xticks, xtick_labels, rotation=45, ha='right')
    plt.grid(True)
    plt.xlabel('Input Size (Elements/Allocated Memory)')
    plt.ylabel('Speedup Multiple (GPU vs CPU)')
    plt.title('Derivative ReLU Speedup: GPU vs CPU')
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / 'derivative_relu_speedup.png')
    plt.close()

    # Plot 3: Forward ReLU times
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, forward_cpu_times, 'b-o', label='CPU Time')
    plt.plot(sizes, forward_gpu_times, 'r-o', label='GPU Time')
    plt.xscale('log', base=2)
    plt.xticks(xticks, xtick_labels, rotation=45, ha='right')
    plt.grid(True)
    plt.xlabel('Input Size (Elements/Allocated Memory)')
    plt.ylabel('Time (ms)')
    plt.title('Forward ReLU Execution Time')
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / 'forward_relu_times.png')
    plt.close()

    # Plot 4: Derivative ReLU times
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, derivative_cpu_times, 'b-o', label='CPU Time')
    plt.plot(sizes, derivative_gpu_times, 'r-o', label='GPU Time')
    plt.xscale('log', base=2)
    plt.xticks(xticks, xtick_labels, rotation=45, ha='right')
    plt.grid(True)
    plt.xlabel('Input Size (Elements/Allocated Memory)')
    plt.ylabel('Time (ms)')
    plt.title('Derivative ReLU Execution Time')
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / 'derivative_relu_times.png')
    plt.close()

    print("All plots have been generated in the plots directory")

if __name__ == "__main__":
    main()