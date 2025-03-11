import os
import subprocess
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def build_test_executable():
    """
    Build the test_read executable from the benchmarks directory
    """
    # Get paths
    current_dir = Path(__file__).parent
    tests_dir = current_dir.parent / "tests"
    test_file = tests_dir / "test_read.cu"
    mnist_cuda_obj = current_dir.parent / "mnist_cuda.o"
    mnist_obj = current_dir.parent / "mnist.o"
    output_file = tests_dir / "test_read"

    # Build command
    build_cmd = [
        "nvcc",
        "-O3",
        "-arch=sm_86",
        "-rdc=true",
        "-DRUN_READ_TEST",
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
        print("Successfully built test_read executable")
    except subprocess.CalledProcessError as e:
        print(f"Failed to build test_read executable: {e}")
        exit(1)

def run_test():
    """
    Run the test_read executable and parse results
    """
    tests_dir = Path(__file__).parent.parent / "tests"
    test_exe = tests_dir / "test_read"

    try:
        # Run the test
        result = subprocess.run(
            [str(test_exe)],
            cwd=tests_dir,
            capture_output=True,
            text=True
        )

        # Print output for debugging if there's an error
        if result.returncode != 0:
            print(f"Test output: {result.stdout}")
            if result.stderr:
                print(f"Error output: {result.stderr}")
            print(f"Error: Test process returned non-zero exit code: {result.returncode}")
        
        # Parse the output
        output = result.stdout
        results = {}
        inside_results = False
        
        for line in output.split('\n'):
            if line.strip() == "READ_RESULTS:":
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
                    
        # Check if required keys are present
        required_keys = [
            'CPU_Images_Read_Time', 'GPU_Images_Read_Time', 
            'CPU_Labels_Read_Time', 'GPU_Labels_Read_Time',
            'CPU_Images_Normalize_Time', 'CPU_Images_Total_Time',
            'Images_Speedup', 'Labels_Speedup', 'Total_Speedup',
            'Images_Valid', 'Labels_Valid', 'Overall_Valid'
        ]
        
        missing_keys = [key for key in required_keys if key not in results]
        if missing_keys:
            print(f"Error: Missing required output values: {missing_keys}")
            print("This might be due to the program exiting early.")
            exit(1)

        return results

    except subprocess.SubprocessError as e:
        print(f"Error running test: {e}")
        if hasattr(e, 'stdout'):
            print(f"Test output: {e.stdout}")
        if hasattr(e, 'stderr'):
            print(f"Error output: {e.stderr}")
        exit(1)

def main():
    # Create plots directory if it doesn't exist
    plots_dir = Path(__file__).parent / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Build the executable
    build_test_executable()
    
    # Run the test and get results
    print("Running MNIST reading benchmark...")
    results = run_test()
    
    if results.get('Overall_Valid', 0) != 1:
        print("Warning: Test results were not valid! Plots may not be meaningful.")
    
    # Extract timing data for plotting
    read_operations = ['Images', 'Labels', 'Total']
    cpu_times = [
        results['CPU_Images_Total_Time'],  # Include normalization for fair comparison
        results['CPU_Labels_Read_Time'],
        results['CPU_Total_Time']
    ]
    gpu_times = [
        results['GPU_Images_Read_Time'],
        results['GPU_Labels_Read_Time'],
        results['GPU_Total_Time']
    ]
    speedups = [
        results['Images_Speedup'],
        results['Labels_Speedup'],
        results['Total_Speedup']
    ]
    
    # Plot 1: CPU vs GPU Time Comparison
    plt.figure(figsize=(10, 6))
    x = np.arange(len(read_operations))
    width = 0.35
    
    plt.bar(x - width/2, cpu_times, width, label='CPU')
    plt.bar(x + width/2, gpu_times, width, label='GPU')
    
    plt.xlabel('Operation')
    plt.ylabel('Time (ms)')
    plt.title('MNIST Read Operations: CPU vs GPU Time')
    plt.xticks(x, read_operations)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # Add time values above each bar
    for i, v in enumerate(cpu_times):
        plt.text(i - width/2, v + 1, f'{v:.1f}', ha='center')
    for i, v in enumerate(gpu_times):
        plt.text(i + width/2, v + 1, f'{v:.1f}', ha='center')
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'mnist_read_times.png')
    plt.close()
    
    # Plot 2: Speedup Comparison
    plt.figure(figsize=(10, 6))
    plt.bar(read_operations, speedups, color='green')
    plt.axhline(y=1, color='r', linestyle='-', alpha=0.7)
    
    plt.xlabel('Operation')
    plt.ylabel('Speedup (CPU Time / GPU Time)')
    plt.title('MNIST Read Operations: GPU Speedup vs CPU')
    plt.grid(axis='y', alpha=0.3)
    
    # Add speedup values above each bar
    for i, v in enumerate(speedups):
        plt.text(i, v + 0.1, f'{v:.2f}x', ha='center')
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'mnist_read_speedup.png')
    plt.close()
    
    # Print summary
    print("\nMNIST Reading Benchmark Results:")
    print("-" * 60)
    print(f"Dataset Size: {results.get('Images_Count', 'N/A')} images, {results.get('Labels_Count', 'N/A')} labels")
    print(f"Image Dimensions: {int(np.sqrt(results.get('Image_Size', 784)))}x{int(np.sqrt(results.get('Image_Size', 784)))}")
    print("-" * 60)
    print("CPU vs GPU Timing (milliseconds):")
    print(f"{'Operation':<15} {'CPU Time':<15} {'GPU Time':<15} {'Speedup':<10}")
    print("-" * 60)
    print(f"{'Images':<15} {results['CPU_Images_Total_Time']:<15.2f} {results['GPU_Images_Read_Time']:<15.2f} {results['Images_Speedup']:<10.2f}x")
    print(f"{'Labels':<15} {results['CPU_Labels_Read_Time']:<15.2f} {results['GPU_Labels_Read_Time']:<15.2f} {results['Labels_Speedup']:<10.2f}x")
    print(f"{'Total':<15} {results['CPU_Total_Time']:<15.2f} {results['GPU_Total_Time']:<15.2f} {results['Total_Speedup']:<10.2f}x")
    print("-" * 60)
    
    # Still keeping the CPU breakdown text output for reference
    print("CPU Image Processing Breakdown:")
    print(f"- Reading:       {results['CPU_Images_Read_Time']:.2f} ms")
    print(f"- Normalization: {results['CPU_Images_Normalize_Time']:.2f} ms")
    print(f"- Total:         {results['CPU_Images_Total_Time']:.2f} ms")
    print("-" * 60)
    
    print(f"Validation Result: {'PASSED' if results.get('Overall_Valid', 0) == 1 else 'FAILED'}")
    print("\nPlots generated in the plots directory:")
    print("1. mnist_read_times.png - Time comparison between CPU and GPU")
    print("2. mnist_read_speedup.png - GPU speedup relative to CPU")

if __name__ == "__main__":
    main()