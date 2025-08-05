import torch
import numba
from numba import cuda
import cupy as cp
import numpy as np

def check_gpu_availability():
    """Check if CUDA GPU is available and display device info"""
    print("=== GPU Device Information ===")
    # Check if CUDA is available
    if not cuda.is_available():
        print("CUDA is not available on this system")
        return False
    # Get device information
    device = cuda.get_current_device()
    print(f"Device Name: {device.name}")
    print(f"Compute Capability: {device.compute_capability}")
    # Get memory info using context
    try:
        ctx = cuda.current_context()
        free_mem, total_mem = ctx.get_memory_info()
        print(f"Total Memory: {total_mem / (1024 ** 3):.2f} GB")
        print(f"Free Memory: {free_mem / (1024 ** 3):.2f} GB")
    except Exception as e:
        print(f"Memory info not available: {e}")
    # Device attributes
    try:
        print(f"Max Threads per Block: {device.MAX_THREADS_PER_BLOCK}")
        print(f"Max Block Dimensions: {device.MAX_BLOCK_DIM_X} x {device.MAX_BLOCK_DIM_Y} x {device.MAX_BLOCK_DIM_Z}")
        print(f"Max Grid Dimensions: {device.MAX_GRID_DIM_X} x {device.MAX_GRID_DIM_Y} x {device.MAX_GRID_DIM_Z}")
        print(f"Warp Size: {device.WARP_SIZE}")
        print(f"Multiprocessors: {device.MULTIPROCESSOR_COUNT}")
    except AttributeError as e:
        print(f"Some device attributes not available: {e}")
        # Try alternative method to get basic info
        try:
            print(
                f"Max Threads per Block: {device.get_attribute(cuda.cudadrv.driver.device_attribute.MAX_THREADS_PER_BLOCK)}")
            print(f"Multiprocessors: {device.get_attribute(cuda.cudadrv.driver.device_attribute.MULTIPROCESSOR_COUNT)}")
            print(f"Warp Size: {device.get_attribute(cuda.cudadrv.driver.device_attribute.WARP_SIZE)}")
        except:
            print("Using basic device detection")
    return True

def get_device():
    """Get the appropriate device (GPU if available, else CPU)"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device

def get_numba_device():
    """Get the Numba CUDA device if available"""
    if cuda.is_available():
        device = cuda.get_current_device()
        print(f"Numba using GPU: {device.name}")
        return device
    else:
        print("Numba using CPU")
        return None

def test_gpu_computation():
    """Test GPU computation with CuPy"""
    if not cuda.is_available():
        print("CUDA not available, skipping GPU test")
        return
    print("\n=== Testing GPU Computation ===")
    # Create large arrays on GPU
    size = 10 ** 6  # 1 million elements
    a_gpu = cp.random.rand(size)  # GPU array
    b_gpu = cp.random.rand(size)  # GPU array
    # Perform GPU computation
    result_gpu = cp.sin(a_gpu) + cp.cos(b_gpu)
    # Move result back to CPU for inspection
    result_cpu = cp.asnumpy(result_gpu)
    print("First 5 results with GPU:", result_cpu[:5])
    print("GPU computation test completed successfully!")

def test_pytorch_gpu():
    """Test PyTorch GPU computation"""
    if not torch.cuda.is_available():
        print("PyTorch CUDA not available, skipping PyTorch GPU test")
        return
    print("\n=== Testing PyTorch GPU Computation ===")
    # Create large tensors on GPU
    size = 10 ** 6  # 1 million elements
    a_gpu = torch.randn(size, device='cuda')
    b_gpu = torch.randn(size, device='cuda')
    # Perform GPU computation
    result_gpu = torch.sin(a_gpu) + torch.cos(b_gpu)
    # Move result back to CPU for inspection
    result_cpu = result_gpu.cpu().numpy()
    print("First 5 results with PyTorch GPU:", result_cpu[:5])
    print("PyTorch GPU computation test completed successfully!")

if __name__ == "__main__":
    if check_gpu_availability():
        test_gpu_computation()
        test_pytorch_gpu()
        get_device()
        get_numba_device()