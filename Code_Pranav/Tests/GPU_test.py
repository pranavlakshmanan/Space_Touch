#!/usr/bin/env python3
"""
GPU Availability Check Script
Checks if PyTorch can access GPU in the current conda environment
"""

import sys

def check_gpu():
    print("=" * 60)
    print("GPU AVAILABILITY CHECK")
    print("=" * 60)
    
    # Check if PyTorch is installed
    try:
        import torch
        print(f"✓ PyTorch is installed (version: {torch.__version__})")
    except ImportError:
        print("✗ PyTorch is not installed")
        print("\nInstall PyTorch with: pip install torch torchvision")
        sys.exit(1)
    
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"Python version: {sys.version.split()[0]}")
    
    # Check CUDA availability
    print("\n" + "-" * 60)
    print("CUDA CHECK")
    print("-" * 60)
    
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    if cuda_available:
        print(f"✓ GPU will be used for computations")
        print(f"\nCUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        # List all available GPUs
        print("\nAvailable GPU(s):")
        for i in range(torch.cuda.device_count()):
            print(f"  [{i}] {torch.cuda.get_device_name(i)}")
            print(f"      Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
        
        # Check current device
        print(f"\nCurrent CUDA device: {torch.cuda.current_device()}")
        print(f"Current device name: {torch.cuda.get_device_name()}")
        
        # Test GPU with a simple operation
        print("\n" + "-" * 60)
        print("GPU TEST")
        print("-" * 60)
        try:
            x = torch.rand(1000, 1000).cuda()
            y = torch.rand(1000, 1000).cuda()
            z = x @ y
            print("✓ Successfully performed matrix multiplication on GPU")
            print(f"  Result tensor device: {z.device}")
        except Exception as e:
            print(f"✗ GPU test failed: {e}")
    else:
        print(f"✗ GPU will NOT be used - running on CPU only")
        print("\nPossible reasons:")
        print("  1. No NVIDIA GPU detected")
        print("  2. CUDA drivers not installed")
        print("  3. PyTorch CPU version installed (not CUDA version)")
        print("  4. CUDA version mismatch")
        
        # Check if it's a CPU-only build
        if '+cpu' in torch.__version__:
            print("\n⚠ You have PyTorch CPU version installed!")
            print("  Install CUDA version with:")
            print("  pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130")
    
    print("\n" + "=" * 60)
    
    return cuda_available

if __name__ == "__main__":
    gpu_available = check_gpu()
    sys.exit(0 if gpu_available else 1)