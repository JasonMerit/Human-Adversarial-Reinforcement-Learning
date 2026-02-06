"""
CUDA availability checker for DTU HPC job submission.
Run this from your bash job script to verify GPU access.
"""

import sys
import torch
import os

print("=" * 60)
print("DTU HPC CUDA Check - Run from job script")
print("=" * 60)
print(f"Python: {sys.executable}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
print()

# Core CUDA checks
if torch.cuda.is_available():
    print("✅ CUDA is AVAILABLE")
    print(f"   CUDA Version: {torch.version.cuda}")
    print(f"   GPU Count: {torch.cuda.device_count()}")
    
    gpu_name = torch.cuda.get_device_name(0)
    print(f"   GPU 0: {gpu_name}")
    
    # Memory check
    mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"   GPU Memory: {mem_gb:.1f} GB")
    
    # Quick test tensor
    x = torch.randn(1000, 1000).cuda()
    print("   ✅ GPU tensor allocation test PASSED")
    
else:
    print("❌ CUDA UNAVAILABLE")
    print("   Check: CUDA toolkit, PyTorch CUDA build, GPU allocation")
    sys.exit(1)

print("\n" + "=" * 60)
print("Ready to train on GPU!")
print("=" * 60)
