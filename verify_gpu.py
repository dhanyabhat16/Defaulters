"""
GPU Verification Script
Run this first to verify PyTorch sees your GPU
"""
import torch

print("=" * 60)
print("GPU VERIFICATION")
print("=" * 60)

print("\nPyTorch Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("\n✓ CUDA is available")
    print(f"✓ Device Name: {torch.cuda.get_device_name(0)}")
    print(f"✓ CUDA Version: {torch.version.cuda}")
    print(f"✓ GPU Count: {torch.cuda.device_count()}")
else:
    print("\n✗ CUDA is NOT available - GPU support not detected")
    print("GPU operations will run on CPU instead")

print("\n" + "=" * 60)
