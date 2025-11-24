import subprocess
import sys

def check_nvidia_gpu():
    try:
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            print("NVIDIA GPU detected!")
            print(result.stdout)
            return True
        else:
            return False
    except FileNotFoundError:
        return False

def check_rocm_gpu():
    try:
        result = subprocess.run(['rocminfo'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            print("AMD GPU detected!")
            print(result.stdout.splitlines()[0])  # print summary line
            return True
        else:
            return False
    except FileNotFoundError:
        return False

def suggest_pytorch_install():
    print("\nSuggested PyTorch installation command:")
    if check_nvidia_gpu():
        print("For NVIDIA GPU (with CUDA), run:")
        print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    elif check_rocm_gpu():
        print("For AMD GPU (with ROCm), run:")
        print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.8")
    else:
        print("No GPU detected. Installing CPU-only PyTorch:")
        print("  pip install torch torchvision torchaudio")

if __name__ == "__main__":
    suggest_pytorch_install()
