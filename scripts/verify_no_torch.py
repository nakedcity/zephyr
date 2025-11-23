import sys
import importlib.util

def check_torch():
    print("Checking for PyTorch dependency...")
    # Check if torch is installed/importable
    torch_spec = importlib.util.find_spec("torch")
    if torch_spec is not None:
        print("WARNING: 'torch' is installed in the environment!")
        try:
            import torch
            print(f"Torch version: {torch.__version__}")
            sys.exit(1)
        except ImportError:
            print("But it cannot be imported.")
    else:
        print("SUCCESS: 'torch' is NOT installed.")

    # Check loaded modules just in case
    if 'torch' in sys.modules:
        print("FAILURE: 'torch' is loaded in sys.modules!")
        sys.exit(1)

    print("Verification passed: No PyTorch dependency found.")

if __name__ == "__main__":
    check_torch()
