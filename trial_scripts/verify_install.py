# verify_install.py
import sys
import importlib

packages = [
    "torch",
    "transformers",
    "datasets",
    "peft",
    "bitsandbytes",
    "jaxtyping",
    "einops",
    "vllm",
    "wandb",
    "flash_attn",
]

print("Python version:", sys.version)
print("PYTHONPATH:", sys.path)
print("\nChecking packages...\n")

for pkg in packages:
    try:
        module = importlib.import_module(pkg)
        print(f"[OK] {pkg} -> version: {getattr(module, '__version__', 'unknown')}")
    except ModuleNotFoundError:
        print(f"[MISSING] {pkg}")
    except Exception as e:
        print(f"[ERROR] {pkg} -> {e}")