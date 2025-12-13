import os
import ctypes
from pathlib import Path
import logging

import onnxruntime as ort

LOG = logging.getLogger(__name__)

def get_cuda_providers():
    """
    Ensure CUDA dependencies are loaded and return the providers list.
    """
    ensure_cuda_libs()
    return ["CUDAExecutionProvider"]


def _load_nvidia_cuda_libs():
    """
    Preload CUDA/cuDNN libs from NVIDIA pip wheels and expose them via LD_LIBRARY_PATH.
    Raises if wheels are missing or no libraries can be loaded.
    """
    try:
        import nvidia  # type: ignore
    except ImportError as exc:
        raise RuntimeError("NVIDIA CUDA wheels not installed; cannot preload CUDA libs") from exc

    nvidia_root = Path(nvidia.__file__).resolve().parent
    candidate_dirs = [
        nvidia_root / "cudnn" / "lib",
        nvidia_root / "cublas" / "lib",
        nvidia_root / "cufft" / "lib",
        nvidia_root / "curand" / "lib",
        nvidia_root / "cuda_nvrtc" / "lib",
        nvidia_root / "cuda_runtime" / "lib",
        nvidia_root / "nvjitlink" / "lib",
    ]

    existing_dirs = [d for d in candidate_dirs if d.exists()]
    if not existing_dirs:
        raise RuntimeError("No CUDA library directories found in NVIDIA wheels")

    existing = os.environ.get("LD_LIBRARY_PATH", "")
    existing_parts = existing.split(os.pathsep) if existing else []
    new_parts = [str(d) for d in existing_dirs if str(d) not in existing_parts]
    if new_parts:
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(new_parts + existing_parts)

    loaded_any = False
    for lib_dir in existing_dirs:
        for so_path in sorted(lib_dir.glob("*.so*")):
            try:
                ctypes.CDLL(str(so_path), mode=ctypes.RTLD_GLOBAL)
                loaded_any = True
            except OSError:
                continue

    if not loaded_any:
        raise RuntimeError("Failed to preload CUDA libraries from NVIDIA wheels")


def ensure_cuda_libs():
    """
    First try onnxruntime's preload_dlls helper; if it fails, fall back to manual LD_LIBRARY_PATH + dlopen.
    If both fail, raise to fail fast.
    """
    preload_exc = None
    if hasattr(ort, "preload_dlls"):
        try:
            ort.preload_dlls(directory="")
            return
        except Exception as exc:  # noqa: BLE001
            preload_exc = exc

    try:
        _load_nvidia_cuda_libs()
    except Exception as exc:  # noqa: BLE001
        if preload_exc:
            raise RuntimeError(
                f"Failed to preload CUDA libraries via onnxruntime preload_dlls and LD_LIBRARY_PATH bootstrap: {exc}"
            ) from preload_exc
        raise
