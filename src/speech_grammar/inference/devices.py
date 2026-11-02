"""Device and numeric-precision selection for CTranslate2 / PyTorch runtimes."""

from __future__ import annotations

import ctypes
import importlib.util
import os
import sys
from functools import lru_cache
from pathlib import Path

from speech_grammar.config import Device

# Libraries CTranslate2 4.x loads lazily on first GPU use (CUDA 12 + cuDNN 9).
_CUDA_LIBRARIES = {
    "win32": ("cublas64_12.dll", "cudnn64_9.dll"),
    "linux": ("libcublas.so.12", "libcudnn.so.9"),
}

# Pip wheels that bundle those DLLs outside the Windows search path: PyTorch's
# CUDA builds (torch/lib) and nvidia-cublas-cu12 / nvidia-cudnn-cu12 (nvidia/*/bin).
_PIP_CUDA_DIRS = (("torch", "lib"), ("nvidia", "cublas", "bin"), ("nvidia", "cudnn", "bin"))


def cuda_device_count() -> int:
    """Number of CUDA devices visible to CTranslate2 (0 when unavailable)."""
    try:
        import ctranslate2

        return int(ctranslate2.get_cuda_device_count())
    except Exception:
        return 0


def _can_load(names: tuple[str, ...]) -> bool:
    loader = getattr(ctypes, "WinDLL", ctypes.CDLL) if sys.platform == "win32" else ctypes.CDLL
    try:
        for name in names:
            # winmode=0: the standard Windows search order (incl. PATH), as CTranslate2 uses.
            loader(name, winmode=0)
    except OSError:
        return False
    return True


def _pip_cuda_library_dirs() -> list[Path]:
    """Folders of pip-installed CUDA DLLs, located without importing the packages."""
    dirs = []
    for package, *subdirs in _PIP_CUDA_DIRS:
        try:
            spec = importlib.util.find_spec(package)
        except (ImportError, ValueError):
            continue
        roots = (spec.submodule_search_locations or []) if spec is not None else []
        dirs += [d for root in roots if (d := Path(root, *subdirs)).is_dir()]
    return dirs


def _expose_pip_cuda_libraries(names: tuple[str, ...]) -> bool:
    """Prepend pip CUDA folders to ``PATH`` (Windows); True if any was added.

    CTranslate2 loads cuBLAS lazily by name with the standard search order, which
    ignores ``os.add_dll_directory`` but honours ``PATH``.
    """
    added = [d for d in _pip_cuda_library_dirs() if any((d / n).is_file() for n in names)]
    if added:
        os.environ["PATH"] = os.pathsep.join([*map(str, added), os.environ.get("PATH", "")])
    return bool(added)


@lru_cache(maxsize=1)
def cuda_libraries_available() -> bool:
    """True when cuBLAS and cuDNN can be loaded by this process.

    A visible GPU driver is not enough: without these libraries CTranslate2 fails
    (or, observed with faster-whisper on Windows, *hangs*) at the first decode,
    which no try/except can recover from. Checking with ``ctypes`` is instant.
    On Windows, copies shipped in pip wheels (e.g. PyTorch CUDA) are used when
    no system-wide CUDA toolkit is installed.
    """
    names = _CUDA_LIBRARIES.get(sys.platform)
    if names is None:
        return False
    if _can_load(names):
        return True
    return sys.platform == "win32" and _expose_pip_cuda_libraries(names) and _can_load(names)


def resolve_device(device: Device) -> str:
    """Map the configured device to ``"cpu"`` or ``"cuda"``.

    ``auto`` picks CUDA only when a GPU is visible *and* its libraries load.
    """
    if device is Device.CUDA:
        cuda_libraries_available()  # exposes pip-provided libraries; errors surface at load
        return "cuda"
    if device is Device.CPU:
        return "cpu"
    return "cuda" if cuda_device_count() > 0 and cuda_libraries_available() else "cpu"


def resolve_t5_compute_type(requested: str, device: str) -> str:
    """Pick the CTranslate2 compute type for the T5 corrector.

    On GPU, int8 weights with float32 activations: original T5 checkpoints are
    known to overflow in float16, and on the v2 model bfloat16 scored lower
    (JFLEG BLEU 0.867 vs 0.872, see README) without being faster.
    """
    if requested != "auto":
        return requested
    if device == "cpu":
        return "int8"
    try:
        import ctranslate2

        supported = set(ctranslate2.get_supported_compute_types("cuda"))
    except Exception:
        supported = set()
    for candidate in ("int8_float32", "float32"):
        if candidate in supported:
            return candidate
    return "int8_float32"


def resolve_whisper_compute_type(requested: str, device: str) -> str:
    """Pick the CTranslate2 compute type for Whisper (float16 is safe for Whisper)."""
    if requested != "auto":
        return requested
    return "float16" if device == "cuda" else "int8"
