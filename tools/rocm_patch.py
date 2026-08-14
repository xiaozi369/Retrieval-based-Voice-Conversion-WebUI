"""ROCm / RDNA4 (gfx1200/gfx1201) compatibility patch — single self-contained file.

Why this file exists
--------------------
pip-distributed torch ROCm wheels on Windows have two defects that crash
AMD RDNA4 GPUs (RX 9070 / 9070 XT, gfx1200/gfx1201):

1. MIOpen's hiprtc JIT compilation (used for conv / RNN kernels) cannot find
   C++ standard library headers (e.g. ``<type_traits>``) on Windows, so any
   operator that triggers MIOpen fails with ``miopenStatusUnknownError`` /
   ``HIPRTC_ERROR_COMPILATION``, even in plain eager inference.

2. torch's flash / memory-efficient SDPA backends crash with
   ``hipErrorInvalidValue`` on RDNA4.

This module applies both workarounds WITHOUT modifying any project source
file. It is idempotent and a no-op on every other platform / arch
(NVIDIA CUDA, RDNA3, CPU), so it is safe to call unconditionally.

Usage (pick one)
----------------
- As a standalone diagnostic / one-shot fix:

      python tools/rocm_patch.py

- Imported from application code (e.g. your entry point or launcher):

      from tools.rocm_patch import apply_rocm_patch
      apply_rocm_patch()          # idempotent

Both paths return a status dict describing what was applied.
"""

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Standard library headers MIOpen's hiprtc compilation may use; libcudacxx
# provides same-named implementations under cuda/std. Generate a wrapper per
# header, only for entries that actually exist under cuda/std.
_STD_HEADERS = (
    "type_traits",
    "utility",
    "cstdint",
    "cstddef",
    "cstdlib",
    "cstring",
    "cmath",
    "limits",
    "memory",
    "algorithm",
    "new",
    "exception",
    "iterator",
    "functional",
    "array",
    "tuple",
    "variant",
    "optional",
    "initializer_list",
    "typeinfo",
    "string_view",
    "span",
    "numeric",
    "ratio",
    "bit",
)

_HEADER_GUARD_PREFIX = "_RVC_STD_COMPAT_"


def is_rdna4():
    """True only on ROCm RDNA4 (gfx1200/gfx1201) devices."""
    import torch

    if not getattr(torch.version, "hip", None) or not torch.cuda.is_available():
        return False
    try:
        return torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).gcnArchName.startswith("gfx12")
    except Exception:
        return False


def _windows_include_path(path):
    """Convert a path into the Windows-style form hiprtc's clang accepts."""
    path = str(Path(path))
    if len(path) >= 2 and path[1] == ":":
        return path.replace("\\", "/")
    return path


def _find_rocm_devel_include():
    """Locate the rocm_sdk devel include dir inside site-packages."""
    import torch

    site_packages = Path(torch.__file__).resolve().parent.parent
    candidates = (
        site_packages / "_rocm_sdk_devel" / "include",
        site_packages / "rocm_sdk_devel" / "include",
    )
    for cand in candidates:
        if (cand / "cuda" / "std").is_dir():
            return cand
    return None


def _write_compat_header(include_dir, name):
    """Write one std compatibility wrapper header; skip if unchanged."""
    header = include_dir / name
    body = (
        f"#ifndef {_HEADER_GUARD_PREFIX}{name.upper()}_\n"
        f"#define {_HEADER_GUARD_PREFIX}{name.upper()}_\n"
        f"#include <cuda/std/{name}>\n"
        "namespace std { using namespace cuda::std; }\n"
        "#endif\n"
    )
    if header.exists() and header.read_text(encoding="utf-8") == body:
        return False
    header.write_text(body, encoding="utf-8")
    return True


def _apply_miopen_header_fix():
    """Fix MIOpen hiprtc JIT: write std wrapper headers, expose include dir."""
    status = {"miopen_header_fix": False, "header_fixes": [], "reason": ""}

    include_dir = _find_rocm_devel_include()
    if include_dir is None:
        status["reason"] = (
            "rocm_sdk devel include dir not found; check that the pip torch "
            "ROCm wheel ships _rocm_sdk_devel"
        )
        logger.warning(status["reason"])
        return status

    cuda_std = include_dir / "cuda" / "std"
    written = []
    for name in _STD_HEADERS:
        if (cuda_std / name).is_file():
            if _write_compat_header(include_dir, name):
                written.append(name)
    status["header_fixes"] = written
    if written:
        logger.info(
            "Wrote %d std compat wrapper headers: %s",
            len(written),
            ", ".join(written),
        )

    win_include = _windows_include_path(include_dir)
    existing = os.environ.get("CPLUS_INCLUDE_PATH", "")
    parts = [p for p in existing.split(os.pathsep) if p] if existing else []
    if win_include not in parts:
        parts.append(win_include)
    os.environ["CPLUS_INCLUDE_PATH"] = os.pathsep.join(parts)
    status["miopen_header_fix"] = True
    status["cplus_include_path"] = os.environ["CPLUS_INCLUDE_PATH"]
    return status


def _apply_sdpa_disable():
    """Disable flash / memory-efficient SDPA backends (crash on RDNA4)."""
    import torch

    status = {"sdpa_disabled": False, "reason": ""}
    try:
        if torch.cuda.is_available() and hasattr(torch.backends, "cuda"):
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            status["sdpa_disabled"] = True
            logger.info("Disabled flash / memory-efficient SDPA (RDNA4 compat)")
    except Exception as exc:
        status["reason"] = str(exc)
        logger.exception("Failed to disable SDPA backends")
    return status


def apply_rocm_patch():
    """Apply all ROCm / RDNA4 compatibility fixes. Idempotent; no-op elsewhere.

    Returns a status dict describing what was applied.
    """
    if not is_rdna4():
        logger.info("Not RDNA4; ROCm patch skipped (no-op)")
        return {
            "rdna4": False,
            "miopen_header_fix": False,
            "sdpa_disabled": False,
            "reason": "not an RDNA4 device",
        }

    result = {"rdna4": True}
    result.update(_apply_miopen_header_fix())
    result.update(_apply_sdpa_disable())
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    st = apply_rocm_patch()
    print("\n=== ROCm / RDNA4 patch diagnostics ===")
    for key, value in st.items():
        print(f"{key}: {value}")
