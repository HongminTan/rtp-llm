"""Resolve and cache FlashMLA for the existing MLA/DSv4 cache formats."""

import logging
from functools import cache
from importlib import import_module


# Cache successful resolution: a missing legacy wheel otherwise triggers an
# import-finder search on every attention call before falling back to flash_mla.
@cache
def get_legacy_flash_mla():
    try:
        return import_module("flash_mla_legacy")
    except ModuleNotFoundError as error:
        if error.name != "flash_mla_legacy":
            raise
        return import_module("flash_mla")


def flash_mla_available() -> bool:
    try:
        backend = get_legacy_flash_mla()
        getattr(backend, "flash_mla_sparse_fwd")
        getattr(backend, "flash_mla_with_kvcache")
        getattr(backend, "get_mla_metadata")
        return True
    except (ImportError, AttributeError, OSError, ValueError, RuntimeError) as error:
        logging.warning("[dsv4-fp8] FlashMLA unavailable: %s", error)
        return False


def __getattr__(name):
    if name.startswith("__") and name.endswith("__"):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(get_legacy_flash_mla(), name)
