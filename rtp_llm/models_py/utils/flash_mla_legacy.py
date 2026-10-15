"""Resolve the FlashMLA runtime for existing MLA/DSv4 cache formats.

CUDA13 ARM ships separate current V4.1 and legacy wheels. Other platform locks
retain their original FlashMLA distribution, so use that only when the isolated
legacy distribution is absent. V4.1 never imports this module.

Resolution is cached after the first success. The attention hot paths import
this module inside their functions, so every call used to re-resolve — and on
locks without the ``flash_mla_legacy`` wheel (e.g. cu12_9) the failed import
walks the import finders each time, which py-spy sampled as the dominant host
cost of eager prefill (R102/R103). Tests that substitute a fake module under
either name (and any caller needing fresh resolution) can set
``DSV4_FLASH_MLA_LEGACY_IMPORT_PER_CALL=1`` to restore the historical
per-access resolution semantics.
"""

import os
from importlib import import_module

_resolved = None


def _resolve():
    try:
        return import_module("flash_mla_legacy")
    except ModuleNotFoundError as error:
        if error.name != "flash_mla_legacy":
            raise
        return import_module("flash_mla")


def get_legacy_flash_mla():
    global _resolved
    if os.environ.get("DSV4_FLASH_MLA_LEGACY_IMPORT_PER_CALL", "0") != "1":
        if _resolved is None:
            _resolved = _resolve()
        return _resolved
    return _resolve()


def __getattr__(name):
    return getattr(get_legacy_flash_mla(), name)
