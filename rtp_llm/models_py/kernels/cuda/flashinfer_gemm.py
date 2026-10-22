"""Shared access to FlashInfer's SM90 FP8 block-scale GEMM."""

# FlashInfer fp8_blockscale_gemm_sm90 uses swapAB for M < 32 and the
# normal kernel otherwise. Recheck this threshold when updating FlashInfer.
FLASHINFER_M_THRESHOLD = 32


def get_flashinfer_sm90_fp8_gemm():
    try:
        from flashinfer.gemm import fp8_blockscale_gemm_sm90

        return fp8_blockscale_gemm_sm90
    except Exception:
        return None
