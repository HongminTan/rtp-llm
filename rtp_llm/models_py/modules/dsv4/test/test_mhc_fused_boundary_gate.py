"""P8 fused mHC post+pre boundary compile gate (R465.5 ①, r252 指令一②).

Config-class test-only carrier (R355 jit-dryrun-02 precedent): imports the
ported kernel and triggers the tilelang JIT construction under the pinned
production tilelang, zero kernel launch. Green = compile gate passed;
failure counts against the config 3-strike budget (r252 指令一③).
"""

import unittest


class FusedPostPreCompileGateTest(unittest.TestCase):
    def test_fma_kernel_constructs(self):
        # importlib for the digit-leading dotted path (same reason as
        # mhc_tilelang._import_tk).
        import importlib

        mod = importlib.import_module(
            "rtp_llm.models_py.3rdparty.tile_kernels.mhc.fused_post_pre_fma_kernel"
        )
        kernel = mod._mhc_fused_post_pre_fma(
            4, 512, 24, n_thr=256, tile_mix_outputs=2, split_k=4
        )
        self.assertIsNotNone(kernel)

    def test_boundary_op_imports(self):
        import importlib

        functional = importlib.import_module(
            "rtp_llm.models_py.3rdparty.tile_kernels.modeling.mhc.functional"
        )
        self.assertTrue(hasattr(functional, "mhc_fused_post_pre"))

    def test_wrapper_module_imports(self):
        from rtp_llm.models_py.modules.dsv4.hc import mhc_tilelang

        self.assertTrue(hasattr(mhc_tilelang, "tk_mhc_fused_post_pre"))
        self.assertTrue(hasattr(mhc_tilelang, "tk_mhc_fused_post_pre_enabled"))


if __name__ == "__main__":
    unittest.main()
