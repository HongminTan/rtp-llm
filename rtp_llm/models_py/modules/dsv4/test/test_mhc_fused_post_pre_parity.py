"""P8 fused mHC post+pre numerical parity gate (double-green #2, R466.3 / r253 指令三①).

Compares the fused boundary op ``mhc_fused_post_pre`` against the unfused
production pair (``mhc_post`` -> ``mhc_pre``) on identical inputs at decode
shapes only (num_tokens <= 32, the FMA threshold). Three gates in the R326
mold: (1) the post residual, (2) the next unit's post/comb mixes,
(3) the next layer input.

The unfused pair's pre-norm GEMM rounds ``fn`` to tf32 while the fused FMA
kernel consumes fp32 directly, and the two paths reduce in different orders,
so the mix/layer gates carry tolerances sized for tf32-vs-fp32 plus bf16
quantization — structural errors (wrong index, swapped mix) land orders of
magnitude above them.
"""

import importlib
import unittest

import torch


def _functional():
    # tilelang_kernels first: module-level env-prep (libz3 preload + TVM tmpdir)
    # must run before any tilelang JIT import; importlib for the digit-leading
    # dotted path.
    from rtp_llm.models_py.modules.dsv4 import tilelang_kernels  # noqa: F401

    return importlib.import_module(
        "rtp_llm.models_py.3rdparty.tile_kernels.modeling.mhc.functional"
    )


_HC = 4
_PARAMS = dict(
    norm_eps=1e-6,
    pre_eps=1e-6,
    sinkhorn_eps=1e-6,
    sinkhorn_repeat=10,
    post_mult_value=2.0,
)


def _make_inputs(num_tokens: int, hidden: int, seed: int):
    # 4-D (S=1, T, hc, H): the unfused mhc_post asserts this rank; the fused
    # op views/flattens either rank itself.
    gen = torch.Generator(device="cpu").manual_seed(seed)
    mhc_mult3 = _HC * (2 + _HC)

    def rand(shape, dtype, scale):
        return (torch.randn(*shape, generator=gen) * scale).to(dtype).cuda()

    residual = rand((1, num_tokens, _HC, hidden), torch.bfloat16, 1.0)
    x = rand((1, num_tokens, hidden), torch.bfloat16, 1.0)
    # Production domains: post_mix = 2*sigmoid(...) in (0, 2); comb_mix after
    # sinkhorn ~ 1/hc each (positive, near-normalized).
    post_mix = (0.5 + 0.5 * torch.rand(1, num_tokens, _HC, 1, generator=gen)).cuda()
    comb_mix = (0.2 + 0.1 * torch.rand(1, num_tokens, _HC, _HC, generator=gen)).cuda()
    # fn scale keyed to its K dim (hc*hidden) so the mixes stay O(1) — and the
    # sigmoid gates out of saturation — at both hidden sizes.
    fn = rand((mhc_mult3, _HC * hidden), torch.float32, 4.0 / (_HC * hidden) ** 0.5)
    scale = torch.tensor([0.5, 0.8, 0.3]).cuda()
    base = rand((mhc_mult3,), torch.float32, 0.1)
    return residual, x, post_mix, comb_mix, fn, scale, base


def _run_unfused(functional, residual, x, post_mix, comb_mix, fn, scale, base):
    residual_cur = functional.mhc_post(x, residual, post_mix, comb_mix)
    layer_input, (post_mix_cur, comb_mix_cur) = functional.mhc_pre(
        residual_cur,
        fn,
        scale,
        base,
        norm_weight=None,
        mhc_mult=_HC,
        **_PARAMS,
    )
    return residual_cur, post_mix_cur, comb_mix_cur, layer_input


def _run_fused(functional, residual, x, post_mix, comb_mix, fn, scale, base):
    return functional.mhc_fused_post_pre(
        x,
        residual,
        post_mix,
        comb_mix,
        fn,
        scale,
        base,
        rms_eps=_PARAMS["norm_eps"],
        mhc_pre_eps=_PARAMS["pre_eps"],
        mhc_sinkhorn_eps=_PARAMS["sinkhorn_eps"],
        mhc_post_mult_value=_PARAMS["post_mult_value"],
        sinkhorn_repeat=_PARAMS["sinkhorn_repeat"],
    )


class FusedPostPreParityTest(unittest.TestCase):
    def _check_case(self, num_tokens, hidden, seed):
        functional = _functional()
        inputs = _make_inputs(num_tokens, hidden, seed)
        with torch.no_grad():
            ref = _run_unfused(functional, *inputs)
            fused = _run_fused(functional, *inputs)

        residual_ref, post_ref, comb_ref, layer_ref = ref
        residual_cur, post_cur, comb_cur, layer_cur = fused

        # Shapes: the fused op returns (T, hc, 1)/(T, hc, hc) views, mhc_pre
        # the same; compare the flat (T, hc) / (T, hc, hc) forms.
        self.assertEqual(residual_cur.shape, residual_ref.shape)
        self.assertEqual(layer_cur.shape, layer_ref.shape)
        post_cur_f = post_cur.reshape(num_tokens, _HC)
        post_ref_f = post_ref.reshape(num_tokens, _HC)
        comb_cur_f = comb_cur.reshape(num_tokens, _HC, _HC)
        comb_ref_f = comb_ref.reshape(num_tokens, _HC, _HC)

        # Gate 1: post residual (bf16). Same arithmetic in fp32 then a bf16
        # round-trip on both sides; only codegen-level FMA contraction can
        # differ, so a couple of bf16 ulps at the max magnitude is the bound.
        d1 = (residual_cur.float() - residual_ref.float()).abs().max().item()
        m1 = residual_ref.float().abs().max().item()
        self.assertLess(
            d1,
            max(2 * 0.0078 * m1, 1e-3),
            f"residual gate: max|d|={d1:.3e} vs max|ref|={m1:.3e}",
        )

        # Gate 2: next unit's post/comb mixes (fp32). Differences come from
        # tf32-vs-fp32 fn and reduction order in the GEMM partials.
        d2p = (post_cur_f - post_ref_f).abs().max().item()
        d2c = (comb_cur_f - comb_ref_f).abs().max().item()
        self.assertLess(d2p, 5e-3, f"post_mix gate: max|d|={d2p:.3e}")
        self.assertLess(d2c, 1e-2, f"comb_mix gate: max|d|={d2c:.3e}")

        # Gate 3: next layer input (bf16, O(1)); pre_mix sigmoid differences
        # plus one bf16 quantization step.
        d3 = (layer_cur.float() - layer_ref.float()).abs().max().item()
        m3 = layer_ref.float().abs().max().item()
        self.assertLess(
            d3,
            max(3 * 0.0078 * m3, 1e-2),
            f"layer_input gate: max|d|={d3:.3e} vs max|ref|={m3:.3e}",
        )

    def test_parity_hidden512(self):
        for num_tokens in (1, 4, 8, 32):
            self._check_case(num_tokens, 512, seed=1000 + num_tokens)

    def test_parity_hidden7168(self):
        # Production hidden size; exercises the non-divisible-by-n_thr loop
        # and the hidden>4096 split selection.
        for num_tokens in (1, 8, 32):
            self._check_case(num_tokens, 7168, seed=2000 + num_tokens)


if __name__ == "__main__":
    unittest.main()
