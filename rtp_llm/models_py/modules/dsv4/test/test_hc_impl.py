import importlib
import os
import unittest
from contextlib import contextmanager
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv4.hc import build_hc_head, build_hc_unit
from rtp_llm.models_py.modules.dsv4.hc.fallback_impl import (
    FallbackHCHead,
    FallbackHCUnit,
)
from rtp_llm.models_py.modules.dsv4.hc.tilelang_impl import (
    TileLangHCHead,
    TileLangHCUnit,
)


@contextmanager
def _env(name: str, value: str | None):
    old = os.environ.get(name)
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = old


def _weights(hc: int, dim: int, device: str = "cpu"):
    torch.manual_seed(0)
    mix_hc = (2 + hc) * hc
    fn = torch.randn(mix_hc, hc * dim, device=device, dtype=torch.float32) * 0.02
    base = torch.zeros(mix_hc, device=device, dtype=torch.float32)
    scale = torch.ones(3, device=device, dtype=torch.float32)
    return fn, base, scale


class TestHCImpl(unittest.TestCase):
    def test_factory_fallback_cpu_shapes(self) -> None:
        hc, dim = 4, 16
        fn, base, scale = _weights(hc, dim)
        with _env("DSV4_HC_IMPL", "fallback"):
            unit = build_hc_unit(
                fn,
                base,
                scale,
                dim=dim,
                hc_mult=hc,
                hc_sinkhorn_iters=3,
                norm_eps=1e-6,
                hc_eps=1e-6,
            )
            head = build_hc_head(
                fn[:hc],
                base[:hc],
                scale[:1],
                dim=dim,
                hc_mult=hc,
                norm_eps=1e-6,
                hc_eps=1e-6,
            )
        self.assertIsInstance(unit, FallbackHCUnit)
        self.assertIsInstance(head, FallbackHCHead)
        x = torch.randn(2, 5, hc, dim, dtype=torch.bfloat16)
        y, post, comb = unit.pre(x)
        self.assertEqual(tuple(y.shape), (2, 5, dim))
        self.assertEqual(tuple(post.shape), (2, 5, hc, 1))
        self.assertEqual(tuple(comb.shape), (2, 5, hc, hc))
        torch.testing.assert_close(
            comb.sum(dim=-1),
            torch.ones_like(comb.sum(dim=-1)),
            atol=5e-3,
            rtol=5e-3,
        )
        torch.testing.assert_close(
            comb.sum(dim=-2),
            torch.ones_like(comb.sum(dim=-2)),
            atol=5e-3,
            rtol=5e-3,
        )
        out = unit.post(y, x, post, comb)
        self.assertEqual(tuple(out.shape), tuple(x.shape))
        reduced = head.head(x)
        self.assertEqual(tuple(reduced.shape), (2, 5, dim))

    def test_factory_default_tilelang_fails_fast_on_cpu(self) -> None:
        hc, dim = 4, 16
        fn, base, scale = _weights(hc, dim)
        with _env("DSV4_HC_IMPL", None):
            unit = build_hc_unit(
                fn,
                base,
                scale,
                dim=dim,
                hc_mult=hc,
                hc_sinkhorn_iters=3,
                norm_eps=1e-6,
                hc_eps=1e-6,
            )
        self.assertIsInstance(unit, TileLangHCUnit)
        x = torch.randn(2, 5, hc, dim, dtype=torch.bfloat16)
        with self.assertRaises(RuntimeError):
            unit.pre(x)

    def test_tilelang_none_result_is_not_fallback(self) -> None:
        hc, dim = 4, 16
        fn, base, scale = _weights(hc, dim)
        unit = TileLangHCUnit(
            fn,
            base,
            scale,
            dim=dim,
            hc_mult=hc,
            hc_sinkhorn_iters=3,
            norm_eps=1e-6,
            hc_eps=1e-6,
        )
        import rtp_llm.models_py.modules.dsv4.hc.tilelang_impl as tilelang_impl

        old_pre = tilelang_impl.tk_mhc_pre
        tilelang_impl.tk_mhc_pre = lambda *args, **kwargs: None
        self.addCleanup(lambda: setattr(tilelang_impl, "tk_mhc_pre", old_pre))

        x = torch.randn(2, 5, hc, dim, dtype=torch.bfloat16)
        with self.assertRaises(RuntimeError):
            unit.pre(x)

    def test_tilelang_wrap_is_view_and_requires_contiguous(self) -> None:
        hc, dim = 4, 16
        fn, base, scale = _weights(hc, dim)
        unit = TileLangHCUnit(
            fn,
            base,
            scale,
            dim=dim,
            hc_mult=hc,
            hc_sinkhorn_iters=3,
            norm_eps=1e-6,
            hc_eps=1e-6,
        )
        import rtp_llm.models_py.modules.dsv4.hc.tilelang_impl as tilelang_impl

        seen = {}

        def fake_pre(residual, *args, **kwargs):
            seen["shape"] = tuple(residual.shape)
            seen["stride"] = tuple(residual.stride())
            seen["data_ptr"] = residual.data_ptr()
            seen["is_contiguous"] = residual.is_contiguous()
            return (
                torch.zeros(1, 5, dim, dtype=torch.bfloat16),
                torch.zeros(1, 5, hc, 1, dtype=torch.float32),
                torch.zeros(1, 5, hc, hc, dtype=torch.float32),
            )

        old_pre = tilelang_impl.tk_mhc_pre
        tilelang_impl.tk_mhc_pre = fake_pre
        self.addCleanup(lambda: setattr(tilelang_impl, "tk_mhc_pre", old_pre))

        x = torch.randn(5, hc, dim, dtype=torch.bfloat16)
        y, post, comb = unit.pre(x)
        self.assertEqual(tuple(y.shape), (5, dim))
        self.assertEqual(tuple(post.shape), (5, hc, 1))
        self.assertEqual(tuple(comb.shape), (5, hc, hc))
        self.assertEqual(seen["shape"], (1, 5, hc, dim))
        self.assertTrue(seen["is_contiguous"])
        self.assertEqual(seen["data_ptr"], x.data_ptr())

        x_noncontig = torch.randn(hc, 5, dim, dtype=torch.bfloat16).transpose(0, 1)
        self.assertEqual(tuple(x_noncontig.shape), (5, hc, dim))
        self.assertFalse(x_noncontig.is_contiguous())
        with self.assertRaisesRegex(ValueError, "must be contiguous"):
            unit.pre(x_noncontig)

    def test_tilelang_head_requires_fused_when_enabled(self) -> None:
        hc, dim = 4, 16
        fn, base, scale = _weights(hc, dim)
        head = TileLangHCHead(
            fn[:hc],
            base[:hc],
            scale[:1],
            dim=dim,
            hc_mult=hc,
            norm_eps=1e-6,
            hc_eps=1e-6,
        )
        import rtp_llm.models_py.modules.dsv4.hc.tilelang_impl as tilelang_impl

        calls: list[str] = []

        def fake_fused(residual, *args, **kwargs):
            calls.append("fused")
            return torch.ones(1, 5, dim, dtype=torch.bfloat16)

        def fake_old(residual, *args, **kwargs):
            calls.append("old")
            return torch.zeros(1, 5, dim, dtype=torch.bfloat16)

        old_fused = tilelang_impl.tk_mhc_head_fused
        old_head = tilelang_impl.tk_mhc_head
        tilelang_impl.tk_mhc_head_fused = fake_fused
        tilelang_impl.tk_mhc_head = fake_old
        self.addCleanup(lambda: setattr(tilelang_impl, "tk_mhc_head_fused", old_fused))
        self.addCleanup(lambda: setattr(tilelang_impl, "tk_mhc_head", old_head))

        x = torch.randn(5, hc, dim, dtype=torch.bfloat16)
        y = head.head(x)
        self.assertEqual(calls, ["fused"])
        self.assertEqual(tuple(y.shape), (5, dim))
        self.assertTrue(torch.all(y == 1))

        calls.clear()
        tilelang_impl.tk_mhc_head_fused = (
            lambda *args, **kwargs: calls.append("fused") or None
        )
        with self.assertRaisesRegex(RuntimeError, "fused head must succeed"):
            head.head(x)
        self.assertEqual(calls, ["fused"])

        calls.clear()
        with _env("DSV4_MHC_HEAD_FUSED", "0"):
            y = head.head(x)
        self.assertEqual(calls, ["old"])
        self.assertTrue(torch.all(y == 0))

    def test_shape_contract_is_checked_before_impl(self) -> None:
        hc, dim = 4, 16
        fn, base, scale = _weights(hc, dim)
        for mode in ("fallback", "tilelang"):
            with self.subTest(mode=mode), _env("DSV4_HC_IMPL", mode):
                unit = build_hc_unit(
                    fn,
                    base,
                    scale,
                    dim=dim,
                    hc_mult=hc,
                    hc_sinkhorn_iters=3,
                    norm_eps=1e-6,
                    hc_eps=1e-6,
                )
                with self.assertRaises(ValueError):
                    unit.pre(torch.randn(2, hc * dim))
                residual = torch.randn(2, 5, hc, dim, dtype=torch.bfloat16)
                x = torch.randn(2, 5, dim, dtype=torch.bfloat16)
                bad_post = torch.randn(2, 5, hc)
                comb = torch.randn(2, 5, hc, hc)
                with self.assertRaises(ValueError):
                    unit.post(x, residual, bad_post, comb)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    @torch.inference_mode()
    def test_prenorm_backends_match_reference_cuda(self):
        from rtp_llm.models_py.modules.dsv4.hc.mhc_tilelang import _import_tk

        _, mhc_post, _, _ = _import_tk()
        op = importlib.import_module(
            "rtp_llm.models_py.3rdparty.tile_kernels.modeling.mhc.ops.pre_big_fuse"
        )
        self.addCleanup(
            setattr,
            torch.backends.cuda.matmul,
            "allow_tf32",
            torch.backends.cuda.matmul.allow_tf32,
        )
        torch.backends.cuda.matmul.allow_tf32 = False
        hc, eps = 4, 1e-6
        for tokens, hidden_size, n_splits in (
            (64, 128, None),
            (1, 4096, None),
            (65, 4096, None),
            (512, 4096, None),
            (4096, 4096, None),
            (65, 4096, 1),
        ):
            torch.manual_seed(tokens)
            residual = torch.randn(
                tokens, hc, hidden_size, device="cuda", dtype=torch.bfloat16
            )
            weight = (
                torch.randn(24, hc * hidden_size, device="cuda")
                / (hc * hidden_size) ** 0.5
            )
            scale = torch.tensor([0.8, 1.2, 0.7], device="cuda")
            base = torch.randn(24, device="cuda") * 0.1
            flat = residual.float().flatten(1)
            mixes = (flat @ weight.T) * torch.rsqrt(
                flat.square().mean(-1, keepdim=True) + eps
            )
            pre = torch.sigmoid(mixes[:, :4] * scale[0] + base[:4]) + eps
            post = (2 * torch.sigmoid(mixes[:, 4:8] * scale[1] + base[4:8])).unsqueeze(
                -1
            )
            comb = (mixes[:, 8:] * scale[2] + base[8:]).view(-1, hc, hc)
            comb = comb.softmax(-1) + eps
            comb = comb / (comb.sum(-2, keepdim=True) + eps)
            for _ in range(19):
                comb = comb / (comb.sum(-1, keepdim=True) + eps)
                comb = comb / (comb.sum(-2, keepdim=True) + eps)
            output = (residual.float() * pre.unsqueeze(-1)).sum(1).bfloat16()
            sublayer = torch.randn(
                tokens, hidden_size, device="cuda", dtype=torch.bfloat16
            )
            ref_out = (
                post * sublayer.float().unsqueeze(-2)
                + comb.transpose(-1, -2) @ residual.float()
            ).bfloat16()
            for backend in ("deepgemm", "tilelang"):
                with self.subTest(
                    tokens=tokens,
                    hidden_size=hidden_size,
                    n_splits=n_splits,
                    backend=backend,
                ), patch.object(op._policy(), "resolve_backend", return_value=backend):
                    actual = op.mhc_pre_big_fuse(
                        residual,
                        weight,
                        scale,
                        base,
                        eps,
                        eps,
                        eps,
                        2.0,
                        20,
                        n_splits=n_splits,
                    )
                    from rtp_llm.models_py.modules.base.cuda.norm import RMSNorm

                    norm = RMSNorm(
                        torch.randn(hidden_size, device="cuda", dtype=torch.bfloat16),
                        1e-5,
                    )
                    normalized = op.mhc_pre_big_fuse(
                        residual,
                        weight,
                        scale,
                        base,
                        eps,
                        eps,
                        eps,
                        2.0,
                        20,
                        n_splits=n_splits,
                        output_norm_weight=norm.weight.data,
                        output_norm_eps=norm.variance_epsilon,
                    )[2]
                    expected_norm = norm(output)
                    norm_error = (
                        normalized.float() - expected_norm.float()
                    ).norm() / expected_norm.float().norm()
                    self.assertLess(norm_error.item(), 0.005)
                    actual_post = mhc_post(
                        sublayer.unsqueeze(0),
                        residual.unsqueeze(0),
                        actual[0].unsqueeze(0),
                        actual[1].unsqueeze(0),
                    ).squeeze(0)
                    for result, expected in zip(
                        (*actual, actual_post), (post, comb, output, ref_out)
                    ):
                        relative_error = (
                            result.float() - expected.float()
                        ).norm() / expected.float().norm()
                        self.assertLess(relative_error.item(), 0.005)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    @torch.inference_mode()
    def test_fused_post_pre_matches_unfused_cuda(self):
        from rtp_llm.models_py.modules.dsv4.hc.mhc_tilelang import _import_tk

        _import_tk()
        functional = importlib.import_module(
            "rtp_llm.models_py.3rdparty.tile_kernels.modeling.mhc.functional"
        )
        for tokens, hidden in ((1, 128), (1, 4096), (16, 4096), (16, 7168)):
            with self.subTest(tokens=tokens, hidden=hidden):
                torch.manual_seed(tokens)
                residual = torch.randn(
                    1, tokens, 4, hidden, device="cuda", dtype=torch.bfloat16
                )
                x = torch.randn(1, tokens, hidden, device="cuda", dtype=torch.bfloat16)
                post = torch.rand(1, tokens, 4, 1, device="cuda") * 2
                comb = torch.rand(1, tokens, 4, 4, device="cuda").softmax(-1)
                fn = torch.randn(24, 4 * hidden, device="cuda") / (4 * hidden) ** 0.5
                scale = torch.tensor([0.8, 1.2, 0.7], device="cuda")
                base = torch.randn(24, device="cuda") * 0.1
                ref_residual = functional.mhc_post(x, residual, post, comb)
                ref_input, (ref_post, ref_comb) = functional.mhc_pre(
                    ref_residual,
                    fn,
                    scale,
                    base,
                    post_mult_value=2.0,
                    sinkhorn_repeat=20,
                )

                from rtp_llm.models_py.modules.base.cuda.norm import RMSNorm

                norm = RMSNorm(
                    torch.randn(hidden, device="cuda", dtype=torch.bfloat16), 1e-5
                )
                ref_input = norm(ref_input.reshape(-1, hidden)).view_as(ref_input)

                def run():
                    return functional.mhc_fused_post_pre(
                        x,
                        residual,
                        post,
                        comb,
                        fn,
                        scale,
                        base,
                        rms_eps=1e-6,
                        mhc_pre_eps=1e-6,
                        mhc_sinkhorn_eps=1e-6,
                        mhc_post_mult_value=2.0,
                        sinkhorn_repeat=20,
                        output_norm_weight=norm.weight.data,
                        output_norm_eps=norm.variance_epsilon,
                    )

                actual = run()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured = run()
                for _ in range(3):
                    graph.replay()
                for outputs in (actual, captured):
                    for got, expected in zip(
                        outputs, (ref_residual, ref_post, ref_comb, ref_input)
                    ):
                        error = (
                            got.float() - expected.float()
                        ).norm() / expected.float().norm()
                        self.assertLess(error.item(), 0.005)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    @torch.inference_mode()
    def test_fused_decode_boundaries_match_unfused_cuda(self):
        from types import SimpleNamespace

        from rtp_llm.models_py.modules.dsv4.block import Block, forward_decode_layer

        layers = []
        for layer_id in range(2):
            layer = Block.__new__(Block)
            torch.nn.Module.__init__(layer)
            layer.layer_id = layer_id
            layer._mega_csa_adapter = layer._mega_hca_adapter = None
            from rtp_llm.models_py.modules.base.cuda.norm import RMSNorm

            layer.attn_norm = RMSNorm(
                torch.randn(4096, device="cuda", dtype=torch.bfloat16), 1e-5
            )
            layer.ffn_norm = RMSNorm(
                torch.randn(4096, device="cuda", dtype=torch.bfloat16), 1e-6
            )
            layer.attn = SimpleNamespace(
                forward_decode=lambda x, *args, **kwargs: x * 0.5
            )
            layer.ffn = lambda x, *args, **kwargs: x * 0.25
            for name in ("attn_hc", "ffn_hc"):
                fn, base, scale = _weights(4, 4096, device="cuda")
                setattr(
                    layer,
                    name,
                    TileLangHCUnit(
                        fn,
                        base,
                        scale,
                        dim=4096,
                        hc_mult=4,
                        hc_sinkhorn_iters=20,
                        norm_eps=1e-6,
                        hc_eps=1e-6,
                    ),
                )
            layers.append(layer)
        for tokens in (1, 8, 16, 17):
            with self.subTest(tokens=tokens):
                original = torch.randn(
                    1, tokens, 4, 4096, device="cuda", dtype=torch.bfloat16
                )
                ids = torch.zeros(1, tokens, device="cuda", dtype=torch.long)
                expected = original.clone()
                for layer in layers:
                    for hc, norm, factor in (
                        (layer.attn_hc, layer.attn_norm, 0.5),
                        (layer.ffn_hc, layer.ffn_norm, 0.25),
                    ):
                        inp, post, comb = hc.pre(expected)
                        out = norm(inp.reshape(-1, 4096)).view_as(inp) * factor
                        expected = hc.post(out, expected, post, comb)

                def run():
                    actual, prepared_input = original.clone(), None
                    for i, layer in enumerate(layers):
                        actual, prepared_input = forward_decode_layer(
                            layer,
                            layers[i + 1] if i + 1 < len(layers) else None,
                            actual,
                            None,
                            ids,
                            None,
                            prepared_input=prepared_input,
                        )
                    return actual

                eager = run()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured = run()
                for _ in range(2):
                    graph.replay()
                for actual in (eager, captured):
                    error = (
                        actual.float() - expected.float()
                    ).norm() / expected.float().norm()
                    self.assertLess(error.item(), 0.005)

    def test_tilelang_post_reuses_residual_buffer_in_place(self) -> None:
        # Pins the memory-saving wiring (no CUDA needed): _post_impl must pass
        # out=residual so the kernel writes in place instead of allocating a
        # fresh empty_like(residual). A future refactor that drops the alias
        # would silently re-introduce the per-call allocation; this catches it.
        hc, dim, T = 4, 16, 5
        fn, base, scale = _weights(hc, dim)
        unit = TileLangHCUnit(
            fn,
            base,
            scale,
            dim=dim,
            hc_mult=hc,
            hc_sinkhorn_iters=3,
            norm_eps=1e-6,
            hc_eps=1e-6,
        )
        import rtp_llm.models_py.modules.dsv4.hc.tilelang_impl as tilelang_impl

        captured: dict = {}

        def fake_post(x, residual, post, comb, hc_mult=4, out=None):
            captured["out_is_residual"] = out is residual
            captured["out_ptr"] = None if out is None else out.data_ptr()
            return residual if out is None else out

        old_post = tilelang_impl.tk_mhc_post
        tilelang_impl.tk_mhc_post = fake_post
        self.addCleanup(lambda: setattr(tilelang_impl, "tk_mhc_post", old_post))

        residual = torch.randn(T, hc, dim, dtype=torch.bfloat16)
        x = torch.randn(T, dim, dtype=torch.bfloat16)
        post = torch.randn(T, hc, 1, dtype=torch.float32)
        comb = torch.randn(T, hc, hc, dtype=torch.float32)
        out = unit.post(x, residual, post, comb)

        self.assertTrue(captured["out_is_residual"])
        # the aliased out points at the caller's residual storage
        self.assertEqual(captured["out_ptr"], residual.data_ptr())
        self.assertEqual(out.data_ptr(), residual.data_ptr())
        self.assertEqual(tuple(out.shape), (T, hc, dim))

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_tilelang_post_nonbf16_x_raises_dtype_error(self) -> None:
        # A non-bf16 sublayer output x is an upstream dtype bug, not a TileLang
        # availability miss: it must raise loudly (with the offending dtype),
        # not return None and get disguised as "TileLang unavailable". Reaches
        # the x.dtype check before any kernel import, so tilelang is not needed.
        from rtp_llm.models_py.modules.dsv4.hc.mhc_tilelang import tk_mhc_post

        hc, dim, T = 4, 128, 8
        residual = torch.randn(1, T, hc, dim, device="cuda", dtype=torch.bfloat16)
        post = torch.randn(1, T, hc, 1, device="cuda", dtype=torch.float32)
        comb = torch.randn(1, T, hc, hc, device="cuda", dtype=torch.float32)
        x_fp32 = torch.randn(1, T, dim, device="cuda", dtype=torch.float32)
        with torch.inference_mode():
            with self.assertRaisesRegex(RuntimeError, "bfloat16 sublayer output"):
                tk_mhc_post(x_fp32, residual, post, comb, hc_mult=hc)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_tilelang_post_in_place_matches_fresh_buffer_cuda(self) -> None:
        # Pins the out=residual aliasing invariant: writing the post output back
        # into the residual buffer must be bit-identical to writing into a fresh
        # buffer. The vendored kernel's safety relies on reading residual[pid_n]
        # into shared memory before overwriting out[pid_n]; if a future kernel
        # change breaks that read-before-write ordering, this fails loudly.
        from rtp_llm.models_py.modules.dsv4.hc.mhc_tilelang import tk_mhc_post

        hc, dim, T = 4, 128, 64
        torch.manual_seed(0)
        x = torch.randn(1, T, dim, device="cuda", dtype=torch.bfloat16)
        residual = torch.randn(1, T, hc, dim, device="cuda", dtype=torch.bfloat16)
        post = torch.randn(1, T, hc, 1, device="cuda", dtype=torch.float32)
        comb = torch.randn(1, T, hc, hc, device="cuda", dtype=torch.float32)

        residual_fresh = residual.clone()
        residual_alias = residual.clone()
        with torch.inference_mode():
            out_fresh = tk_mhc_post(x, residual_fresh, post, comb, hc_mult=hc)
            out_alias = tk_mhc_post(
                x, residual_alias, post, comb, hc_mult=hc, out=residual_alias
            )

        assert out_fresh is not None and out_alias is not None
        # fresh path allocated a new buffer; aliased path wrote into residual
        self.assertNotEqual(out_fresh.data_ptr(), residual_fresh.data_ptr())
        self.assertEqual(out_alias.data_ptr(), residual_alias.data_ptr())
        # in-place and fresh-buffer results must be bit-identical
        torch.testing.assert_close(out_alias, out_fresh, atol=0, rtol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_tilelang_head_matches_fallback_cuda(self) -> None:
        hc, dim = 4, 4096
        fn, base, scale = _weights(hc, dim, device="cuda")
        head_fn = fn[:hc]
        head_base = base[:hc]
        head_scale = scale[:1]
        x = torch.randn(1, 65, hc, dim, device="cuda", dtype=torch.bfloat16)
        fallback = FallbackHCHead(
            head_fn,
            head_base,
            head_scale,
            dim=dim,
            hc_mult=hc,
            norm_eps=1e-6,
            hc_eps=1e-6,
        )
        tilelang = TileLangHCHead(
            head_fn,
            head_base,
            head_scale,
            dim=dim,
            hc_mult=hc,
            norm_eps=1e-6,
            hc_eps=1e-6,
        )
        with torch.inference_mode():
            ref_y = fallback.head(x)
            for fused in ("0", "1"):
                with self.subTest(fused=fused), _env("DSV4_MHC_HEAD_FUSED", fused):
                    tk_y = tilelang.head(x)
                    torch.testing.assert_close(tk_y, ref_y, atol=2e-2, rtol=2e-2)


if __name__ == "__main__":
    unittest.main()
