"""Resource setup and SM partition boundaries for DSV4 green contexts."""

import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from rtp_llm.models_py.modules.dsv4 import _green_context as green_context


class GreenContextTest(unittest.TestCase):
    def test_setup_cleanup_and_empty_comm_partition(self):
        def resource(count):
            return SimpleNamespace(sm=SimpleNamespace(smCount=count))

        for failure in (None, "context", "stream", "wrapper"):
            with self.subTest(failure=failure):
                driver = mock.Mock()
                driver.CUresult.CUDA_SUCCESS = 0
                driver.cuDeviceGetDevResource.return_value = (0, resource(16))
                driver.cuDevSmResourceSplitByCount.return_value = (
                    0,
                    [resource(8), resource(8)],
                    2,
                    resource(0),
                )
                driver.cuDevResourceGenerateDesc.return_value = (0, "desc")
                driver.cuGreenCtxCreate.side_effect = [
                    (0, "shared"),
                    (1,) if failure == "context" else (0, "comm"),
                ]
                driver.cuGreenCtxStreamCreate.side_effect = [
                    (0, 101),
                    (1,) if failure == "stream" else (0, 102),
                ]
                driver.cuStreamDestroy.return_value = (0,)
                driver.cuGreenCtxDestroy.return_value = (0,)
                bindings = SimpleNamespace(driver=driver)
                with mock.patch.dict(
                    sys.modules, {"cuda.bindings": bindings}
                ), mock.patch.object(
                    green_context, "_DEVICE_PARTITIONS", {}
                ), mock.patch(
                    "torch.cuda.ExternalStream"
                ) as external:
                    external.side_effect = [
                        mock.Mock(),
                        (
                            RuntimeError("wrapper failed")
                            if failure == "wrapper"
                            else mock.Mock()
                        ),
                    ]
                    if failure is None:
                        windows = green_context.get_or_create_partition_pairs(
                            torch.device("cuda:0")
                        )
                        self.assertEqual(len(windows), 2)
                        self.assertEqual(windows[1].compute_sms, 8)
                        driver.cuStreamDestroy.assert_not_called()
                        driver.cuGreenCtxDestroy.assert_not_called()
                        self.assertIn(0, green_context._DEVICE_PARTITIONS)
                        self.assertIs(
                            green_context.get_or_create_partition_pairs(
                                torch.device("cuda:0")
                            ),
                            windows,
                        )
                        driver.cuDeviceGetDevResource.assert_called_once()
                    else:
                        with self.assertRaises(RuntimeError):
                            green_context.get_or_create_partition_pairs(
                                torch.device("cuda:0")
                            )
                        self.assertEqual(green_context._DEVICE_PARTITIONS, {})
                        handles = [102, 101] if failure == "wrapper" else [101]
                        contexts = (
                            ["shared"] if failure == "context" else ["comm", "shared"]
                        )
                        self.assertEqual(
                            driver.cuStreamDestroy.call_args_list,
                            [mock.call(handle) for handle in handles],
                        )
                        self.assertEqual(
                            driver.cuGreenCtxDestroy.call_args_list,
                            [mock.call(context) for context in contexts],
                        )
                    for args in driver.cuDevResourceGenerateDesc.call_args_list:
                        self.assertTrue(args.args[0])

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
    def test_partitions_are_complementary(self):
        import triton
        import triton.language as tl

        from rtp_llm.models_py.utils.arch import get_num_device_sms

        @triton.jit
        def record_smid(out_ptr):
            smid = tl.inline_asm_elementwise(
                "mov.u32 $0, %smid;", "=r", [], dtype=tl.int32, is_pure=False, pack=1
            )
            tl.store(out_ptr + tl.program_id(0), smid)

        def sms_of(stream):
            out = torch.full((4096,), -1, dtype=torch.int32, device="cuda")
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                record_smid[(out.numel(),)](out)
            torch.cuda.synchronize()
            return set(out.tolist())

        num_sms = get_num_device_sms()
        windows = green_context.get_or_create_partition_pairs(torch.device("cuda"))
        self.assertIsNone(windows[0])
        for window in windows[1:]:
            shared, comm = sms_of(window.compute_stream), sms_of(window.comm_stream)
            self.assertEqual(len(shared), window.compute_sms)
            self.assertFalse(shared & comm)
            self.assertEqual(len(shared | comm), num_sms)
        self.assertLess(
            sms_of(windows[1].compute_stream), sms_of(windows[2].compute_stream)
        )
