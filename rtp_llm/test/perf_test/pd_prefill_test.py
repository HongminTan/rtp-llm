"""PD-disaggregated prefill perf harness for DeepSeek-V4-Flash on H20 (SM90).

Starts the two-role deployment used by the F-1 anchor evaluation (R149/R150):

  * PREFILL role (lane A, GPU 0-3): CP4 / EP4, FP8-converted checkpoint,
    ``grouped_fp8`` MoE, eager (no CUDA graph) — the TTFT-line form.
  * DECODE role (lane B, GPU 4-7): DP4 / EP4, CUDA-graph decode — the
    TPOT-line form.

Both roles cross-reference each other through ``REMOTE_SERVER_PORT`` and a
``MODEL_SERVICE_CONFIG`` service route (requests entering the prefill role
route their decode phase to the decode role). The measurement protocol is
reused verbatim from ``batch_decode_test``: this module only swaps the
single-process ``EngineServer`` for the dual-process ``PdEngineServer`` and
delegates to ``batch_decode_test.main()``.

Layer-1 verification greps (r65 gate 2): each role's process.log records its
engine CLI; check ``--role_type``, ``--cp_rotate_method``, ``--dp_size`` and
``--checkpoint_path`` lines per role.
"""

import argparse
import json
import logging
import os
from typing import Any, Dict, List, Optional

from rtp_llm.server.host_service import EndPoint, GroupEndPoint, ServiceRoute
from rtp_llm.test.utils.maga_server_manager import MagaServerManager

_PREFILL_DEVICES = [0, 1, 2, 3]
_DECODE_DEVICES = [4, 5, 6, 7]

# Role deltas appended to the common engine CLI built by the perf framework.
_PREFILL_ROLE_ARGS = (
    "--role_type PREFILL "
    "--enable_cuda_graph 0 "
    "--cp_rotate_method ALL_GATHER "
    "--enable_remote_cache "
    f"--world_size {len(_PREFILL_DEVICES)}"
)
_DECODE_ROLE_ARGS = (
    "--role_type DECODE "
    "--enable_cuda_graph 1 "
    "--decode_capture_config 1,2,4 "
    "--cp_rotate_method PREFILL_CP "
    "--enable_remote_cache "
    "--tp_size 1 "
    "--dp_size 4 "
    f"--world_size {len(_DECODE_DEVICES)}"
)


def _service_route_json(prefill_port: int, decode_port: int) -> str:
    route = ServiceRoute(
        service_id="v4_flash_pd_perf",
        use_local=True,
        role_endpoints=[
            GroupEndPoint(
                group="default",
                prefill_endpoint=EndPoint(
                    type="Vipserver",
                    address=f"127.0.0.1:{prefill_port}",
                    protocol="http",
                    path="/",
                ),
                decode_endpoint=EndPoint(
                    type="Vipserver",
                    address=f"127.0.0.1:{decode_port}",
                    protocol="http",
                    path="/",
                ),
            )
        ],
    )
    return route.model_dump_json()


class PdEngineServer:
    """Drop-in replacement for perf_test.server.EngineServer that starts the
    PREFILL/DECODE role pair and exposes the prefill role's HTTP port.

    The perf measurement requests enter at the prefill port (matching the
    pd_contract smoke precedent); the decode phase is routed to the decode
    role through the service route.
    """

    def __init__(self, args: argparse.Namespace, remaining_args: List[str]):
        self._args = args
        self._remaining_args = remaining_args
        self._prefill: Optional[MagaServerManager] = None
        self._decode: Optional[MagaServerManager] = None

    def start(
        self,
        max_seq_len: int,
        max_concurrency: int,
        server_start_timeout: int = 3600,
        use_batch_decode_scheduler: Optional[bool] = True,
    ) -> None:
        from rtp_llm.test.perf_test.server import EngineServer

        # Reuse the single-server CLI builder for the shared (common) part of
        # both roles' engine command lines; it also inherits the perf target's
        # baked args (checkpoint, parallel sizes, sched config, ...).
        shim = EngineServer(self._args, self._remaining_args)
        common_cli = shim._build_engine_cli(max_seq_len, max_concurrency)
        shim.stop()

        prefill_port = MagaServerManager.get_free_port()
        decode_port = MagaServerManager.get_free_port()

        common_env: Dict[str, Any] = {
            "FAKE_BALANCE_EXPERT": "1",
            "BATCH_DECODE_SCHEDULER_WARMUP_TYPE": (
                "0" if self._args.partial in (0, 1) else "1"
            ),
            "TORCH_CUDA_PROFILER_DIR": self._args.result_dir,
            "MODEL_SERVICE_CONFIG": _service_route_json(prefill_port, decode_port),
        }
        if use_batch_decode_scheduler is not None:
            common_env["USE_BATCH_DECODE_SCHEDULER"] = (
                "1" if use_batch_decode_scheduler else "0"
            )

        # world_size in the common CLI covers the combined 8-card form; each
        # role overrides it to its own slice via _PREFILL/_DECODE_ROLE_ARGS.
        self._prefill = MagaServerManager(
            env_args={
                **common_env,
                "REMOTE_SERVER_PORT": str(decode_port),
            },
            port=str(prefill_port),
            device_ids=list(_PREFILL_DEVICES),
            role_name="v4_flash_pd_prefill",
            process_file_name="process_prefill.log",
            smoke_args_str=f"{common_cli} {_PREFILL_ROLE_ARGS}",
        )
        self._decode = MagaServerManager(
            env_args={
                **common_env,
                "REMOTE_SERVER_PORT": str(prefill_port),
                # The decode role must not re-import the prefill-only envs.
            },
            port=str(decode_port),
            device_ids=list(_DECODE_DEVICES),
            role_name="v4_flash_pd_decode",
            process_file_name="process_decode.log",
            smoke_args_str=f"{common_cli} {_DECODE_ROLE_ARGS}",
        )
        try:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=2) as executor:
                starts = [
                    executor.submit(
                        manager.start_server, timeout=server_start_timeout
                    )
                    for manager in (self._prefill, self._decode)
                ]
                for result in starts:
                    if not result.result():
                        self._print_logs()
                        raise RuntimeError(
                            "PD engine role failed to start. Check role logs above."
                        )
        except Exception:
            self.stop()
            raise

    def _print_logs(self) -> None:
        for manager in (self._prefill, self._decode):
            if manager is not None:
                manager.print_process_log()

    def stop(self) -> None:
        prefill, decode = self._prefill, self._decode
        self._prefill = None
        self._decode = None
        # Stop the decode role first so the prefill role's remote calls fail
        # fast instead of hanging on a peer that is already gone.
        if decode is not None:
            decode.stop_server()
        if prefill is not None:
            prefill.stop_server()

    @property
    def port(self) -> int:
        assert self._prefill is not None, "PD server not started"
        return self._prefill.port

    @property
    def log_file_path(self) -> Optional[str]:
        return self._prefill.log_file_path if self._prefill else None


def main() -> str:
    from rtp_llm.test.perf_test import batch_decode_test

    # Swap the single-process server for the PD pair; everything else
    # (arg parsing, grid runner, warmup, trace flush, metrics) is reused.
    batch_decode_test.EngineServer = PdEngineServer
    return batch_decode_test.main()


if __name__ == "__main__":
    main()
