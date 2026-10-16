"""#66 drift arm (R222): long free-flow phase-drift decode load.

Reuses the batch_decode_test engine machinery (args / env / EngineServer)
and replaces the fixed grid with a drifting load generator:

* wave size uniform{1..4} requests, inter-wave gap uniform(0, 2.5)s;
* an idle window of uniform(10, 30)s every uniform(90, 150)s — the empty-
  batch / fake-stream crossover phase (R168.3-4);
* greedy short-decode requests (decode_test_length tokens, top_k=1);
* a semantic spot-check ("capital of France") every ~3 min;
* duration >= --drift_duration (default 1200 s).

The test FAILS if any request errors/times out (client-side harvest = one
of the R168 five-signature wedges) or any semantic spot-check loses
content correctness. The wave/gap/spot-check log doubles as the drift-load
self-evidence fingerprint (R222.2.4).
"""

import logging
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Tuple

from rtp_llm.test.perf_test.batch_decode_test import (
    _apply_engine_env,
    _apply_run_overrides,
    _effective_performance_config,
    _redact_argv,
    parse_args,
    resolve_perf_engine_paths,
    write_test_info,
)
from rtp_llm.test.perf_test.batch_perf_impl import _curl_server_single_worker
from rtp_llm.test.perf_test.dataset import extract_arg
from rtp_llm.test.perf_test.perf_config import _engine_arg_argv
from rtp_llm.test.perf_test.perf_utils import query_engine_status
from rtp_llm.test.perf_test.server import EngineServer
from rtp_llm.test.perf_test.test_util import create_query

SEMANTIC_PROMPT = "What is the capital of France?"
SEMANTIC_NEEDLE = "Paris"
SEMANTIC_MAX_TOKENS = 32


def _pop_int(remaining: List[str], key: str, default: int) -> int:
    flag = f"--{key}"
    prefix = f"--{key}="
    for i, arg in enumerate(remaining):
        if arg == flag and i + 1 < len(remaining):
            value = int(remaining[i + 1])
            del remaining[i : i + 2]
            return value
        if arg.startswith(prefix):
            value = int(arg[len(prefix) :])
            del remaining[i : i + 1]
            return value
    return default


def _decode_ms(info: "ResponseInfo") -> float:
    return float(info.decode_time_per_token or 0.0)


def _semantic_check(port: int) -> bool:
    """One greedy spot-check via the internal engine API (same request shape
    as _curl_server_single_worker). Content correctness only — the #79 noise
    family's formatting differences are exempt by design."""
    import requests

    req = {
        "prompt": SEMANTIC_PROMPT,
        "top_k": 1,
        "generate_config": {
            "max_new_tokens": SEMANTIC_MAX_TOKENS,
            "min_new_tokens": SEMANTIC_MAX_TOKENS,
        },
    }
    try:
        response = requests.post(f"http://127.0.0.1:{port}", json=req, timeout=300)
        if response.status_code != 200:
            logging.warning("[drift] semantic check HTTP %d", response.status_code)
            return False
        data = response.json()
        text = "".join(str(part) for part in (data.get("response") or []))
        if SEMANTIC_NEEDLE not in text:
            logging.warning("[drift] semantic check lost needle: %r", text[:120])
            return False
        return True
    except Exception as exc:  # noqa: BLE001
        logging.warning("[drift] semantic check exception: %s", exc)
        return False


def main() -> str:
    from rtp_llm.config.log_config import setup_logging

    setup_logging()

    args, remaining = parse_args()
    drift_duration = _pop_int(remaining, "drift_duration", 1200)
    drift_seed = _pop_int(remaining, "drift_seed", 20261017)
    # Fixed wave size (0 = rng 1..4). The #66-family transition wedge
    # triggers on partial waves (real+fake mixing, R231); a fixed full wave
    # (4 on DP4) keeps the semantic gate focused on the multi-step path
    # without that orthogonal defect contaminating the run.
    drift_fixed_wave = _pop_int(remaining, "drift_fixed_wave", 0)

    engine_env_names = _apply_engine_env(args.engine_env)
    _apply_run_overrides(args)
    remaining.extend(_engine_arg_argv(args.engine_arg))
    remaining = resolve_perf_engine_paths(remaining)
    if extract_arg(remaining, "use_batch_decode_scheduler") is None:
        remaining.extend(["--use_batch_decode_scheduler", "1"])
    os.makedirs(args.result_dir, exist_ok=True)
    EngineServer.propagate_engine_env(remaining)
    logging.info(f"Result directory: {args.result_dir}")
    logging.info(f"Engine args forwarded to server: {_redact_argv(remaining)}")

    random.seed(drift_seed)
    rng = random.Random(drift_seed)

    # Phase 1/2: configure + serve (same machinery as the grid carrier).
    from rtp_llm.test.perf_test.perf_config import prepare_config

    config = prepare_config(args, remaining)
    effective_runtime_config = _effective_performance_config(
        remaining,
        engine_env_names,
        {
            "DP_SIZE": args.dp_size,
            "MAX_SEQ_LEN": config.max_seq_len,
            "CONCURRENCY_LIMIT": config.max_concurrency,
        },
    )
    write_test_info(
        args,
        remaining,
        engine_env_names,
        status="running",
        effective_max_seq_len=config.max_seq_len,
        service_concurrency_limit=config.max_concurrency,
        effective_runtime_config=effective_runtime_config,
    )

    server = EngineServer(args, remaining)
    events: List[Dict[str, Any]] = []
    request_failures = 0
    semantic_failures = 0
    semantic_total = 0
    ok_count = 0
    try:
        server.start(
            max_seq_len=config.max_seq_len,
            max_concurrency=config.max_concurrency,
            use_batch_decode_scheduler=True,
        )
        query_engine_status(server.port)
        input_query_dict = create_query(input_len_list=config.all_seq_lens)
        queries = list(input_query_dict.values())
        if not queries:
            raise RuntimeError("no input queries built for drift load")

        logging.info(
            "[drift] start: duration=%ds seed=%d queries=%d",
            drift_duration,
            drift_seed,
            len(queries),
        )
        end_time = time.time() + drift_duration
        next_gap = time.time() + rng.uniform(90, 150)
        next_semantic = time.time() + 30
        wave_index = 0
        with ThreadPoolExecutor(max_workers=4) as pool:
            while time.time() < end_time:
                now = time.time()
                if now >= next_gap:
                    gap = rng.uniform(10, 30)
                    events.append(
                        {
                            "t": round(now, 1),
                            "event": "idle_gap",
                            "dur_s": round(gap, 1),
                        }
                    )
                    logging.info("[drift] idle gap %.1fs", gap)
                    time.sleep(min(gap, max(0.0, end_time - time.time())))
                    next_gap = time.time() + rng.uniform(90, 150)
                    continue
                if now >= next_semantic and now + 5 < end_time:
                    semantic_total += 1
                    passed = _semantic_check(server.port)
                    if not passed:
                        semantic_failures += 1
                    events.append(
                        {
                            "t": round(now, 1),
                            "event": "semantic",
                            "passed": bool(passed),
                        }
                    )
                    next_semantic = time.time() + 180
                    continue

                wave_size = (
                    drift_fixed_wave if drift_fixed_wave > 0 else rng.randint(1, 4)
                )
                wave_index += 1
                prompts = [rng.choice(queries) for _ in range(wave_size)]
                futures = [
                    pool.submit(
                        _curl_server_single_worker,
                        i,
                        server.port,
                        prompt,
                        True,
                        args.decode_test_length,
                        wait_time=300,
                    )
                    for i, prompt in enumerate(prompts)
                ]
                wave_ms: List[float] = []
                for future in futures:
                    resp = future.result()
                    if not resp.success:
                        request_failures += 1
                        logging.warning("[drift] request failed (wave %d)", wave_index)
                    else:
                        ok_count += 1
                        wave_ms.append(_decode_ms(resp))
                events.append(
                    {
                        "t": round(now, 1),
                        "event": "wave",
                        "size": wave_size,
                        "decode_ms_med": (
                            round(sorted(wave_ms)[len(wave_ms) // 2], 2)
                            if wave_ms
                            else None
                        ),
                    }
                )
                logging.info(
                    "[drift] wave %d size=%d ok=%d decode_ms_med=%s",
                    wave_index,
                    wave_size,
                    len(wave_ms),
                    round(sorted(wave_ms)[len(wave_ms) // 2], 2) if wave_ms else "n/a",
                )
                time.sleep(rng.uniform(0.0, 2.5))

        summary = {
            "duration_s": drift_duration,
            "seed": drift_seed,
            "waves": wave_index,
            "ok_requests": ok_count,
            "request_failures": request_failures,
            "semantic_total": semantic_total,
            "semantic_failures": semantic_failures,
            "events": events,
        }
        with open(os.path.join(args.result_dir, "drift_summary.json"), "w") as f:
            import json

            json.dump(summary, f, indent=1)
        failed = request_failures > 0 or semantic_failures > 0
        write_test_info(
            args,
            remaining,
            engine_env_names,
            status="failed" if failed else "completed",
            effective_max_seq_len=config.max_seq_len,
            service_concurrency_limit=config.max_concurrency,
            effective_runtime_config=effective_runtime_config,
        )
        if failed:
            raise RuntimeError(
                f"drift arm failed: request_failures={request_failures} "
                f"semantic_failures={semantic_failures}/{semantic_total}"
            )
        logging.info(
            "[drift] PASSED: waves=%d ok=%d semantic=%d",
            wave_index,
            ok_count,
            semantic_total,
        )
        return args.result_dir
    finally:
        server.stop()


if __name__ == "__main__":
    main()
