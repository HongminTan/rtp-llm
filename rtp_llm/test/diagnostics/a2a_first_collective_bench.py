"""a2a-only CUDA-graph microbench: discriminate the per-graph first-collective
window (R181/R182) — is the stall paid once per graph launch, or per collective?

Test-carrier adaptation of bench/a2a_first_collective_bench.py (workspace
design artifact): torchrun replaced by torch.multiprocessing spawn so the case
runs as a single bazel py_test on CUDA_VISIBLE_DEVICES=0,1,2,3.

Variants (per R182.2 pre-registration):
  A: [~100 prefix tiny kernels] -> a2a x1
  B: [~100 prefix tiny kernels] -> a2a x1 -> a2a x2 (back to back)
Phases per variant: back-to-back replays (no host work between) and 1 ms
host-sleep between replays. Rank 0 profiles the last 16 replays and saves a
chrome trace; the built-in analysis reports the gap before a2a#0/a2a#1.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

MASTER_PORT = 29711


def _build_prefix(n_kernels: int, x: torch.Tensor) -> torch.Tensor:
    for _ in range(n_kernels):
        x = x.add_(1e-9)
    return x


def _capture_graph(device, prefix_kernels, n_a2a, msg_elems):
    send = torch.randn(msg_elems, device=device, dtype=torch.float32)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):  # side-stream warmup (official NCCL capture pattern)
            _build_prefix(prefix_kernels, send)
            dist.all_to_all_single(send, send)
    torch.cuda.current_stream().wait_stream(s)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        _build_prefix(prefix_kernels, send)
        for _ in range(n_a2a):
            dist.all_to_all_single(send, send)
    return g


def _worker(rank: int, world_size: int, args) -> None:
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(MASTER_PORT)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    device = torch.cuda.current_device()

    graphs = {}
    for variant, n_a2a in (("A", 1), ("B", 2)):
        graphs[variant] = _capture_graph(
            device, args.prefix_kernels, n_a2a, args.msg_elems
        )
        torch.cuda.synchronize()
    dist.barrier()

    for variant in ("A", "B"):
        g = graphs[variant]
        for _ in range(3):
            g.replay()
        torch.cuda.synchronize()
        dist.barrier()
        t0 = time.time()
        for _ in range(args.replays):
            g.replay()
        torch.cuda.synchronize()
        dist.barrier()
        b2b_ms = (time.time() - t0) / args.replays * 1000
        if rank == 0:
            print(f"[{variant}] back-to-back: {b2b_ms:.3f} ms/replay", flush=True)
        for _ in range(3):
            g.replay()
            time.sleep(0.001)
        torch.cuda.synchronize()
        dist.barrier()
        t0 = time.time()
        for _ in range(args.replays):
            g.replay()
            time.sleep(0.001)
        torch.cuda.synchronize()
        dist.barrier()
        gap_ms = (time.time() - t0) / args.replays * 1000
        if rank == 0:
            print(
                f"[{variant}] with 1ms host sleep: {gap_ms:.3f} ms/replay", flush=True
            )

    # Skew phase (R189 diagnostic): per-rank pre-launch delay = rank * skew_ms.
    # If the engine's first-collective window is absorbed inter-rank launch skew,
    # the per-replay cadence should grow by ~ (world-1) * skew_ms (the last rank
    # gates every replay), and rank 0's replay() call duration should approach
    # the full skew spread (its graph stalls at the first a2a waiting for the
    # late ranks). Measured without the profiler (the R187.3 CUPTI x tight a2a
    # hang face).
    if args.skew_ms > 0:
        g = graphs["B"]
        for _ in range(3):
            time.sleep(rank * args.skew_ms / 1000.0)
            g.replay()
        torch.cuda.synchronize()
        dist.barrier()
        launch_durs = []
        t0 = time.time()
        for i in range(args.replays):
            time.sleep(rank * args.skew_ms / 1000.0)
            t1 = time.time()
            g.replay()
            launch_durs.append((time.time() - t1) * 1000)
            if i + 1 < args.replays:
                torch.cuda.synchronize()
        torch.cuda.synchronize()
        dist.barrier()
        total_ms = (time.time() - t0) / args.replays * 1000
        durs = sorted(launch_durs)
        print(
            f"[skew {args.skew_ms}ms/step] rank{rank}: cadence={total_ms:.3f} ms/replay "
            f"replay() dur med={durs[len(durs)//2]:.3f} min={durs[0]:.3f} max={durs[-1]:.3f}",
            flush=True,
        )

    if rank == 0:
        from torch.profiler import ProfilerActivity, profile

        g = graphs["B"]
        with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
            for _ in range(16):
                g.replay()
            torch.cuda.synchronize()
        trace_path = os.path.join(args.output_dir, "a2a_bench_rank0.json")
        prof.export_chrome_trace(trace_path)
        print("trace saved:", trace_path, flush=True)
        _analyze_trace(trace_path)
    dist.barrier()
    dist.destroy_process_group()


def _analyze_trace(trace_path: str) -> None:
    evts = json.load(open(trace_path))
    evts = [
        e
        for e in (evts.get("traceEvents") or evts)
        if isinstance(e, dict) and "ts" in e and "dur" in e
    ]
    launches = sorted(
        (
            e
            for e in evts
            if e.get("cat") == "cuda_runtime" and "cudaGraphLaunch" in e.get("name", "")
        ),
        key=lambda e: e["ts"],
    )
    kernels = sorted(
        (e for e in evts if e.get("cat") == "kernel" and e.get("dur", 0) > 0),
        key=lambda e: e["ts"],
    )
    rows = []
    for i, L in enumerate(launches):
        ks = [k for k in kernels if L["ts"] <= k["ts"] < L["ts"] + 50000][:6000]
        a2a = [k for k in ks if "ncclDevKernel" in k["name"]]
        for j, k in enumerate(a2a):
            pre = [x for x in ks if x["ts"] < k["ts"]]
            last_end = max((x["ts"] + x["dur"] for x in pre), default=k["ts"])
            rows.append(
                {
                    "launch": i,
                    "a2a_idx": j,
                    "gap_before_ms": (k["ts"] - last_end) / 1000,
                    "a2a_dur_ms": k["dur"] / 1000,
                }
            )
    out_path = trace_path + ".a2a_gaps.json"
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=1)
    for j in range(2):
        g = [r["gap_before_ms"] for r in rows if r["a2a_idx"] == j]
        d = [r["a2a_dur_ms"] for r in rows if r["a2a_idx"] == j]
        if g:
            g.sort()
            d.sort()
            print(
                f"a2a#{j}: n={len(g)} gap med={g[len(g)//2]:.3f} min={g[0]:.3f} max={g[-1]:.3f} | "
                f"SR_dur med={d[len(d)//2]:.3f}",
                flush=True,
            )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default=os.environ.get("TEST_TMPDIR", "/tmp"))
    ap.add_argument("--replays", type=int, default=32)
    ap.add_argument("--prefix-kernels", type=int, default=100)
    ap.add_argument("--msg-elems", type=int, default=1024)
    ap.add_argument("--skew-ms", type=float, default=2.0)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    world = min(4, torch.cuda.device_count())
    assert world == 4, f"need 4 GPUs, got {world}"
    mp.spawn(_worker, args=(world, args), nprocs=world, join=True)
    print("a2a first-collective bench PASSED")


if __name__ == "__main__":
    main()
