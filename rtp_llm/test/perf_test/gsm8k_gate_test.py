"""GSM8K task-level gate: batch-driven eval on the perf-target engine form.

Design (team/perf-log.md R34): no HTTP serving shell — the engine comes up
through the same MagaServerManager/EngineServer chain as the decode perf
target (graph1 / ctxbs128 / reserve 16384 form, proven deadlock-free over
dozens of runs), and 200 fixed GSM8K questions are driven through the
internal completion port at concurrency <= 8 (pitfalls #10). Both arms use
this same target; the only difference is DSV4_MHC_PRE_GEMM_BACKEND (set via
bazel --test_env, which overrides the BUILD env per pitfall #6).

Scoring: exact match of the first "#### <number>" marker in the generated
CoT against the gold answer (thousands separators stripped). The gate
criterion (INBOX round 7) is arm-B accuracy >= arm-A accuracy - 1pp with no
pathological outputs (finish rate, output-length stats, spot checks).
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List

import requests

from rtp_llm.test.perf_test.dataset import extract_arg
from rtp_llm.test.perf_test.server import EngineServer


def _parse(argv: List[str]):
    parser = argparse.ArgumentParser(
        description="GSM8K gate runner. Unrecognized args forward to the engine."
    )
    parser.add_argument("--question_file", type=str, required=True)
    parser.add_argument("--fewshot_prefix", type=str, required=True)
    parser.add_argument("--arm_tag", type=str, required=True)
    parser.add_argument("--num_questions", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--request_timeout", type=int, default=300)
    parser.add_argument("--dp_size", type=int, default=4)
    parser.add_argument("--partial", type=int, default=1)
    parser.add_argument("--max_seq_len", type=int, default=8192)
    parser.add_argument("--result_dir", type=str, default=os.getcwd())
    return parser.parse_known_args(argv)


def _post_question(
    port: int, prompt: str, max_new_tokens: int, timeout: int
) -> Dict[str, Any]:
    req = {
        "prompt": prompt,
        "generate_config": {
            "max_new_tokens": max_new_tokens,
            "min_new_tokens": 1,
            "temperature": 0.0,
            "top_k": 1,
            "force_sp_accept": True,
        },
        "top_k": 1,
    }
    try:
        resp = requests.post(f"http://127.0.0.1:{port}", json=req, timeout=timeout)
        if resp.status_code != 200:
            return {"ok": False, "error": f"http {resp.status_code}: {resp.content[:200]}"}
        return {"ok": True, "body": resp.json()}
    except Exception as e:  # noqa: BLE001 - record per-question failures verbatim
        return {"ok": False, "error": str(e)[:200]}


def _score(text: str, gold: str) -> str:
    m = re.search(r"####\s*([\-\d\.,]+)", text or "")
    if not m:
        return "no_marker"
    pred = m.group(1).replace(",", "").rstrip(".")
    gold_s = str(gold).replace(",", "").rstrip(".")
    try:
        return "hit" if abs(float(pred) - float(gold_s)) < 1e-9 else "miss"
    except ValueError:
        return "hit" if pred == gold_s else "miss"


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    args, remaining = _parse(sys.argv[1:])
    EngineServer.propagate_engine_env(remaining)

    with open(args.question_file, encoding="utf-8") as f:
        questions = [json.loads(line) for line in f][: args.num_questions]
    with open(args.fewshot_prefix, encoding="utf-8") as f:
        prefix = f.read()
    prompts = {q["qid"]: prefix + q["question"] for q in questions}
    logging.info(
        "[GSM8K GATE] arm=%s questions=%d concurrency=%d max_new_tokens=%d",
        args.arm_tag,
        len(questions),
        args.concurrency,
        args.max_new_tokens,
    )

    server = EngineServer(args, remaining)
    results: List[Dict[str, Any]] = []
    start_all = time.monotonic()
    try:
        server.start(
            max_seq_len=args.max_seq_len,
            max_concurrency=args.concurrency,
            use_batch_decode_scheduler=True,
        )
        port = server.port
        logging.info("[GSM8K GATE] engine up on port %d", port)

        def _one(q):
            t0 = time.monotonic()
            r = _post_question(port, prompts[q["qid"]], args.max_new_tokens, args.request_timeout)
            wall = time.monotonic() - t0
            entry: Dict[str, Any] = {
                "qid": q["qid"],
                "gold": q["gold"],
                "wall_s": round(wall, 2),
            }
            if r.get("ok"):
                body = r["body"]
                texts = body.get("response") or []
                aux = (body.get("aux_info") or [{}])[0]
                text = texts[0] if texts else ""
                entry["ok"] = True
                entry["finished"] = bool(body.get("finished"))
                entry["output_len"] = int(aux.get("output_len", len(text)))
                entry["text"] = text
                entry["verdict"] = _score(text, q["gold"])
            else:
                entry["ok"] = False
                entry["error"] = r.get("error")
                entry["verdict"] = "request_failed"
            return entry

        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            for i, entry in enumerate(pool.map(_one, questions), 1):
                results.append(entry)
                if i % 25 == 0:
                    logging.info("[GSM8K GATE] %d/%d done", i, len(questions))
    finally:
        server.stop()

    wall_total = time.monotonic() - start_all
    n = len(results)
    hits = sum(1 for r in results if r.get("verdict") == "hit")
    misses = sum(1 for r in results if r.get("verdict") == "miss")
    no_marker = sum(1 for r in results if r.get("verdict") == "no_marker")
    failed = sum(1 for r in results if r.get("verdict") == "request_failed")
    finished = sum(1 for r in results if r.get("finished"))
    lens = sorted(r.get("output_len", 0) for r in results if r.get("ok"))

    def _pct(vals, p):
        if not vals:
            return 0
        k = max(0, min(len(vals) - 1, int(round(p / 100.0 * (len(vals) - 1)))))
        return vals[k]

    summary = {
        "arm_tag": args.arm_tag,
        "n": n,
        "accuracy": round(hits / n, 4) if n else 0.0,
        "hits": hits,
        "misses": misses,
        "no_marker": no_marker,
        "request_failed": failed,
        "finished_rate": round(finished / n, 4) if n else 0.0,
        "output_len_p50": _pct(lens, 50),
        "output_len_p90": _pct(lens, 90),
        "output_len_max": lens[-1] if lens else 0,
        "wall_total_s": round(wall_total, 1),
    }
    out_path = os.path.join(args.result_dir, f"gsm8k_gate_{args.arm_tag}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_question": results}, f, ensure_ascii=False, indent=1)
    logging.info("[GSM8K GATE SUMMARY] %s", json.dumps(summary))
    logging.info("[GSM8K GATE] full dump: %s", out_path)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
