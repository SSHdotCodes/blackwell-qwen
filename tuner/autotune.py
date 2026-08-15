from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any
from urllib.request import urlopen

from tuner.bench import (
    REALISTIC_SPEED_PROMPT,
    compare_quality,
    run_concurrent,
    run_long_context,
    run_quality,
)


MODEL_NAME = "Qwen/Qwen3.8-27B-FP8"
PORT = 30000
BASE_URL = f"http://127.0.0.1:{PORT}/v1"


@dataclass(frozen=True)
class Candidate:
    name: str
    attention_backend: str
    fp8_gemm_backend: str
    chunked_prefill: int
    speculative_algorithm: str | None = None
    draft_model_path: str | None = None
    speculative_num_steps: int = 3
    speculative_num_draft_tokens: int = 4
    speculative_attention_mode: str | None = None
    page_size: int = 1
    enable_spec_v2: bool = False
    cuda_graph_max_bs: int = 32
    mamba_ratio: float = 1.8
    max_running_requests: int = 32

    @property
    def speculative(self) -> bool:
        return self.speculative_algorithm is not None

    def profile_flags(self) -> list[str]:
        flags = [
            "--attention-backend",
            self.attention_backend,
            "--fp8-gemm-backend",
            self.fp8_gemm_backend,
            "--chunked-prefill-size",
            str(self.chunked_prefill),
            "--max-prefill-tokens",
            str(self.chunked_prefill),
            "--cuda-graph-max-bs",
            str(self.cuda_graph_max_bs),
            "--mamba-full-memory-ratio",
            str(self.mamba_ratio),
            "--max-running-requests",
            str(self.max_running_requests),
        ]
        if self.page_size != 1:
            flags.extend(["--page-size", str(self.page_size)])
        if self.speculative_attention_mode:
            flags.extend(
                ["--speculative-attention-mode", self.speculative_attention_mode]
            )
        if self.speculative_algorithm == "EAGLE":
            flags.extend(
                [
                    "--speculative-algorithm",
                    "EAGLE",
                    "--speculative-num-steps",
                    str(self.speculative_num_steps),
                    "--speculative-eagle-topk",
                    "1",
                    "--speculative-num-draft-tokens",
                    str(self.speculative_num_draft_tokens),
                ]
            )
        elif self.speculative_algorithm == "DSPARK":
            if not self.draft_model_path:
                raise ValueError("DSPARK requires a draft model path")
            flags.extend(
                [
                    "--speculative-algorithm",
                    "DSPARK",
                    "--speculative-draft-model-path",
                    self.draft_model_path,
                    "--speculative-dspark-block-size",
                    "7",
                    "--speculative-draft-model-quantization",
                    "unquant",
                    "--mamba-scheduler-strategy",
                    "extra_buffer",
                ]
            )
        return flags


BASELINE = Candidate("flashinfer_auto_c2k", "flashinfer", "auto", 2048)
MTP_CANDIDATES = [
    Candidate("flashinfer_cutlass_mtp_c1k", "flashinfer", "cutlass", 1024, "EAGLE"),
    Candidate("flashinfer_cutlass_mtp_c2k", "flashinfer", "cutlass", 2048, "EAGLE"),
    Candidate("flashinfer_cutlass_mtp_c4k", "flashinfer", "cutlass", 4096, "EAGLE"),
    Candidate("flashinfer_cutlass_mtp_c8k", "flashinfer", "cutlass", 8192, "EAGLE"),
    Candidate("flashinfer_triton_mtp_c2k", "flashinfer", "triton", 2048, "EAGLE"),
    Candidate(
        "flashinfer_trtllm_mtp_c2k", "flashinfer", "flashinfer_trtllm", 2048, "EAGLE"
    ),
    Candidate("triton_cutlass_mtp_c2k", "triton", "cutlass", 2048, "EAGLE"),
]
DSPARK_CANDIDATES = [
    Candidate("flashinfer_cutlass_dspark_c1k", "flashinfer", "cutlass", 1024, "DSPARK"),
    Candidate("flashinfer_cutlass_dspark_c2k", "flashinfer", "cutlass", 2048, "DSPARK"),
    Candidate("flashinfer_cutlass_dspark_c4k", "flashinfer", "cutlass", 4096, "DSPARK"),
    Candidate("flashinfer_auto_dspark_c2k", "flashinfer", "auto", 2048, "DSPARK"),
    Candidate("fa3_cutlass_dspark_c2k", "fa3", "cutlass", 2048, "DSPARK"),
]
MTP_DEPTH_CANDIDATES = [
    Candidate(
        f"flashinfer_cutlass_mtp_s{steps}_c1k",
        "flashinfer",
        "cutlass",
        1024,
        "EAGLE",
        speculative_num_steps=steps,
        speculative_num_draft_tokens=steps + 1,
    )
    for steps in range(2, 8)
]
MTP_ADVANCED_CANDIDATES = [
    Candidate(
        f"flashinfer_cutlass_mtp_s{steps}_decode_p{page_size}",
        "flashinfer",
        "cutlass",
        1024,
        "EAGLE",
        speculative_num_steps=steps,
        speculative_num_draft_tokens=steps + 1,
        speculative_attention_mode="decode",
        page_size=page_size,
        enable_spec_v2=True,
    )
    for page_size in (1, 64)
    for steps in (2, 3)
]


def select_candidates(candidate_set: str, draft_model_path: str | None) -> list[Candidate]:
    if candidate_set == "mtp":
        chosen = [BASELINE, *MTP_CANDIDATES]
    elif candidate_set == "mtp_depth":
        chosen = [BASELINE, *MTP_DEPTH_CANDIDATES]
    elif candidate_set == "finalists":
        chosen = [BASELINE, *MTP_DEPTH_CANDIDATES[:2]]
    elif candidate_set == "advanced":
        chosen = [BASELINE, *MTP_ADVANCED_CANDIDATES]
    elif candidate_set == "dspark":
        chosen = [BASELINE, *DSPARK_CANDIDATES]
    else:
        chosen = [BASELINE, *DSPARK_CANDIDATES, *MTP_CANDIDATES]
    return [
        replace(candidate, draft_model_path=draft_model_path)
        if candidate.speculative_algorithm == "DSPARK"
        else candidate
        for candidate in chosen
    ]


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def gpu_snapshot() -> str:
    command = [
        "nvidia-smi",
        "--query-gpu=name,uuid,driver_version,memory.total,power.limit,clocks.max.sm,compute_cap",
        "--format=csv,noheader",
    ]
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:  # noqa: BLE001
        return f"unavailable: {exc}"


def base_server_command(model_path: str, candidate: Candidate) -> list[str]:
    return [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        model_path,
        "--served-model-name",
        MODEL_NAME,
        "--trust-remote-code",
        "--context-length",
        "262144",
        "--mem-fraction-static",
        "0.90",
        "--kv-cache-dtype",
        "bfloat16",
        "--mamba-ssm-dtype",
        "float32",
        "--mamba-radix-cache-strategy",
        "extra_buffer_lazy",
        "--reasoning-parser",
        "qwen3",
        "--tool-call-parser",
        "qwen3_coder",
        "--enable-metrics",
        "--host",
        "127.0.0.1",
        "--port",
        str(PORT),
        *candidate.profile_flags(),
    ]


def wait_ready(process: subprocess.Popen[str], timeout: int = 1200) -> tuple[bool, str]:
    deadline = time.monotonic() + timeout
    last_error = "server did not answer"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return False, f"server exited with code {process.returncode}"
        try:
            with urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=5) as response:
                if response.status == 200:
                    return True, "ready"
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
        time.sleep(5)
    return False, last_error


def stop_server(process: subprocess.Popen[str] | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=90)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=30)
    # Give CUDA and shared-memory workers time to release resources.
    time.sleep(8)


def start_server(
    model_path: str, candidate: Candidate, output_dir: Path
) -> tuple[subprocess.Popen[str], Any, float]:
    log_path = output_dir / f"server-{candidate.name}.log"
    log_handle = log_path.open("a", buffering=1)
    command = base_server_command(model_path, candidate)
    started = time.perf_counter()
    process = subprocess.Popen(
        command,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
        env={
            **os.environ,
            "PYTHONUNBUFFERED": "1",
            **({"SGLANG_ENABLE_SPEC_V2": "1"} if candidate.enable_spec_v2 else {}),
        },
    )
    ready, detail = wait_ready(process)
    startup_s = time.perf_counter() - started
    if not ready:
        stop_server(process)
        log_handle.close()
        raise RuntimeError(f"{candidate.name} failed startup after {startup_s:.1f}s: {detail}")
    return process, log_handle, startup_s


def candidate_benchmarks(tokenizer: Any) -> dict[str, Any]:
    return {
        "realistic_single_stream": run_concurrent(
            BASE_URL,
            MODEL_NAME,
            tokenizer,
            input_tokens=0,
            output_tokens=512,
            concurrency=1,
            requests=6,
            prompt=REALISTIC_SPEED_PROMPT,
        ),
        "latency": run_concurrent(
            BASE_URL,
            MODEL_NAME,
            tokenizer,
            input_tokens=1024,
            output_tokens=512,
            concurrency=1,
            requests=6,
        ),
        "throughput_c16": run_concurrent(
            BASE_URL,
            MODEL_NAME,
            tokenizer,
            input_tokens=1024,
            output_tokens=512,
            concurrency=16,
            requests=32,
        ),
        "throughput_c32": run_concurrent(
            BASE_URL,
            MODEL_NAME,
            tokenizer,
            input_tokens=1024,
            output_tokens=512,
            concurrency=32,
            requests=64,
        ),
    }


def safe_number(value: Any, default: float) -> float:
    try:
        number = float(value)
        return number if number == number else default
    except (TypeError, ValueError):
        return default


def pick_winners(results: dict[str, Any]) -> tuple[str, str]:
    eligible = {
        name: row
        for name, row in results["candidates"].items()
        if row.get("status") == "completed" and row.get("quality", {}).get("quality_safe")
    }
    if not eligible:
        # The non-speculative reference is the checkpoint-precision fallback.
        eligible = {
            name: row
            for name, row in results["candidates"].items()
            if row.get("status") == "completed"
            and row["candidate"].get("speculative_algorithm") is None
        }
    if not eligible:
        raise RuntimeError("No serving candidate completed")
    latency = min(
        eligible,
        key=lambda name: safe_number(
            eligible[name]["benchmarks"]["latency"].get("median_ttft_ms"), float("inf")
        ),
    )
    throughput = max(
        eligible,
        key=lambda name: safe_number(
            eligible[name]["benchmarks"]["realistic_single_stream"].get(
                "median_decode_tokens_per_s"
            ),
            float("-inf"),
        ),
    )
    return latency, throughput


def shell_join(values: list[str]) -> str:
    import shlex

    return " ".join(shlex.quote(value) for value in values)


def deployment_flags(candidate: Candidate) -> list[str]:
    flags = candidate.profile_flags()
    if candidate.draft_model_path:
        flags = [
            "__DRAFT_MODEL_PATH__" if value == candidate.draft_model_path else value
            for value in flags
        ]
    return flags


def write_selected_profiles(
    output_dir: Path,
    results: dict[str, Any],
    latency_name: str,
    throughput_name: str,
    candidates: list[Candidate],
) -> None:
    candidates_by_name = {candidate.name: candidate for candidate in candidates}
    latency = candidates_by_name[latency_name]
    throughput = candidates_by_name[throughput_name]
    latency_bench = results["candidates"][latency_name]["benchmarks"]["latency"]
    throughput_bench = results["candidates"][throughput_name]["benchmarks"]
    contents = (
        "# Generated by the measured RTX PRO 6000 autotune run.\n"
        f"LATENCY_CANDIDATE={latency_name}\n"
        f"LATENCY_FLAGS={json.dumps(shell_join(deployment_flags(latency)))}\n"
        f"THROUGHPUT_CANDIDATE={throughput_name}\n"
        f"THROUGHPUT_FLAGS={json.dumps(shell_join(deployment_flags(throughput)))}\n"
        f"MEASURED_LATENCY_TTFT_MS={latency_bench['median_ttft_ms']:.3f}\n"
        f"MEASURED_LATENCY_DECODE_TPS={latency_bench['median_decode_tokens_per_s']:.3f}\n"
        "MEASURED_SINGLE_STREAM_TPS="
        f"{throughput_bench['realistic_single_stream']['median_decode_tokens_per_s']:.3f}\n"
        "MEASURED_CONCURRENCY32_TPS="
        f"{throughput_bench['throughput_c32']['output_tokens_per_s']:.3f}\n"
    )
    (output_dir / "selected_profiles.env").write_text(contents)


def write_summary(output_dir: Path, results: dict[str, Any]) -> None:
    latency_name = results["selected"]["latency"]
    throughput_name = results["selected"]["throughput"]
    latency = results["candidates"][latency_name]
    throughput = results["candidates"][throughput_name]
    long_context = results.get("long_context", {})
    lines = [
        "# RTX PRO 6000 tuning result",
        "",
        f"- GPU: `{results['environment']['gpu']}`",
        f"- Image: `{results['environment']['image']}`",
        "- Model: `Qwen/Qwen3.8-27B-FP8`",
        "- Context configured: `262144` tokens",
        "- KV cache: `bfloat16` (quality-preserving)",
        "- GDN recurrent state: `float32` (quality-preserving)",
        f"- Lowest-TTFT candidate: `{latency_name}`",
        (
            "- Median TTFT / per-request decode: "
            f"`{latency['benchmarks']['latency']['median_ttft_ms']:.1f} ms` / "
            f"`{latency['benchmarks']['latency']['median_decode_tokens_per_s']:.1f} tok/s`"
        ),
        f"- Highest-throughput candidate: `{throughput_name}`",
        (
            "- Realistic single-stream decode / aggregate concurrency-32: "
            f"`{throughput['benchmarks']['realistic_single_stream']['median_decode_tokens_per_s']:.1f}` / "
            f"`{throughput['benchmarks']['throughput_c32']['output_tokens_per_s']:.1f} tok/s`"
        ),
        (
            "- Task-level quality gate: "
            f"`{throughput['quality']['correct_answers']}/{throughput['quality']['total']}` correct; "
            f"`{throughput['quality']['exact_matches']}/{throughput['quality']['total']}` byte-exact"
        ),
        (
            "- Near-limit context request: "
            f"`{'PASS' if long_context.get('passed') else 'FAIL'}` "
            f"({long_context.get('local_input_tokens', 'unknown')} input tokens)"
        ),
        "",
        "Full machine-readable measurements and server logs are adjacent to this file.",
    ]
    (output_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n")


def start_gpu_telemetry(output_dir: Path) -> tuple[subprocess.Popen[str] | None, Any | None]:
    if not shutil.which("nvidia-smi"):
        return None, None
    handle = (output_dir / "gpu-telemetry.csv").open("w", buffering=1)
    command = [
        "nvidia-smi",
        "--query-gpu=timestamp,utilization.gpu,memory.used,power.draw,temperature.gpu,clocks.sm",
        "--format=csv",
        "-l",
        "5",
    ]
    process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, text=True)
    return process, handle


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/model")
    parser.add_argument("--output", default="/results/latest")
    parser.add_argument("--budget-seconds", type=int, default=10620)
    parser.add_argument("--min-candidate-seconds", type=int, default=600)
    parser.add_argument(
        "--candidate-set",
        choices=("all", "mtp", "mtp_depth", "finalists", "advanced", "dspark"),
        default="all",
    )
    parser.add_argument("--draft-model-path")
    args = parser.parse_args()

    candidates = select_candidates(args.candidate_set, args.draft_model_path)
    if any(candidate.speculative_algorithm == "DSPARK" for candidate in candidates):
        if not args.draft_model_path:
            parser.error("--draft-model-path is required for a candidate set containing DSPARK")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    job_started = time.monotonic()
    deadline = job_started + args.budget_seconds
    results: dict[str, Any] = {
        "schema_version": 1,
        "started_unix": time.time(),
        "environment": {
            "gpu": gpu_snapshot(),
            "image": os.environ.get("BLACKWELL_QWEN_IMAGE", "lmsysorg/sglang:qwen38-27b"),
            "model_path": args.model_path,
            "draft_model_path": args.draft_model_path,
            "candidate_set": args.candidate_set,
            "python": sys.version,
        },
        "candidates": {},
    }
    atomic_json(output_dir / "results.json", results)
    telemetry, telemetry_handle = start_gpu_telemetry(output_dir)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    reference_quality: dict[str, Any] | None = None

    try:
        for candidate in candidates:
            if deadline - time.monotonic() < args.min_candidate_seconds:
                results["candidates"][candidate.name] = {
                    "candidate": asdict(candidate),
                    "status": "skipped_deadline",
                }
                continue
            process: subprocess.Popen[str] | None = None
            log_handle = None
            row: dict[str, Any] = {"candidate": asdict(candidate), "status": "starting"}
            results["candidates"][candidate.name] = row
            atomic_json(output_dir / "results.json", results)
            try:
                process, log_handle, startup_s = start_server(
                    args.model_path, candidate, output_dir
                )
                row["startup_s"] = startup_s
                current_quality = run_quality(BASE_URL, MODEL_NAME)
                if reference_quality is None and not candidate.speculative:
                    reference_quality = current_quality
                    quality = {
                        "quality_safe": current_quality["all_correct"],
                        "correct_answers": sum(
                            row["correct"] for row in current_quality["rows"]
                        ),
                        "exact_matches": len(current_quality["rows"]),
                        "total": len(current_quality["rows"]),
                        "reference": True,
                    }
                    atomic_json(output_dir / "quality-reference.json", current_quality)
                elif reference_quality is not None:
                    quality = compare_quality(current_quality, reference_quality)
                else:
                    quality = {"quality_safe": False, "error": "reference unavailable"}
                row["quality"] = quality
                atomic_json(output_dir / f"quality-{candidate.name}.json", current_quality)
                row["benchmarks"] = candidate_benchmarks(tokenizer)
                row["status"] = "completed"
            except Exception as exc:  # noqa: BLE001 - continue through the matrix
                row["status"] = "failed"
                row["error"] = f"{type(exc).__name__}: {exc}"
            finally:
                stop_server(process)
                if log_handle is not None:
                    log_handle.close()
                atomic_json(output_dir / "results.json", results)

        latency_name, throughput_name = pick_winners(results)
        results["selected"] = {"latency": latency_name, "throughput": throughput_name}
        write_selected_profiles(
            output_dir, results, latency_name, throughput_name, candidates
        )
        atomic_json(output_dir / "results.json", results)

        # Prove the native 256K window on the selected high-throughput kernel stack.
        winner = next(candidate for candidate in candidates if candidate.name == throughput_name)
        process = None
        log_handle = None
        try:
            process, log_handle, startup_s = start_server(args.model_path, winner, output_dir)
            results["winner_restart_s"] = startup_s
            results["long_context"] = run_long_context(BASE_URL, MODEL_NAME, tokenizer)
            atomic_json(output_dir / "results.json", results)

            # Spend the remaining rental on a thermal/stability soak of the measured winner.
            soak_rows = []
            while deadline - time.monotonic() > 240:
                soak_rows.append(
                    {
                        "elapsed_s": time.monotonic() - job_started,
                        "benchmark": run_concurrent(
                            BASE_URL,
                            MODEL_NAME,
                            tokenizer,
                            input_tokens=512,
                            output_tokens=256,
                            concurrency=16,
                            requests=32,
                        ),
                    }
                )
                results["soak"] = soak_rows
                atomic_json(output_dir / "results.json", results)
                time.sleep(15)
        finally:
            stop_server(process)
            if log_handle is not None:
                log_handle.close()

        results["finished_unix"] = time.time()
        results["runtime_s"] = time.monotonic() - job_started
        atomic_json(output_dir / "results.json", results)
        write_summary(output_dir, results)
        return 0 if results.get("long_context", {}).get("passed") else 2
    finally:
        if telemetry is not None:
            telemetry.terminate()
            try:
                telemetry.wait(timeout=10)
            except subprocess.TimeoutExpired:
                telemetry.kill()
        if telemetry_handle is not None:
            telemetry_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
