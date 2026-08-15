from __future__ import annotations

import concurrent.futures
import http.client
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from typing import Any, Iterable
from urllib.parse import urlsplit


QUALITY_PROMPTS = [
    "What is 37 * 43? Reply with only the number.",
    "Write a Python function that returns the nth Fibonacci number iteratively.",
    "If all glibs are trobs and no trob is a flan, can any glib be a flan? Explain briefly.",
    "A train travels 120 km in 90 minutes. What is its average speed in km/h?",
    "Summarize the difference between TCP and UDP in two sentences.",
    "Return valid JSON with keys name and primes, where primes contains the first five primes.",
    "Solve 3x + 7 = 31. Reply with x and one verification step.",
    "Translate 'The weather is pleasant today' into French.",
    "Name the planet with the shortest year and state its orbital period approximately.",
    "Give one counterexample to the claim that every odd number is prime.",
    "Implement binary search in JavaScript without recursion.",
    "Which is larger: 2^10 or 10^3? Reply with both values.",
]


@dataclass
class RequestMetric:
    ok: bool
    ttft_s: float | None
    latency_s: float
    output_tokens: int
    text: str
    error: str | None = None


def percentile(values: Iterable[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _connection(base_url: str, timeout: float) -> tuple[http.client.HTTPConnection, str]:
    url = urlsplit(base_url)
    path = (url.path.rstrip("/") or "") + "/chat/completions"
    cls = http.client.HTTPSConnection if url.scheme == "https" else http.client.HTTPConnection
    return cls(url.hostname, url.port, timeout=timeout), path


def _completion_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        return ""
    choice = choices[0]
    delta = choice.get("delta") or choice.get("message") or {}
    return delta.get("content") or ""


def chat_request(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    *,
    stream: bool = True,
    ignore_eos: bool = False,
    timeout: float = 1800,
    enable_thinking: bool = False,
) -> RequestMetric:
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 42,
        "max_tokens": max_tokens,
        "stream": stream,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    if ignore_eos:
        body["ignore_eos"] = True
    if stream:
        body["stream_options"] = {"include_usage": True}

    started = time.perf_counter()
    first_token_at: float | None = None
    pieces: list[str] = []
    usage_tokens: int | None = None
    conn: http.client.HTTPConnection | None = None
    try:
        conn, path = _connection(base_url, timeout)
        conn.request(
            "POST",
            path,
            body=json.dumps(body),
            headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
        )
        response = conn.getresponse()
        if response.status >= 400:
            message = response.read().decode("utf-8", "replace")
            raise RuntimeError(f"HTTP {response.status}: {message[:1000]}")

        if stream:
            while True:
                raw = response.readline()
                if not raw:
                    break
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data or data == "[DONE]":
                    continue
                event = json.loads(data)
                usage = event.get("usage") or {}
                if usage.get("completion_tokens") is not None:
                    usage_tokens = int(usage["completion_tokens"])
                text = _completion_text(event)
                if text:
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                    pieces.append(text)
        else:
            event = json.loads(response.read())
            pieces.append(_completion_text(event))
            usage = event.get("usage") or {}
            if usage.get("completion_tokens") is not None:
                usage_tokens = int(usage["completion_tokens"])
            first_token_at = time.perf_counter()

        finished = time.perf_counter()
        text = "".join(pieces)
        output_tokens = usage_tokens if usage_tokens is not None else max(1, len(text.split()))
        return RequestMetric(
            ok=True,
            ttft_s=(first_token_at - started) if first_token_at is not None else None,
            latency_s=finished - started,
            output_tokens=output_tokens,
            text=text,
        )
    except Exception as exc:  # noqa: BLE001 - failures are benchmark data
        return RequestMetric(
            ok=False,
            ttft_s=None,
            latency_s=time.perf_counter() - started,
            output_tokens=0,
            text="",
            error=f"{type(exc).__name__}: {exc}",
        )
    finally:
        if conn is not None:
            conn.close()


def run_quality(
    base_url: str,
    model: str,
    prompts: list[str] | None = None,
    max_tokens: int = 96,
) -> dict[str, Any]:
    prompts = prompts or QUALITY_PROMPTS
    rows = []
    for prompt in prompts:
        metric = chat_request(
            base_url,
            model,
            prompt,
            max_tokens,
            stream=False,
            enable_thinking=False,
        )
        rows.append({"prompt": prompt, **asdict(metric)})
    return {"all_ok": all(row["ok"] for row in rows), "rows": rows}


def compare_quality(candidate: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    candidate_rows = candidate["rows"]
    reference_rows = reference["rows"]
    comparisons = []
    for current, expected in zip(candidate_rows, reference_rows, strict=True):
        exact = current["ok"] and expected["ok"] and current["text"] == expected["text"]
        comparisons.append(
            {
                "prompt": current["prompt"],
                "exact": exact,
                "candidate": current["text"],
                "reference": expected["text"],
            }
        )
    return {
        "exact_matches": sum(row["exact"] for row in comparisons),
        "total": len(comparisons),
        "quality_safe": all(row["exact"] for row in comparisons),
        "comparisons": comparisons,
    }


def make_prompt(target_tokens: int, tokenizer: Any) -> str:
    sentence = (
        "Blackwell inference benchmark context with deterministic words and stable tokenization. "
    )
    repeats = max(2, target_tokens // 9 + 16)
    ids = tokenizer.encode(sentence * repeats, add_special_tokens=False)
    while len(ids) < target_tokens:
        repeats *= 2
        ids = tokenizer.encode(sentence * repeats, add_special_tokens=False)
    return tokenizer.decode(ids[:target_tokens], skip_special_tokens=True)


def run_concurrent(
    base_url: str,
    model: str,
    tokenizer: Any,
    *,
    input_tokens: int,
    output_tokens: int,
    concurrency: int,
    requests: int,
) -> dict[str, Any]:
    prompt = make_prompt(input_tokens, tokenizer)
    # A warm request keeps graph capture and one-time tokenizer work out of the measurement.
    warm = chat_request(
        base_url,
        model,
        prompt,
        min(16, output_tokens),
        stream=True,
        ignore_eos=True,
    )
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(
                chat_request,
                base_url,
                model,
                prompt,
                output_tokens,
                stream=True,
                ignore_eos=True,
            )
            for _ in range(requests)
        ]
        rows = [future.result() for future in futures]
    wall_s = time.perf_counter() - started
    successful = [row for row in rows if row.ok]
    ttfts = [row.ttft_s for row in successful if row.ttft_s is not None]
    latencies = [row.latency_s for row in successful]
    generated = sum(row.output_tokens for row in successful)
    decode_rates = [
        row.output_tokens / (row.latency_s - row.ttft_s)
        for row in successful
        if row.ttft_s is not None and row.latency_s > row.ttft_s
    ]
    return {
        "input_tokens_requested": input_tokens,
        "output_tokens_requested": output_tokens,
        "concurrency": concurrency,
        "requests": requests,
        "successful": len(successful),
        "failed": requests - len(successful),
        "wall_s": wall_s,
        "output_tokens": generated,
        "output_tokens_per_s": generated / wall_s if wall_s else 0.0,
        "median_ttft_ms": statistics.median(ttfts) * 1000 if ttfts else math.nan,
        "p90_ttft_ms": percentile(ttfts, 0.90) * 1000,
        "median_latency_s": statistics.median(latencies) if latencies else math.nan,
        "median_decode_tokens_per_s": statistics.median(decode_rates) if decode_rates else math.nan,
        "warmup_ok": warm.ok,
        "errors": [row.error for row in rows if not row.ok],
    }


def make_long_context_prompt(tokenizer: Any, total_input_tokens: int = 262080) -> tuple[str, int]:
    messages = [{"role": "user", "content": ""}]
    empty = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, enable_thinking=False
    )
    content_target = max(1, total_input_tokens - len(empty))
    content = make_prompt(content_target, tokenizer)
    for _ in range(4):
        full = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        difference = len(full) - total_input_tokens
        if abs(difference) <= 2:
            return content, len(full)
        content_ids = tokenizer.encode(content, add_special_tokens=False)
        new_length = max(1, len(content_ids) - difference)
        content = tokenizer.decode(content_ids[:new_length], skip_special_tokens=True)
    full = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return content, len(full)


def run_long_context(
    base_url: str,
    model: str,
    tokenizer: Any,
    total_input_tokens: int = 262080,
) -> dict[str, Any]:
    prompt, measured_input_tokens = make_long_context_prompt(tokenizer, total_input_tokens)
    metric = chat_request(
        base_url,
        model,
        prompt,
        8,
        stream=True,
        ignore_eos=True,
        timeout=3600,
        enable_thinking=False,
    )
    return {
        "target_input_tokens": total_input_tokens,
        "local_input_tokens": measured_input_tokens,
        "passed": metric.ok and measured_input_tokens >= 262000 and metric.output_tokens >= 1,
        "request": asdict(metric),
    }
