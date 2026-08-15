# blackwell-qwen

Measured, quality-gated FP8 serving for
[`Qwen/Qwen3.8-27B-FP8`](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) on one
96 GB NVIDIA RTX PRO 6000 Blackwell.

The repository tunes the kernel stack instead of changing the model: FlashInfer versus Triton
attention, prefill chunk sizes, CUDA graph coverage, and the checkpoint's native MTP draft head.
Every promoted profile keeps the native 262,144-token window, BF16 KV cache, and FP32 GDN state.
MTP is promoted only after exact greedy-output parity against non-speculative inference.

## Measured result

The single-GPU Hugging Face run is in progress. This section and
`config/selected_profiles.env` are replaced with the measured winners when it completes.

## Why these quality constraints

- The official checkpoint is already block-scaled FP8; the repository does not requantize it.
- KV cache remains BF16. FP8 KV can save memory, but it is intentionally excluded because it can
  change model quality.
- GDN recurrent state remains FP32, matching the model configuration.
- Speculative decoding uses the in-checkpoint MTP head and is verified against the target model.
  It is accepted only if all deterministic quality probes match the non-speculative reference.
- The 256K claim is tested with a near-limit 262,080-token input plus generation, rather than only
  setting a command-line flag.

## Run on your RTX PRO 6000

Install a recent NVIDIA driver and Docker with NVIDIA Container Toolkit, then download the model
locally:

```bash
hf download Qwen/Qwen3.8-27B-FP8 --local-dir ./model
```

Build and launch the measured throughput profile:

```bash
docker compose build
PROFILE=throughput MODEL_DIR="$PWD/model" docker compose up
```

For lowest time to first token, use `PROFILE=latency`. Both expose an OpenAI-compatible endpoint:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen3.8-27B-FP8",
    "messages": [{"role": "user", "content": "Explain tensor cores briefly."}],
    "temperature": 0.7,
    "top_p": 0.8,
    "max_tokens": 256,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

You can also run directly inside `lmsysorg/sglang:qwen38-27b`:

```bash
PROFILE=throughput MODEL_PATH=/path/to/Qwen3.8-27B-FP8 bash scripts/serve.sh
```

## Reproduce the three-hour tuner

The launch script enforces one GPU, checks that no other Job is running, creates a private results
bucket, and sets the three-hour hardware timeout. At the current Hugging Face price of $2.75/hour,
the maximum compute charge is $8.25.

```bash
hf auth login
bash scripts/launch_hf_job.sh
```

The job mounts the public model repository and stages it once onto local NVMe, avoiding repeated
downloads and network-filesystem faults during concurrent safetensors reads. It benchmarks
eight kernel configurations, rejects candidates that fail exact quality parity, tests the selected
stack at 262K context, then spends the remaining rental on a sustained thermal/stability soak.

Fetch the artifacts with:

```bash
bash scripts/fetch_results.sh
```

## Tuning matrix

Each candidate uses the same correctness-sensitive settings:

```text
model context       262144
KV cache            bfloat16
GDN recurrent state float32
GDN radix strategy  extra_buffer_lazy
GPU memory fraction 0.90
```

The measured axes are:

- FlashInfer and Triton attention on SM120.
- Auto, CUTLASS, Triton, and FlashInfer TRT-LLM block-FP8 GEMM runners.
- 1K, 2K, 4K, and 8K chunked prefill.
- Native EAGLE/MTP with 3 steps, top-k 1, and 4 draft tokens, plus a non-speculative baseline.
- Batch-1 streaming TTFT/decode rate and aggregate concurrency-16/32 output throughput.

## License

Apache-2.0.
