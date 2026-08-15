# blackwell-qwen

Measured, quality-gated FP8 serving for
[`Qwen/Qwen3.8-27B-FP8`](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) on one
96 GB NVIDIA RTX PRO 6000 Blackwell.

The repository tunes the kernel stack instead of changing the target model: FlashInfer versus
Triton attention, prefill chunk sizes, CUDA graph coverage, the checkpoint's native MTP draft
head, and the target-specific 1.36B-parameter
[`RadixArk/Qwen3.8-27B-DSpark`](https://huggingface.co/RadixArk/Qwen3.8-27B-DSpark)
draft model (the DFlash-family path).
Every promoted profile keeps the native 262,144-token window, BF16 KV cache, and FP32 GDN state.
Speculative profiles are promoted only after all task-level correctness probes pass; byte-exact
parity is retained as a stricter diagnostic, not a rejection criterion.

## Measured result

Measured on an NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB) with the pinned SGLang
Qwen3.8 image. Values are medians across six streaming requests; the natural workload uses a
108-token technical prompt and 512 generated tokens, while stress uses 1,024 repeated input
tokens and 512 generated tokens.

| Profile | Natural single-stream | 1K stress single-stream | Natural TTFT | Concurrency-32 |
|---|---:|---:|---:|---:|
| Non-speculative reference | 45.96 tok/s | 45.96 tok/s | 64.96 ms | 1,087.1 tok/s |
| `latency` — MTP depth 2 | 80.61 tok/s | **102.38 tok/s** | **54.95 ms** | **1,344.5 tok/s** |
| `throughput` — MTP depth 3 | **87.70 tok/s** | 97.40 tok/s | 57.36 ms | 1,265.1 tok/s |

The throughput profile is 1.91x the natural single-stream baseline. All three final profiles
answered 12/12 task-level quality probes correctly. The two MTP profiles were byte-identical on
9/12; the differences were semantically equivalent wording or formatting. A second clean run
reproduced the finalist rates within 0.1%.

The selected depth-3 profile passed a request with **262,080 measured input tokens plus 8 output
tokens**. Near-limit TTFT was 105.08 seconds. BF16 KV and FP32 GDN state were retained.

### DSpark / DFlash-family result

The trained 1.36B DSpark checkpoint was tested, not assumed. Its best 1K-stress result was
61.23 tok/s, 72.34 ms TTFT, and 550.0 tok/s at concurrency 32; all 12 correctness probes passed.
Its average accepted block collapsed on the repeated stress output, while real task probes showed
bursts up to roughly 153 tok/s. FA3 is not available on SM120 in the pinned image, and loading the
BF16 draft reduced its allocated KV pool below 262K. DSpark is therefore included for
experimentation but is not eligible as a default; native MTP is faster, uses less memory, and
passes the full-context requirement here.

## Why these quality constraints

- The official checkpoint is already block-scaled FP8; the repository does not requantize it.
- KV cache remains BF16. FP8 KV can save memory, but it is intentionally excluded because it can
  change model quality.
- GDN recurrent state remains FP32, matching the model configuration.
- Speculative decoding uses either the in-checkpoint MTP head or the target-specific DSpark
  draft and is verified against the target model.
  It is accepted only if all deterministic task-level correctness probes pass.
- The 256K claim is tested with a near-limit 262,080-token input plus generation, rather than only
  setting a command-line flag.

## Run on your RTX PRO 6000

Install a recent NVIDIA driver and Docker with NVIDIA Container Toolkit, then download the target
model locally. The DSpark download is optional and only needed to rerun that part of the sweep:

```bash
hf download Qwen/Qwen3.8-27B-FP8 --local-dir ./model
# Optional:
hf download RadixArk/Qwen3.8-27B-DSpark --local-dir ./draft
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

The job mounts both public model repositories and stages them once onto local NVMe, avoiding
repeated downloads and network-filesystem faults during concurrent safetensors reads. It
benchmarks the kernel/speculator matrix, rejects candidates that fail task-level correctness,
tests the selected stack at 262K context, then spends the remaining rental on a sustained
thermal/stability soak.

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
- Native EAGLE/MTP with 3 steps, top-k 1, and 4 draft tokens.
- A focused native-MTP depth sweep from 2 through 7 verification steps.
- Target-specific DSpark/DFlash-family decoding with a trained 7-token draft block.
- A non-speculative, checkpoint-precision reference baseline.
- Batch-1 streaming TTFT/decode rate on both a 1K synthetic stress prompt and a natural
  long-form response, plus aggregate concurrency-16/32 output throughput.

## License

Apache-2.0.
