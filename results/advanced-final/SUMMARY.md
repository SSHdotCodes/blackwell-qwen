# RTX PRO 6000 tuning result

- GPU: `NVIDIA RTX PRO 6000 Blackwell Server Edition, GPU-b11b1ed3-cbfd-b494-d39b-e6cddeab7266, 580.159.03, 97887 MiB, 600.00 W, 2430 MHz, 12.0`
- Image: `lmsysorg/sglang:qwen38-27b`
- Model: `Qwen/Qwen3.8-27B-FP8`
- Context configured: `262144` tokens
- KV cache: `bfloat16` (quality-preserving)
- GDN recurrent state: `float32` (quality-preserving)
- Lowest-TTFT candidate: `flashinfer_cutlass_mtp_s2_decode_p1`
- Median TTFT / per-request decode: `59.8 ms` / `102.2 tok/s`
- Highest-throughput candidate: `flashinfer_cutlass_mtp_s3_decode_p1`
- Realistic single-stream decode / aggregate concurrency-32: `87.7` / `1264.3 tok/s`
- Task-level quality gate: `12/12` correct; `9/12` byte-exact
- Near-limit context request: `PASS` (262080 input tokens)

Full machine-readable measurements and server logs are adjacent to this file.
