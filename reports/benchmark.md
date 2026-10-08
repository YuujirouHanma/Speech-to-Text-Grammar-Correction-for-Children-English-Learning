# Benchmark

Laptop: Intel i7-13620H, RTX 4070 Laptop (8 GB), Windows 11, on AC power.
Absolute numbers depend heavily on background load, so compare runtimes
**within one run**, not across runs.

## Run 2026-10-08: CPU vs GPU (t5-gec-v2, laptop busy with a browser open)

`python scripts/benchmark.py --backends ct2-cpu ct2-cuda torch-cpu torch-cuda --whisper base.en small.en --whisper-devices cpu cuda --audio ../recorded.wav`

GPU runs use the cuBLAS/cuDNN DLLs bundled with the PyTorch CUDA wheel (no CUDA toolkit installed).

### Grammar correction (50 JFLEG sentences, beam 5)

| runtime | backend | load_s | p50_ms | p95_ms | paragraph_8_sent_ms | rss_delta_mb |
|---|---|---|---|---|---|---|
| ct2-cpu | ctranslate2:int8_float32@cpu | 0.3 | 229 | 847 | 2935 | 280 |
| ct2-cuda | ctranslate2:int8_float32@cuda | 0.7 | 222 | 586 | 677 | 485 |
| torch-cpu | torch:float32@cpu | 9.0 | 2259 | 6436 | 21310 | 891 |
| torch-cuda | torch:bfloat16@cuda | 1.3 | 1058 | 2108 | 2377 | 850 |

A repeat of the CT2 rows gave ct2-cpu 322 / 938 / 2494 ms and ct2-cuda 224 / 572 / 762 ms
(p50 / p95 / paragraph). Single short sentences are bound by step-by-step beam decoding, so the GPU
mainly helps longer inputs (paragraph 3-4x faster).

### Speech-to-text (recorded.wav, 5 s)

| model | runtime | audio_s | median_s | real_time_factor |
|---|---|---|---|---|
| base.en | int8@cpu | 5.0 | 1.07 | 0.213 |
| small.en | int8@cpu | 5.0 | 3.23 | 0.645 |
| base.en | float16@cuda | 5.0 | 0.22 | 0.044 |
| small.en | float16@cuda | 5.0 | 0.18 | 0.035 |

Repeat: base.en 1.01 s (CPU) vs 0.13 s (GPU), small.en 3.69 s vs 0.18 s.

## Run 2026-10-07: CPU, idle laptop (t5-gec, original model; same architecture as v2)

### Grammar correction (50 JFLEG sentences, beam 5)

| runtime | backend | load_s | p50_ms | p95_ms | paragraph_8_sent_ms | rss_delta_mb |
|---|---|---|---|---|---|---|
| ct2-cpu | ctranslate2:int8_float32@cpu | 0.4 | 156 | 393 | 1065 | 279 |
| torch-cpu | torch:float32@cpu | 4.8 | 603 | 1776 | 7687 | 1903 |
| torch-cuda | torch:bfloat16@cuda | 0.7 | 363 | 956 | 1165 | 1079 |

### Speech-to-text (CPU int8, recorded.wav)

| model | audio_s | median_s | real_time_factor |
|---|---|---|---|
| base.en | 5.0 | 0.45 | 0.090 |
| small.en | 5.0 | 1.37 | 0.274 |
