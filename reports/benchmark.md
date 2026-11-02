# Benchmark

Laptop: Intel i7-13620H, RTX 4070 Laptop (8 GB), 16 GB RAM, Windows 11, 2026-11-02.
One job at a time; a browser and the editor stayed open, as on a normal day.
Absolute numbers depend on background load, so compare runtimes within one table.
GPU runs use the cuBLAS/cuDNN DLLs bundled with the PyTorch CUDA wheel.

## Grammar correction: model generation only

The fair comparison with the original PyTorch pipeline: no confidence filter, no error types.

`python scripts/benchmark.py --backends ct2-cpu ct2-cuda torch-cpu torch-cuda`

| runtime | backend | load_s | p50_ms | p95_ms | paragraph_8_sent_ms | rss_delta_mb |
|---|---|---|---|---|---|---|
| ct2-cpu | ctranslate2:int8_float32@cpu | 0.4 | 262 | 668 | 2142 | 279 |
| ct2-cuda | ctranslate2:int8_float32@cuda | 0.6 | 210 | 538 | 630 | 485 |
| torch-cpu | torch:float32@cpu | 6.0 | 1477 | 4093 | 17311 | 1677 |
| torch-cuda | torch:bfloat16@cuda | 1.1 | 747 | 1962 | 2259 | 893 |

## Grammar correction: as the service runs it

Default service settings: confidence filter 2.5 and ERRANT error types. `rss_delta_mb` includes
spaCy and ERRANT. In the CUDA row it is not isolated, because the CPU row ran first in the same
process and had already loaded them.

`python scripts/benchmark.py --service-config --backends ct2-cpu ct2-cuda`

| runtime | backend | load_s | p50_ms | p95_ms | paragraph_8_sent_ms | rss_delta_mb |
|---|---|---|---|---|---|---|
| ct2-cpu | ctranslate2:int8_float32@cpu | 2.7 | 371 | 1194 | 3397 | 886 |
| ct2-cuda | ctranslate2:int8_float32@cuda | 0.9 | 230 | 586 | 897 | 337 |

## Speech-to-text

`python scripts/benchmark.py --backends --whisper base.en small.en --whisper-devices cpu cuda --audio ../recorded.wav`

| model | runtime | audio_s | median_s | real_time_factor |
|---|---|---|---|---|
| base.en | int8@cpu | 5.0 | 0.43 | 0.085 |
| small.en | int8@cpu | 5.0 | 2.58 | 0.516 |
| base.en | float16@cuda | 5.0 | 0.07 | 0.015 |
| small.en | float16@cuda | 5.0 | 0.16 | 0.031 |

Models were benchmarked in two separate runs because this laptop ran out of commit memory when one
process loaded all of them (drive C: was full, so the page file could not grow).

## CTranslate2 compute types on GPU

150 JFLEG sentences, one at a time, beam 5, two rounds in opposite orders.

| compute type | median ms (round 1 / 2) | total s (round 1 / 2) |
|---|---|---|
| int8_float32 | 191 / 204 | 32.4 / 34.8 |
| float32 | 142 / 146 | 24.4 / 25.1 |
| int8_bfloat16 | 193 / 191 | 33.4 / 33.4 |

## Beam search vs greedy on CPU

The same 150 sentences, two rounds in opposite orders: beam 5 takes 1.78x as long as greedy.
