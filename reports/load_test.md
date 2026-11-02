# Load test

`python scripts/load_test.py --audio ../recorded.wav ...` against a real local server, RTX 4070
Laptop, 2026-11-02, on the final code. Every virtual user sends the 5 s recording, waits for the
answer, pauses ("think") and repeats; each user has its own client IP (`X-Forwarded-For`) unless
noted. One configuration at a time, with no other job of ours running.

## Public demo configuration

GPU, `small.en`, 2 workers, rate limit 600/min, 32 pending requests (what
`scripts/run_public_demo.ps1` starts).

| users | think | requests | ok/s | ok % | p50 s | p95 s | statuses |
|---|---|---|---|---|---|---|---|
| 30 | ~15 s | 121 | 2.02 | 100.0 | 0.35 | 0.64 | 200: 121 |
| 30 | ~15 s (one shared IP) | 239 | 1.99 | 100.0 | 0.37 | 0.66 | 200: 239 |
| 30 | ~3 s | 248 | 5.51 | 100.0 | 2.66 | 3.28 | 200: 248 |
| 60 | ~15 s | 231 | 3.85 | 100.0 | 0.53 | 1.47 | 200: 231 |

A whole class pressing the button at the same moment (one shared IP):

- burst 30 users: {200: 30} | ok: median 3.52s, slowest 6.16s
- burst 30 users: {200: 30} | ok: median 3.38s, slowest 6.06s
- burst 15 users: {200: 15} | ok: median 1.94s, slowest 3.18s

Server: peak 1308 MB RAM (working set), 2989 MB commit, 31 threads. Lowest free commit on the laptop: 3.2 GB.
GPU memory before the server started: 2270 MiB. GPU total: 3283 MiB, 8188 MiB.
Server log: 922 lines, 0 errors, 0 warnings, 0 HTTP 500, 0 HTTP 503, 0 lines with sentence content.

## One worker (comparison)

GPU, `small.en`, 1 worker, rate limit 600/min, 32 pending requests.

| users | think | requests | ok/s | ok % | p50 s | p95 s | statuses |
|---|---|---|---|---|---|---|---|
| 30 | ~3 s | 190 | 4.22 | 100.0 | 4.75 | 5.52 | 200: 190 |
| 30 | ~15 s | 126 | 2.10 | 100.0 | 0.30 | 0.71 | 200: 126 |
| 60 | ~15 s | 226 | 3.77 | 100.0 | 1.42 | 2.52 | 200: 226 |

Server: peak 1309 MB RAM (working set), 3041 MB commit, 19 threads. Lowest free commit on the laptop: 3.2 GB.
Server log: 549 lines, 0 errors, 0 warnings, 0 HTTP 500, 0 HTTP 503, 0 lines with sentence content.

## Default settings

GPU, `small.en`, 1 worker, rate limit 60/min per IP, 16 pending requests.

| users | think | requests | ok/s | ok % | p50 s | p95 s | statuses |
|---|---|---|---|---|---|---|---|
| 30 | ~15 s | 118 | 1.97 | 100.0 | 0.43 | 0.92 | 200: 118 |
| 30 | ~10 s (one shared IP) | 179 | 1.98 | 66.5 | 0.47 | 1.15 | 200: 119, 429: 60 |

Server: peak 1297 MB RAM (working set), 3026 MB commit, 18 threads. Lowest free commit on the laptop: 3.3 GB.
Server log: 304 lines, 0 errors, 0 warnings, 0 HTTP 500, 0 HTTP 503, 0 lines with sentence content.

## Docker image contents

No PyTorch, CPU, `base.en`, 1 worker, 16 pending requests, default rate limit (60/min per IP).
Same packages and settings as the CPU image, installed in a clean virtual environment on Windows;
the image itself has not been built.

| users | think | requests | ok/s | ok % | p50 s | p95 s | statuses |
|---|---|---|---|---|---|---|---|
| 5 | ~15 s | 19 | 0.32 | 100.0 | 1.32 | 2.33 | 200: 19 |
| 10 | ~15 s | 38 | 0.63 | 100.0 | 1.45 | 2.93 | 200: 38 |
| 20 | ~15 s | 60 | 1.00 | 100.0 | 4.88 | 8.37 | 200: 60 |

Server: peak 718 MB RAM (working set), 3573 MB commit, 22 threads. Lowest free commit on the laptop: 2.4 GB.
Server log: 282 lines, 0 errors, 1 warnings, 0 HTTP 500, 0 HTTP 503, 0 lines with sentence content.

## What changed before these numbers

- **Decoder.** Audio used to go through `av.AudioResampler`, whose filter graph starts one thread per
  CPU core for every file. Under load the server reached 944 threads and failed with memory errors
  (3 HTTP 500 at 60 users). The filter graph now has `threads = 1`; samples are identical on 23 test
  files.
- **OpenBLAS.** NumPy's OpenBLAS started 15 unused threads and committed about 500 MB. The service now
  sets `OPENBLAS_NUM_THREADS=1`.
- **Memory errors.** All out-of-memory failures now answer 503, including those from onnxruntime.

## The laptop's memory limit

Windows counts commit memory, not only used RAM, and CTranslate2 commits far more than it touches: about
2.1 GB for the 224 MB T5 model and 1.0 GB for Whisper base.en. Drive C: had only 0.2 GB free during these
tests, so the page file could not grow. In an earlier run, before the OpenBLAS limit and with more going on
on the laptop, free commit fell to 0.04 GB and the CPU configuration answered 2 HTTP 500 and 3 HTTP 503 out of
146 requests. Free up C: and restart before relying on these numbers on another day.
