# Load test

`python scripts/load_test.py --audio ../recorded.wav ...` against a local server
(`python -m speakfix`, RTX 4070 Laptop, `SPEAKFIX_DEVICE=auto`, demo mode), 2026-10-08.
Every virtual user sends the 5 s recording, waits for the answer, pauses
("think") and repeats; each user has its own client IP (`X-Forwarded-For`).

## Clean runs (nothing else running on the laptop)

| users | think | workers | requests | ok/s | ok % | p50 s | p95 s | statuses |
|---|---|---|---|---|---|---|---|---|
| 15 | ~3 s | 1 | 201 | 4.47 | 100.0 | 0.24 | 0.61 | 200: 201 |
| 30 | ~3 s | 1 | 293 | 6.47 | 99.3 | 1.69 | 2.35 | 200: 291, 503: 2 |
| 15 | ~3 s | 2 | 205 | 4.56 | 100.0 | 0.19 | 0.36 | 200: 205 |
| 30 | ~3 s | 2 | 387 | 8.60 | 100.0 | 0.40 | 0.84 | 200: 387 |

## Runs while a CPU evaluation job was also running (1 worker)

The CPU-side work of each request (audio decoding, tokenization, ERRANT) competed
with the evaluation job, so these numbers are pessimistic.

| users | think | requests | ok/s | ok % | p50 s | p95 s | statuses |
|---|---|---|---|---|---|---|---|
| 5 | ~3 s | 64 | 1.42 | 100.0 | 0.25 | 0.68 | 200: 64 |
| 15 | ~3 s | 190 | 4.22 | 100.0 | 0.58 | 1.14 | 200: 190 |
| 30 | ~3 s | 234 | 3.31 | 63.7 | 4.98 | 5.47 | 200: 149, 503: 85 |
| 60 | ~3 s | 672 | 4.31 | 28.9 | 3.46 | 5.22 | 200: 194, 503: 478 |
| 30 | ~15 s | 122 | 2.03 | 100.0 | 0.25 | 0.64 | 200: 122 |
| 60 | ~15 s | 226 | 3.60 | 95.6 | 0.61 | 5.36 | 200: 216, 503: 10 |

## Thread safety with 2 workers

300 text corrections, 16 in flight at a time: all 300 answers (corrected text,
edits and ERRANT error types) were identical to the same requests sent one by one.
