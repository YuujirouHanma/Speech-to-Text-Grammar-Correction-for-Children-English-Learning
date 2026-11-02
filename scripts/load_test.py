"""Load test: how many simultaneous users can one SpeakFix instance serve?

Each virtual user repeatedly sends a recording (or a sentence), waits for the
answer, pauses ("think time") and starts again. Users get distinct client IPs
through ``X-Forwarded-For`` (honoured from 127.0.0.1), so the per-client rate
limit applies as it would to real visitors behind a tunnel or reverse proxy.
``--same-ip`` models a classroom instead: everyone behind one school Wi-Fi address.
``--burst`` sends one request per user at the same moment, like a class pressing the button.

Run it against a server on the same machine, e.g.:
    python -m speakfix                     # in another terminal
    python scripts/load_test.py --users 10 30 60 --audio ../recorded.wav --seconds 60
    python scripts/load_test.py --users 30 --audio ../recorded.wav --same-ip --burst
"""

from __future__ import annotations

import argparse
import asyncio
import random
import statistics
import time
from collections import Counter
from pathlib import Path

import httpx

SENTENCES = [
    "She go to school every day.",
    "He don't like apples.",
    "They is here today.",
    "My mother cook rice every morning.",
    "I have went to Bali last year.",
    "The cat sleep on the sofa.",
]


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]


async def send(
    client: httpx.AsyncClient,
    ip: str,
    audio: bytes | None,
    latencies: list[float],
    statuses: Counter[int],
) -> None:
    headers = {"X-Forwarded-For": ip}
    start = time.perf_counter()
    try:
        if audio is not None:
            response = await client.post(
                "/v1/audio/analyze",
                headers=headers,
                files={"file": ("clip.wav", audio, "audio/wav")},
                data={"consent": "true"},
            )
        else:
            response = await client.post(
                "/v1/text/correct", headers=headers, json={"text": random.choice(SENTENCES)}
            )
        statuses[response.status_code] += 1
        if response.status_code == 200:
            latencies.append(time.perf_counter() - start)
    except httpx.HTTPError:
        statuses[0] += 1  # connection error / client timeout


async def user(
    client: httpx.AsyncClient,
    ip: str,
    deadline: float,
    audio: bytes | None,
    think: float,
    latencies: list[float],
    statuses: Counter[int],
) -> None:
    await asyncio.sleep(random.uniform(0, think))  # users do not all start at once
    while time.perf_counter() < deadline:
        await send(client, ip, audio, latencies, statuses)
        await asyncio.sleep(think * random.uniform(0.5, 1.5))


async def scenario(
    url: str, users: int, seconds: float, audio: bytes | None, think: float, same_ip: bool
) -> dict:
    latencies: list[float] = []
    statuses: Counter[int] = Counter()
    limits = httpx.Limits(max_connections=users + 10)
    async with httpx.AsyncClient(base_url=url, timeout=60.0, limits=limits) as client:
        deadline = time.perf_counter() + seconds
        await asyncio.gather(
            *(
                user(
                    client,
                    "10.0.0.1" if same_ip else f"10.0.{i // 250}.{i % 250 + 1}",
                    deadline,
                    audio,
                    think,
                    latencies,
                    statuses,
                )
                for i in range(users)
            )
        )
    total = sum(statuses.values())
    ok = statuses.get(200, 0)
    return {
        "users": users,
        "requests": total,
        "ok_per_s": round(ok / seconds, 2),
        "ok_%": round(100 * ok / total, 1) if total else 0.0,
        "p50_s": round(statistics.median(latencies), 2) if latencies else None,
        "p95_s": round(percentile(latencies, 0.95), 2) if latencies else None,
        "statuses": dict(sorted(statuses.items())),
    }


async def burst(url: str, users: int, audio: bytes | None, same_ip: bool) -> dict:
    latencies: list[float] = []
    statuses: Counter[int] = Counter()
    limits = httpx.Limits(max_connections=users + 10)
    async with httpx.AsyncClient(base_url=url, timeout=120.0, limits=limits) as client:
        await asyncio.gather(
            *(
                send(
                    client,
                    "10.0.0.1" if same_ip else f"10.0.{i // 250}.{i % 250 + 1}",
                    audio,
                    latencies,
                    statuses,
                )
                for i in range(users)
            )
        )
    return {
        "users": users,
        "median_s": round(statistics.median(latencies), 2) if latencies else None,
        "slowest_s": round(max(latencies), 2) if latencies else None,
        "statuses": dict(sorted(statuses.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--users", type=int, nargs="+", default=[5, 15, 30])
    parser.add_argument("--seconds", type=float, default=45.0, help="duration of each scenario")
    parser.add_argument("--think", type=float, default=3.0, help="mean pause between requests")
    parser.add_argument("--audio", type=Path, help="recording to send (default: text requests)")
    parser.add_argument("--same-ip", action="store_true", help="all users share one client IP")
    parser.add_argument("--burst", action="store_true", help="one request per user, all at once")
    args = parser.parse_args()

    audio = args.audio.read_bytes() if args.audio else None
    kind = f"audio ({args.audio.name})" if args.audio else "text"
    network = "one shared IP" if args.same_ip else "one IP per user"
    if args.burst:
        print(f"{kind}, everyone at the same moment, {network}")
        for users in args.users:
            print(asyncio.run(burst(args.url, users, audio, args.same_ip)), flush=True)
        return
    print(f"{kind}, {args.seconds:.0f}s per scenario, think time ~{args.think:.0f}s, {network}")
    for users in args.users:
        result = asyncio.run(
            scenario(args.url, users, args.seconds, audio, args.think, args.same_ip)
        )
        print(result, flush=True)


if __name__ == "__main__":
    main()
