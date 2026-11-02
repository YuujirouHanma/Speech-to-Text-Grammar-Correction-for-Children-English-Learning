"""End-to-end check of a running SpeakFix server, over real HTTP.

Sends the team's recordings to the server and compares the results with the Expected Output of
``FP/WHISPER DENGAN INPUT AUDIO.docx``, then checks the audio formats browsers send, every
endpoint, the error paths, unusual inputs and the per-visitor rate limit. Prints PASS/FAIL per
check, writes everything to a JSON report and exits with status 1 if a check failed.

    python -m speakfix                     # in another terminal
    python scripts/check_api.py http://127.0.0.1:8000 --out reports/check_api.json

Through a tunnel (scripts/run_public_demo.ps1) pass ``--skip-rate-limit``: the tunnel sets the
client IP itself, so this script cannot pretend to be different visitors.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import statistics
import sys
import time
import wave
from pathlib import Path
from typing import Any

import av
import httpx
import numpy as np

# Recording (start of the file name) -> Expected Output and the old result, both from the document
# above. The sentences in EXTRA are not in that document.
TEAM = [
    (
        "Although she studied hardly",
        "Although she studied hard for the exam, she didn't pass and she was very disappointed "
        "with the results.",
        "Although she studied hardly for the exam but she didn't pass and she was very "
        "disappointed about the results .",
    ),
    (
        "The peoples who lives",
        "The people who live in the city have many problems with pollution and traffic that is "
        "getting worse.",
        "The people who live in the city have many problems with pollution and traffic that is "
        "getting worse.",
    ),
    (
        "He was late because",
        "He was late because he forgot his keys and then he realized that he left them on the "
        "table in the kitchen.",
        "He was late because he forgot his keys and then realized that he left it on the table "
        "in the kitchen .",
    ),
    (
        "In the meeting yesterday",
        "In the meeting yesterday, he said that the company is planning to expand their services "
        "in other countries.",
        "In the meeting yesterday, he told that the company are planning to expand their "
        "services in other countries.",
    ),
    (
        "Many students finds",
        "Many students find it difficult to concentrate in online classes because of "
        "distractions and lack of motivation.",
        "Many students find it difficult to concentrate in online classes because of "
        "distractions and lack of motivations .",
    ),
    (
        "She don",
        "She doesn't have enough experience, so they didn't hire her for the job.",
        "She don't have enough experience, so they didn't hire her for the job.",
    ),
    (
        "The team were",
        "The team did not perform well last game because they weren't properly prepared.",
        "The team did not perform well last game because they didn't prepare properly .",
    ),
    (
        "I am agree",
        "I agree with your opinion, but we need more evidence before making a decision.",
        "I agree with your opinion, but we need more evidence before taking a decision.",
    ),
    (
        "He want to know",
        "He wants to know what is meant by artificial intelligence in modern technologies.",
        "He wants to know what do mean by artificial intelligences in modern technologies .",
    ),
    (
        "This kind of mistakes",
        "This kind of mistake happens when we are not focused during the test.",
        "This kind of mistakes happens when we are not focused during the test",
    ),
]
EXTRA = [
    ("they is here today", "They are here today."),
    ("she go to school", "She goes to school."),
    ("He readed the book", "He read the book."),
]
FRAGMENTED_MP4 = {
    "movflags": "frag_keyframe+empty_moov+default_base_moof"
}  # Safari's MediaRecorder
FORMATS = [
    ("m4a (AAC, file dari iPhone)", "ipod", "aac", 44100, "audio/mp4", "clip.m4a", None),
    (
        "mp4 terfragmentasi (AAC, MediaRecorder Safari)",
        "mp4",
        "aac",
        44100,
        "audio/mp4",
        "recording.m4a",
        FRAGMENTED_MP4,
    ),
    ("webm (Opus, Chrome/Firefox)", "webm", "libopus", 48000, "audio/webm", "clip.webm", None),
    ("ogg (Opus)", "ogg", "libopus", 48000, "audio/ogg", "clip.ogg", None),
    ("flac", "flac", "flac", 44100, "audio/flac", "clip.flac", None),
]
UNUSUAL_INPUTS = [
    "Saya pergi ke sekolah setiap hari.",
    "aku suka makan nasi goreng and I likes apples",
    "He don't like 🍎 apples.",
    "I bought 3.5 kg of rice at 7.30 a.m. yesterday.",
    'He said: "she go home" (yesterday).',
    "asdf qwer zxcv",
    "!!! ??? ...",
    "12345",
    "um uh I I go go to to school",
    "SHE GO TO SCHOOL",
    "she dont like apple",
    "The quick brown fox jumps over the lazy dog.",
    "My name is Budi and I live in Surabaya.",
    "Yesterday I go to the market with my father. My mother cook rice. We eats together.",
    "Dr. Smith don't like it. Mr. Budi go home.",
    "A",
]


class Server:
    """The requests this script sends, each as a visitor with its own IP."""

    def __init__(self, url: str) -> None:
        self.client = httpx.Client(base_url=url, timeout=120.0)
        self.failed: list[str] = []

    def check(self, name: str, ok: bool, detail: object = "") -> None:
        """Print one PASS/FAIL line and remember failures."""
        suffix = f"  -> {detail}" if detail != "" else ""
        print(f"{'PASS' if ok else 'FAIL'}  {name}{suffix}", flush=True)
        if not ok:
            self.failed.append(name)

    def analyze(self, data: bytes, name: str, mime: str, ip: str = "10.20.0.1") -> httpx.Response:
        """POST /v1/audio/analyze with consent."""
        return self.client.post(
            "/v1/audio/analyze",
            files={"file": (name, data, mime)},
            data={"consent": "true"},
            headers={"X-Forwarded-For": ip},
        )

    def correct(self, text: str, ip: str = "10.20.0.2") -> httpx.Response:
        """POST /v1/text/correct."""
        return self.client.post(
            "/v1/text/correct", json={"text": text}, headers={"X-Forwarded-For": ip}
        )


def normalize(text: str) -> str:
    text = text.replace("’", "'").strip()
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    return " ".join(text.split())


def error_code(response: httpx.Response) -> str:
    try:
        return response.json()["error"]["code"]
    except (ValueError, KeyError, TypeError):
        return f"HTTP {response.status_code}"


def recording_row(body: dict[str, Any], spoken: str, expected: str) -> dict[str, Any]:
    correction = body["correction"] or {}
    return {
        "spoken": spoken,
        "transcript": body["transcript"]["text"],
        "corrected": correction.get("corrected"),
        "changed": correction.get("changed"),
        "warnings": list(dict.fromkeys([*body["warnings"], *correction.get("warnings", [])])),
        "edits": [
            (e["original"], e["corrected"], e["error_type"]) for e in correction.get("edits", [])
        ],
        "total_ms": body["timings_ms"]["total_ms"],
        "expected": expected,
        "new_exact": normalize(correction.get("corrected") or "") == normalize(expected),
    }


def check_recordings(server: Server, fp_dir: Path, voice: Path) -> list[dict[str, Any]]:
    print("\n=== Rekaman tim lewat jalur suara (Whisper + T5) ===", flush=True)
    mp3s = {p.name: p for p in fp_dir.glob("*.mp3")}

    def mp3_starting_with(start: str) -> Path:
        return next(p for name, p in mp3s.items() if name.startswith(start))

    rows = []
    for start, expected, old in TEAM:
        path = mp3_starting_with(start)
        body = server.analyze(path.read_bytes(), "clip.mp3", "audio/mpeg").json()
        row = recording_row(body, path.stem.strip(), expected)
        row["old_result"] = old
        row["old_exact"] = normalize(old) == normalize(expected)
        rows.append(row)
    for start, expected in EXTRA:
        path = mp3_starting_with(start)
        body = server.analyze(path.read_bytes(), "clip.mp3", "audio/mpeg").json()
        rows.append(recording_row(body, path.stem.strip(), expected))
    for wav, spoken, expected in (
        (voice, "She can sings well.", "She can sing well."),
        (fp_dir / "recorded.wav", "She go to school.", "She goes to school."),
    ):
        body = server.analyze(wav.read_bytes(), "clip.wav", "audio/wav").json()
        rows.append(recording_row(body, spoken, expected))
    for row in rows:
        mark = "=" if row["new_exact"] else ("~" if row["changed"] else "x")
        print(f"[{mark}] {row['total_ms']:>6.0f} ms  {row['spoken'][:58]}")
        print(f"      STT: {row['transcript']}")
        print(f"      OUT: {row['corrected']}   {row['warnings'] or ''}")
        if not row["new_exact"]:
            print(f"      EXP: {row['expected']}")
    return rows


def check_typed(server: Server, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    print("\n=== Kalimat yang sama lewat jalur teks ===", flush=True)
    typed = []
    for row in rows:
        text = row["spoken"]
        text = text if text.endswith((".", "?", "!")) else text.rstrip(".") + "."
        correction = server.correct(text).json()["correction"]
        typed.append(
            {
                "text": text,
                "corrected": correction["corrected"],
                "changed": correction["changed"],
                "warnings": correction["warnings"],
                "expected": row["expected"],
                "exact": normalize(correction["corrected"]) == normalize(row["expected"]),
            }
        )
    for item in typed:
        if not item["changed"]:
            print("   tidak dikoreksi:", item["text"], item["warnings"], flush=True)
    return typed


def transcode(
    source: Path, container: str, codec: str, rate: int, options: dict[str, str] | None
) -> bytes:
    buffer = io.BytesIO()
    with (
        av.open(str(source)) as inp,
        av.open(buffer, "w", format=container, options=options or {}) as out,
    ):
        stream = out.add_stream(codec, rate=rate)
        for frame in inp.decode(audio=0):
            frame.pts = None
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode(None):
            out.mux(packet)
    return buffer.getvalue()


def check_formats(server: Server, voice: Path) -> list[dict[str, Any]]:
    print("\n=== Format audio (seperti yang dikirim browser/HP) ===", flush=True)
    rows = []
    for label, container, codec, rate, mime, name, options in FORMATS:
        try:
            data = transcode(voice, container, codec, rate, options)
        except av.FFmpegError as exc:  # the encoder is not in this PyAV build
            print(f"SKIP  {label}: encoder tidak tersedia ({exc})", flush=True)
            continue
        response = server.analyze(data, name, mime)
        body = response.json()
        text = body.get("transcript", {}).get("text", "")
        fixed = (body.get("correction") or {}).get("corrected")
        server.check(
            f"{label}, {len(data) // 1024} KB",
            response.status_code == 200 and "sing" in text.lower(),
            f"HTTP {response.status_code}: {text!r} -> {fixed!r}",
        )
        rows.append(
            {
                "format": label,
                "status": response.status_code,
                "transcript": text,
                "corrected": fixed,
            }
        )
    return rows


def check_endpoints(server: Server, fp_dir: Path, voice: Path) -> None:
    print("\n=== Endpoint ===", flush=True)
    client, check = server.client, server.check
    response = client.post(
        "/v1/audio/transcribe",
        files={"file": ("c.wav", voice.read_bytes(), "audio/wav")},
        data={"consent": "true"},
        headers={"X-Forwarded-For": "10.20.0.3"},
    )
    text = response.json().get("transcript", {}).get("text", "")
    check(
        "transcribe",
        response.status_code == 200 and "sing" in text.lower(),
        f"{response.status_code} {text!r}",
    )

    visitor = {"X-Forwarded-For": "10.20.0.4"}
    response = client.post(
        "/v1/jobs/analyze",
        files={"file": ("c.mp3", (fp_dir / "they is here today.mp3").read_bytes(), "audio/mpeg")},
        data={"consent": "true"},
        headers=visitor,
    )
    check(
        "job: diterima 202 + Location",
        response.status_code == 202 and "location" in response.headers,
        response.status_code,
    )
    url = response.headers.get("location", "/v1/jobs/none")
    state, body = "", {}
    for _ in range(120):
        body = client.get(url, headers=visitor).json()
        state = body.get("status", "")
        if state in ("succeeded", "failed"):
            break
        time.sleep(0.25)
    correction = (body.get("result") or {}).get("correction") or {}
    check("job: hasil lengkap", correction.get("corrected") == "They are here today.", state)
    description = ((correction.get("edits") or [{}])[0].get("description") or "").lower()
    check("job: penjelasan Indonesia", "kata kerja" in description, description)
    response = client.delete(url, headers=visitor)
    check("job: bisa dihapus", response.status_code in (200, 204), response.status_code)
    check("job: setelah dihapus 404", client.get(url, headers=visitor).status_code == 404)

    check("healthz", client.get("/healthz").status_code == 200)
    check("readyz", client.get("/readyz").status_code == 200)
    metrics = client.get("/metrics").text
    check("metrics tersedia", "speakfix_" in metrics)
    errors = [
        line
        for line in metrics.splitlines()
        if line.startswith("speakfix_inference_errors_total") and not line.endswith(" 0.0")
    ]
    check("metrics: tidak ada error inferensi", not errors, errors[:3])
    check("docs", client.get("/docs").status_code == 200)
    check("openapi", client.get("/openapi.json").status_code == 200)
    page = client.get("/")
    check(
        "UI + header keamanan",
        page.status_code == 200 and "content-security-policy" in page.headers,
    )
    check("static app.js", client.get("/static/app.js").status_code == 200)


def wav_bytes(seconds: float, tone: bool = False) -> bytes:
    rate = 16000
    t = np.arange(int(rate * seconds)) / rate
    pcm = (
        (np.sin(2 * np.pi * 220 * t) * 8000).astype(np.int16)
        if tone
        else np.zeros(t.size, np.int16)
    )
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(pcm.tobytes())
    return buffer.getvalue()


def check_errors(server: Server, voice: Path) -> None:
    print("\n=== Jalur error ===", flush=True)
    client, check = server.client, server.check

    def status(response: httpx.Response) -> str:
        return f"{response.status_code} {error_code(response)}"

    response = client.post(
        "/v1/audio/analyze",
        files={"file": ("c.wav", voice.read_bytes(), "audio/wav")},
        headers={"X-Forwarded-For": "10.20.0.5"},
    )
    check(
        "tanpa consent -> 403",
        response.status_code == 403 and error_code(response) == "consent_required",
        status(response),
    )
    response = server.analyze(b"ini bukan audio sama sekali" * 20, "x.wav", "audio/wav")
    check("bukan audio -> 415", response.status_code == 415, status(response))
    response = server.analyze(b"ID3" + b"\x00" * 3000, "x.mp3", "audio/mpeg")
    check("mp3 rusak -> 4xx", 400 <= response.status_code < 500, status(response))
    response = server.analyze(b"", "x.wav", "audio/wav")
    check("file kosong -> 4xx", 400 <= response.status_code < 500, status(response))
    response = server.analyze(wav_bytes(0.1), "x.wav", "audio/wav")
    check("terlalu pendek -> 422", response.status_code == 422, status(response))
    response = server.analyze(wav_bytes(70), "x.wav", "audio/wav")
    check("terlalu panjang -> 422", response.status_code == 422, status(response))
    response = server.analyze(wav_bytes(59.4), "x.wav", "audio/wav")
    check("batas rekam UI 59 s + sedikit lebih -> diterima", response.status_code == 200)
    response = server.analyze(wav_bytes(60.2), "x.wav", "audio/wav")
    check(
        "sedikit di atas 60 s -> 422 (alasan UI berhenti di 59 s)",
        response.status_code == 422,
        status(response),
    )
    response = server.analyze(wav_bytes(2), "x.wav", "audio/wav")
    body = response.json()
    check(
        "hening -> 200 + catatan",
        response.status_code == 200 and not body["transcript"]["text"],
        body.get("warnings"),
    )
    response = server.analyze(wav_bytes(2, tone=True), "x.wav", "audio/wav")
    check("nada tanpa ucapan -> 200", response.status_code == 200, response.json().get("warnings"))
    response = server.analyze(b"RIFF" + b"\0" * (11 * 1024 * 1024), "x.wav", "audio/wav")
    check("terlalu besar -> 413", response.status_code == 413, status(response))
    response = server.correct("")
    check("teks kosong -> 422", response.status_code == 422, status(response))
    response = server.correct("word " * 500)
    check("teks terlalu panjang -> 422", response.status_code == 422, status(response))
    check("pesan error tidak menggemakan input", "word word" not in response.text)
    response = client.post(
        "/v1/text/correct",
        content=b"{bukan json",
        headers={"Content-Type": "application/json", "X-Forwarded-For": "10.20.0.2"},
    )
    check("JSON rusak -> 422", response.status_code == 422, status(response))
    check("endpoint tidak ada -> 404", client.get("/v1/tidak-ada").status_code == 404)


def check_unusual_inputs(server: Server) -> list[dict[str, Any]]:
    print("\n=== Input tidak biasa ===", flush=True)
    rows = []
    for text in UNUSUAL_INPUTS:
        response = server.correct(text)
        if response.status_code != 200:
            print(f"  {response.status_code} {error_code(response)}  {text!r}")
            rows.append(
                {"text": text, "status": response.status_code, "code": error_code(response)}
            )
            continue
        correction = response.json()["correction"]
        print(f"  {text!r}\n     -> {correction['corrected']!r}  {correction['warnings'] or ''}")
        rows.append(
            {
                "text": text,
                "status": 200,
                "corrected": correction["corrected"],
                "warnings": correction["warnings"],
            }
        )
    server.check("input tidak biasa tidak ada yang 5xx", all(row["status"] < 500 for row in rows))
    correction = server.correct("I am a student").json()["correction"]
    server.check(
        "kalimat benar tanpa titik: tanpa catatan ragu",
        not correction["warnings"] and correction["corrected"] == "I am a student.",
        (correction["corrected"], correction["warnings"]),
    )
    return rows


def check_rate_limit(server: Server) -> None:
    print("\n=== Rate limit per pengunjung ===", flush=True)
    sentence, visitor = "She go to school.", "10.30.0.1"
    start = time.time()
    first_429 = None
    for i in range(1, 400):
        if server.correct(sentence, ip=visitor).status_code == 429:
            first_429 = i
            break
    took = time.time() - start
    # Capacity 60, plus a refill of 1 per second while the loop runs.
    server.check(
        "pengunjung A dibatasi (kapasitas 60 + 1/detik)",
        first_429 is not None and abs(first_429 - (60 + took)) <= 12,
        f"429 pertama di request ke-{first_429}, setelah {took:.0f} detik",
    )
    response = server.correct(sentence, ip=visitor)
    server.check(
        "429 menyertakan Retry-After",
        response.status_code == 429 and "retry-after" in response.headers,
        response.headers.get("retry-after"),
    )
    server.check(
        "pengunjung B tidak ikut terblokir",
        server.correct(sentence, ip="10.30.0.2").status_code == 200,
    )
    # Empty the bucket again first: visitor B's request took long enough to refill a token.
    while server.correct(sentence, ip=visitor).status_code != 429:
        pass
    response = server.correct(sentence, ip=f"1.1.1.1, {visitor}")
    server.check(
        "XFF palsu di depan tidak lolos", response.status_code == 429, response.status_code
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("url", help="e.g. http://127.0.0.1:8000")
    parser.add_argument("--out", type=Path, default=Path("reports/check_api.json"))
    parser.add_argument("--fp-dir", type=Path, default=Path("../../FP"), help="the team's mp3s")
    parser.add_argument("--voice", type=Path, default=Path("../recorded.wav"), help="real voice")
    parser.add_argument("--skip-rate-limit", action="store_true")
    args = parser.parse_args()

    server = Server(args.url)
    info = server.client.get("/v1/info").json()
    report: dict[str, Any] = {
        "base": args.url,
        "info": {"corrector": info["corrector"], "stt": info["stt"], "ui": info["ui"]},
    }
    print("server:", report["info"], flush=True)

    rows = check_recordings(server, args.fp_dir, args.voice)
    team = rows[: len(TEAM)]
    report["recordings"] = rows
    report["team_summary"] = {
        "exact_new": sum(r["new_exact"] for r in team),
        "exact_old": sum(r["old_exact"] for r in team),
        "n": len(team),
        "changed_new": sum(bool(r["changed"]) for r in team),
    }
    print("ringkasan 10 kalimat dokumen tim:", report["team_summary"], flush=True)
    exact, changed = sum(r["new_exact"] for r in rows), sum(bool(r["changed"]) for r in rows)
    print(f"semua rekaman: sama persis {exact} dari {len(rows)} | dikoreksi {changed}", flush=True)

    report["typed"] = check_typed(server, rows)
    exact = sum(t["exact"] for t in report["typed"])
    changed = sum(bool(t["changed"]) for t in report["typed"])
    print(f"diketik: sama persis {exact} dari {len(rows)} | dikoreksi {changed}", flush=True)

    report["formats"] = check_formats(server, args.voice)
    check_endpoints(server, args.fp_dir, args.voice)
    check_errors(server, args.voice)
    report["odd_inputs"] = check_unusual_inputs(server)
    if not args.skip_rate_limit:
        check_rate_limit(server)

    report["failed_checks"] = server.failed
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    times = [r["total_ms"] for r in rows]
    print(f"\nlatensi rekaman: median {statistics.median(times):.0f} ms, maks {max(times):.0f} ms")
    print(f"HASIL: {len(server.failed)} pengecekan gagal", server.failed)
    sys.exit(1 if server.failed else 0)


if __name__ == "__main__":
    main()
