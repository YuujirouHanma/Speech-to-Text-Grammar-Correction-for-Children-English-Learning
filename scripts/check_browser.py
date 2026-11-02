"""End-to-end check of the Speech Grammar page in a real browser, through Playwright.

Types, uploads and records (with a fake microphone) the way a visitor does, and checks what the
page shows, including its error messages, phone layout and dark mode. Exits with status 1 if a
check failed.

    pip install playwright
    playwright install firefox webkit      # Edge and Chrome use the installed browser
    python -m speech_grammar                     # in another terminal
    python scripts/check_browser.py http://127.0.0.1:8000 --engine msedge

WebKit has no fake microphone, so its recording checks are skipped.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from playwright.sync_api import Browser, BrowserContext, Page, sync_playwright

SNAPSHOT_JS = """() => {
  const q = (id) => document.getElementById(id);
  const vis = (el) => !!el && !el.hidden && el.getClientRects().length > 0;
  return {
    result: vis(q('result')),
    original: q('original').textContent,
    marks: [...q('original').querySelectorAll('mark')].map((m) => m.textContent),
    carets: q('original').querySelectorAll('.missing').length,
    labelVisible: vis(q('correctedLabel')),
    correctedVisible: vis(q('corrected')),
    corrected: q('corrected').textContent,
    edits: [...q('edits').querySelectorAll('li')].map((li) => li.textContent),
    praise: vis(q('praise')),
    warnings: q('warnings').textContent,
    timing: q('timing').textContent,
    status: q('status').textContent,
    statusError: q('status').className === 'error',
  };
}"""
WAIT_DONE = (
    "document.getElementById('timing').textContent !== '' || "
    "document.getElementById('status').className === 'error'"
)
RECORDING = "document.getElementById('recordBtn').textContent === 'Berhenti'"
STOPPED = "document.getElementById('recordBtn').textContent === 'Mulai rekam'"
FINISHED = (
    f"{STOPPED} && "
    "!document.getElementById('status').textContent.startsWith('Mendengarkan') && "
    "!document.getElementById('status').textContent.startsWith('Merekam')"
)

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: object = "") -> None:
    results.append((name, bool(ok), str(detail)))
    suffix = f"  -> {detail}" if detail != "" else ""
    print(f"{'PASS' if ok else 'FAIL'}  {name}{suffix}", flush=True)


def wait_until(page: Page, expression: str, timeout: float = 60.0) -> None:
    """Poll from Python: page-side waiting needs eval, which the page's CSP (rightly) blocks."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if page.evaluate(expression):
            return
        page.wait_for_timeout(100)
    raise TimeoutError(expression)


def reset(page: Page) -> None:
    page.evaluate(
        "() => { const q = (i) => document.getElementById(i); q('timing').textContent='';"
        " q('status').textContent=''; q('status').className=''; }"
    )


def check_text(page: Page, text: str) -> dict[str, Any]:
    page.fill("#textInput", text)
    reset(page)
    page.click("#checkBtn")
    wait_until(page, WAIT_DONE)
    return page.evaluate(SNAPSHOT_JS)


def upload(page: Page, path: Path) -> dict[str, Any]:
    reset(page)
    page.set_input_files("#fileInput", str(path))
    wait_until(page, WAIT_DONE)
    return page.evaluate(SNAPSHOT_JS)


def launch(playwright: Any, engine: str, voice: Path) -> Browser:
    if engine == "webkit":
        return playwright.webkit.launch(headless=True)
    if engine == "firefox":
        return playwright.firefox.launch(
            headless=True,
            firefox_user_prefs={
                "media.navigator.streams.fake": True,
                "media.navigator.permission.disabled": True,
            },
        )
    fake_microphone = voice.resolve().as_posix() + "%noloop"
    return playwright.chromium.launch(
        channel=engine,
        headless=True,
        args=[
            "--use-fake-ui-for-media-stream",
            "--use-fake-device-for-media-stream",
            f"--use-file-for-fake-audio-capture={fake_microphone}",
        ],
    )


def check_page_load(page: Page, url: str) -> dict[str, Any]:
    page.goto(url + "/", wait_until="networkidle")
    check("halaman terbuka, judul Indonesia", "Latihan bicara" in page.title(), page.title())
    check("lang=id", page.evaluate("document.documentElement.lang") == "id")
    timer = page.inner_text("#timer")
    check("timer memakai batas rekam 59 detik", timer == "0:00 / 0:59", timer)
    check("kotak API key tersembunyi (server tanpa key)", not page.is_visible("#apiKeyBox"))
    info = page.evaluate("fetch('/v1/info').then((r) => r.json())")
    demo = info["ui"]["demo_mode"]
    print("server:", info["corrector"], "|", info["stt"], "| ui:", info["ui"], flush=True)
    check("banner demo sesuai server", page.is_visible("#demoBanner") == demo, f"demo_mode={demo}")
    consent = page.inner_text("#consentText")
    if demo:
        check("teks persetujuan versi demo", "18 tahun" in consent, consent[:60])
    else:
        check("teks persetujuan versi orang tua", "orang tua" in consent, consent[:60])
    return info


def check_typed_text(page: Page, shots: Path, engine: str) -> None:
    s = check_text(page, "My mother cook rice every morning.")
    check(
        "teks salah: saran tampil",
        s["correctedVisible"] and "cooks" in s["corrected"],
        s["corrected"],
    )
    check("teks salah: label tampil, pujian tidak", s["labelVisible"] and not s["praise"])
    check("teks salah: kata salah disorot", s["marks"] == ["cook"], s["marks"])
    check(
        "teks salah: penjelasan Indonesia",
        any("Kata kerja" in e and "cook" in e for e in s["edits"]),
        s["edits"],
    )
    check("teks salah: waktu proses tampil", s["timing"].startswith("Diproses dalam"), s["timing"])
    page.screenshot(path=str(shots / f"{engine}_desktop_koreksi.png"), full_page=True)

    s = check_text(page, "She goes to school every day.")
    check("teks benar: pujian tampil", s["praise"], s["warnings"] or "-")
    check(
        "teks benar: saran & label disembunyikan",
        not s["correctedVisible"] and not s["labelVisible"],
    )
    check("teks benar: daftar edit kosong", s["edits"] == [], s["edits"])

    s = check_text(page, "She goes to school every day")
    check(
        "teks tanpa titik: saran titik, tanpa catatan ragu",
        s["corrected"].endswith("day.") and not s["warnings"],
        s["warnings"] or s["corrected"],
    )

    s = check_text(page, "I am agree with your opinion.")
    print("   ragu-ragu ->", json.dumps(s, ensure_ascii=False)[:300], flush=True)
    check("saran ditahan: TIDAK ada pujian", not s["praise"])
    check("saran ditahan: catatan tampil", "belum cukup yakin" in s["warnings"], s["warnings"])
    check("saran ditahan: saran disembunyikan", not s["correctedVisible"])
    page.screenshot(path=str(shots / f"{engine}_desktop_ragu.png"), full_page=True)

    s = check_text(page, "He don't like apples. All of my friends are here.")
    check("multi-kalimat: dua kalimat diproses", "doesn't" in s["corrected"], s["corrected"])

    s = check_text(page, "He likes <b>apples</b> and <script>alert(1)</script> cats")
    check(
        "input HTML tidak dieksekusi",
        page.evaluate("document.querySelectorAll('#original b, #original script').length") == 0,
        s["original"][:70],
    )

    before = page.evaluate("document.getElementById('original').textContent")
    page.fill("#textInput", "   ")
    reset(page)
    page.click("#checkBtn")
    page.wait_for_timeout(400)
    check("teks kosong: tidak mengirim apa pun", page.evaluate(SNAPSHOT_JS)["original"] == before)


def check_consent_and_upload(page: Page, mp3: Path, shots: Path) -> None:
    reset(page)
    page.uncheck("#consent")
    page.click("#recordBtn")
    page.wait_for_timeout(200)
    s = page.evaluate(SNAPSHOT_JS)
    check(
        "rekam tanpa persetujuan: ditolak",
        s["statusError"] and "persetujuan" in s["status"],
        s["status"],
    )
    s = upload(page, mp3)
    check(
        "unggah tanpa persetujuan: ditolak",
        s["statusError"] and "persetujuan" in s["status"],
        s["status"],
    )
    page.check("#consent")

    s = upload(page, mp3)
    check(
        "unggah mp3: berhasil",
        not s["statusError"] and "are" in s["corrected"],
        f"{s['original']!r} -> {s['corrected']!r}",
    )
    check(
        "unggah mp3: penjelasan Indonesia", any("Kata kerja" in e for e in s["edits"]), s["edits"]
    )

    junk = shots / "bukan_audio.txt"
    junk.write_text("ini bukan audio " * 50, encoding="utf-8")
    s = upload(page, junk)
    check(
        "unggah bukan audio: pesan Indonesia",
        s["statusError"] and "Format file" in s["status"],
        s["status"],
    )
    big = shots / "besar.wav"
    big.write_bytes(b"RIFF" + b"\0" * (11 * 1024 * 1024))
    s = upload(page, big)
    check(
        "unggah file besar: pesan Indonesia",
        s["statusError"] and "maksimal 10 MB" in s["status"],
        s["status"],
    )
    big.unlink()


def start_recording(page: Page) -> None:
    reset(page)
    page.click("#recordBtn")
    wait_until(page, RECORDING, 15.0)
    time.sleep(1.5)


def check_recording(page: Page, shots: Path, engine: str) -> None:
    reset(page)
    page.click("#recordBtn")
    wait_until(
        page, f"{RECORDING} || document.getElementById('status').className === 'error'", 15.0
    )
    s = page.evaluate(SNAPSHOT_JS)
    check("rekam: mulai merekam", page.inner_text("#recordBtn") == "Berhenti", s["status"])
    time.sleep(6.0)
    timer = page.inner_text("#timer")
    check("rekam: timer berjalan", not timer.startswith("0:00"), timer)
    mime = page.evaluate("recorder ? recorder.mimeType : ''")
    page.click("#recordBtn")
    wait_until(page, FINISHED)
    s = page.evaluate(SNAPSHOT_JS)
    print(f"   rekaman ({mime}) ->", json.dumps(s, ensure_ascii=False)[:420], flush=True)
    check("rekam: tombol kembali", page.inner_text("#recordBtn") == "Mulai rekam")
    check("rekam: pemutar audio tampil", page.is_visible("#playback"))
    check("rekam: server menerima format browser", not s["statusError"], f"{mime} | {s['status']}")
    if engine == "firefox":  # Firefox's fake microphone is a tone, not speech
        check(
            "rekam: nada tanpa suara ditangani", "tidak ada suara" in s["original"], s["warnings"]
        )
        check("rekam: tanpa pujian saat tidak ada suara", not s["praise"])
    else:
        check(
            "rekam: suara terdengar benar",
            "sing" in s["original"].lower(),
            f"{s['original']!r} -> {s['corrected']!r}",
        )
    page.screenshot(path=str(shots / f"{engine}_desktop_rekam.png"), full_page=True)

    start_recording(page)
    page.evaluate("startedAt = Date.now() - 58500")  # as if 58.5 s had passed
    wait_until(page, STOPPED, 5.0)
    check("rekam: berhenti sendiri di batas 59 detik", True, page.inner_text("#timer"))
    wait_until(page, FINISHED)
    s = page.evaluate(SNAPSHOT_JS)
    check("rekam: rekaman batas waktu diterima server", not s["statusError"], s["status"])

    start_recording(page)
    page.evaluate("recorder.stop()")  # a phone locking its screen ends the recording like this
    wait_until(page, STOPPED, 5.0)
    wait_until(page, FINISHED)
    before_tick = page.inner_text("#timer")
    time.sleep(1.3)
    check(
        "rekam diakhiri browser: tombol kembali, timer berhenti",
        page.inner_text("#timer") == before_tick,
        before_tick,
    )


def check_failures(context: BrowserContext, url: str) -> None:
    for error_name, expected in (
        ("NotAllowedError", "Izin mikrofon ditolak"),
        ("NotFoundError", "Mikrofon tidak ditemukan"),
    ):
        page = context.new_page()
        page.add_init_script(
            "Object.defineProperty(navigator, 'mediaDevices', {value: {getUserMedia: () => "
            f"Promise.reject(new DOMException('x', '{error_name}'))}}}});"
            "window.MediaRecorder = window.MediaRecorder || function () {};"
        )
        page.goto(url + "/", wait_until="networkidle")
        page.check("#consent")
        page.click("#recordBtn")
        page.wait_for_timeout(300)
        message = page.inner_text("#status")
        check(f"mikrofon {error_name}: pesan Indonesia", expected in message, message[:70])
        check(
            f"mikrofon {error_name}: tombol tetap siap",
            page.inner_text("#recordBtn") == "Mulai rekam",
        )
        page.close()

    page = context.new_page()
    full = {"error": {"code": "resource_exhausted", "message": "Out of memory", "request_id": "x"}}
    page.route("**/v1/text/correct", lambda r: r.fulfill(status=503, json=full))
    page.goto(url + "/", wait_until="networkidle")
    s = check_text(page, "She go to school.")
    check("server penuh: pesan Indonesia", s["statusError"] and "penuh" in s["status"], s["status"])
    page.close()

    page = context.new_page()
    page.route("**/v1/text/correct", lambda r: r.abort())
    page.goto(url + "/", wait_until="networkidle")
    s = check_text(page, "She go to school.")
    check(
        "koneksi putus: pesan Indonesia",
        s["statusError"] and "terputus" in s["status"],
        s["status"],
    )
    page.close()

    page = context.new_page()
    page.add_init_script("Object.defineProperty(window, 'isSecureContext', {value: false});")
    page.goto(url + "/", wait_until="networkidle")
    page.check("#consent")
    page.click("#recordBtn")
    page.wait_for_timeout(200)
    message = page.inner_text("#status")
    check("tanpa HTTPS: pesan mikrofon jelas", "https://" in message, message[:70])
    page.close()


def check_phone_layout(browser: Browser, url: str, shots: Path, engine: str) -> None:
    for scheme in ("light", "dark"):
        context = browser.new_context(viewport={"width": 375, "height": 800}, color_scheme=scheme)
        page = context.new_page()
        page.goto(url + "/", wait_until="networkidle")
        check_text(page, "My mother cook rice every morning.")
        page.wait_for_timeout(900)  # smooth scrolling
        top = page.evaluate("document.getElementById('corrected').getBoundingClientRect().top")
        check(f"HP 375px ({scheme}): hasil tergulir ke layar", 0 <= top <= 800, f"top={top:.0f}px")
        overflow = page.evaluate("document.documentElement.scrollWidth - window.innerWidth")
        check(
            f"HP 375px ({scheme}): tidak ada scroll horizontal",
            overflow <= 0,
            f"kelebihan {overflow}px",
        )
        page.screenshot(path=str(shots / f"{engine}_hp_{scheme}.png"), full_page=True)
        context.close()


def check_other_server_settings(context: BrowserContext, url: str, info: dict[str, Any]) -> None:
    """Wording without demo mode, by faking only the server's answer about its settings."""
    page = context.new_page()
    ui = {"demo_mode": False, "feedback_language": "id", "api_key_required": True}
    page.route("**/v1/info", lambda r: r.fulfill(json={**info, "ui": ui}))
    page.goto(url + "/", wait_until="networkidle")
    check("non-demo: banner tersembunyi", not page.is_visible("#demoBanner"))
    check("non-demo: teks persetujuan orang tua", "orang tua" in page.inner_text("#consentText"))
    check("server dengan key: kotak API key tampil", page.is_visible("#apiKeyBox"))
    page.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("url", help="e.g. http://127.0.0.1:8000")
    parser.add_argument(
        "--engine", choices=["msedge", "chrome", "firefox", "webkit"], default="msedge"
    )
    parser.add_argument(
        "--voice", type=Path, default=Path("../recorded.wav"), help="fake microphone"
    )
    parser.add_argument("--mp3", type=Path, default=Path("../../FP/they is here today.mp3"))
    parser.add_argument(
        "--shots", type=Path, default=Path(tempfile.gettempdir()) / "speech-grammar-browser-check"
    )
    args = parser.parse_args()
    url, engine, shots = args.url.rstrip("/"), args.engine, args.shots
    shots.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as playwright:
        browser = launch(playwright, engine, args.voice)
        print("browser:", engine, browser.version, "| url:", url, flush=True)
        context = browser.new_context(viewport={"width": 1100, "height": 900})
        page = context.new_page()
        console_errors: list[str] = []
        page.on(
            "console",
            lambda m: console_errors.append(f"{m.type}: {m.text}") if m.type == "error" else None,
        )
        page.on("pageerror", lambda e: console_errors.append(f"pageerror: {e}"))

        info = check_page_load(page, url)
        check_typed_text(page, shots, engine)
        check_consent_and_upload(page, args.mp3, shots)
        secure = page.evaluate("window.isSecureContext")
        check("secure context (mikrofon diizinkan)", secure, url)
        if secure and engine == "webkit":
            print("   webkit: tidak ada mikrofon tiruan, alur rekam dilewati", flush=True)
        elif secure:
            check_recording(page, shots, engine)
        check_failures(context, url)
        check_phone_layout(browser, url, shots, engine)
        check_other_server_settings(context, url, info)

        # Expected noise only: 4xx answers of the deliberate error tests above, and on WebKit,
        # Playwright's own screenshot style injection being refused by the CSP.
        unexpected = [
            e
            for e in console_errors
            if not ("415" in e or "422" in e)
            and not (
                engine == "webkit"
                and "style" in e.lower()
                and "content security policy" in e.lower()
            )
        ]
        check("tidak ada error konsol / pelanggaran CSP", not unexpected, unexpected[:6])
        browser.close()

    failed = [r for r in results if not r[1]]
    print(f"\n{engine}: {len(results) - len(failed)}/{len(results)} lulus")
    for name, _, detail in failed:
        print("  GAGAL:", name, "->", detail)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
