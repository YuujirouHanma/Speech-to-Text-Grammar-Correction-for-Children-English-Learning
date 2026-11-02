"""End-to-end check of the Speech Grammar page in a real browser, through Playwright.

Types, uploads and records (with a fake microphone) the way a visitor does, and checks what the
page shows, including its error messages, phone layout and dark mode. Exits with status 1 if a
check failed.

    pip install playwright
    playwright install firefox webkit      # Edge and Chrome use the installed browser
    python -m speech_grammar                     # in another terminal, after building frontend/
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
from urllib.parse import urlsplit

from playwright.sync_api import Browser, BrowserContext, Page, Route, sync_playwright

SNAPSHOT_JS = """() => {
  const q = (id) => document.getElementById(id);
  const vis = (el) => !!el && !el.hidden && el.getClientRects().length > 0;
  const text = (id) => (q(id) ? q(id).textContent : '');
  const view = q('diffView');
  const without = (tag) => {
    if (!view) return '';
    const copy = view.cloneNode(true);
    copy.querySelectorAll(tag).forEach((node) => node.remove());
    return copy.textContent;
  };
  const items = (tag) => (view ? [...view.querySelectorAll(tag)].map((n) => n.textContent) : []);
  return {
    result: vis(q('result')),
    original: without('ins'),
    corrected: without('del'),
    marks: items('del'),
    fixes: items('ins'),
    edits: q('edits') ? [...q('edits').querySelectorAll('li')].map((li) => li.textContent) : [],
    praise: vis(q('praise')),
    warnings: text('warnings'),
    timing: text('timing').trim(),
    status: text('status'),
    error: text('apiError') + text('micError'),
  };
}"""
NOT_BUSY = "document.getElementById('resultColumn').getAttribute('aria-busy') !== 'true'"
RECORDING = "document.getElementById('recordBtn').getAttribute('aria-label') === 'Berhenti'"
STOPPED = "document.getElementById('recordBtn').getAttribute('aria-label') === 'Mulai rekam'"
MIC_ERROR = "document.getElementById('micError') !== null"
# Lets the checks end a recording the way the browser does, e.g. when a phone locks its screen.
KEEP_RECORDER = """
if (window.MediaRecorder) {
  const Real = window.MediaRecorder;
  window.MediaRecorder = class extends Real {
    constructor(...args) { super(...args); window.__recorder = this; }
  };
}
"""

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: object = "") -> None:
    results.append((name, bool(ok), str(detail)))
    suffix = f"  -> {detail}" if detail != "" else ""
    print(f"{'PASS' if ok else 'FAIL'}  {name}{suffix}", flush=True)


def wait_until(page: Page, expression: str, timeout: float = 90.0) -> None:
    """Poll from Python: page-side waiting needs eval, which the page's CSP (rightly) blocks."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if page.evaluate(expression):
            return
        page.wait_for_timeout(100)
    raise TimeoutError(expression)


def snapshot(page: Page) -> dict[str, Any]:
    return page.evaluate(SNAPSHOT_JS)


def check_text(page: Page, text: str) -> dict[str, Any]:
    page.fill("#textInput", text)
    page.click("#checkBtn")  # the page is busy as soon as this returns
    wait_until(page, NOT_BUSY)
    return snapshot(page)


def upload(page: Page, path: Path) -> dict[str, Any]:
    page.click("#uploadBtn")
    page.set_input_files("#fileInput", str(path))
    page.click("#uploadSubmit")
    wait_until(page, NOT_BUSY)
    return snapshot(page)


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


def clock(seconds: int) -> str:
    return f"{seconds // 60}:{seconds % 60:02d}"


def check_page_load(page: Page, url: str) -> dict[str, Any]:
    page.goto(url + "/", wait_until="networkidle")
    check("halaman terbuka, judul proyek", "Speech to Text" in page.title(), page.title())
    check("lang=id", page.evaluate("document.documentElement.lang") == "id")
    info = page.evaluate("fetch('/v1/info').then((r) => r.json())")
    limit = int(info["limits"]["max_audio_seconds"]) - 1
    timer = page.inner_text("#timer")
    check("timer memakai batas rekam server", timer == f"0:00 / {clock(limit)}", timer)
    check("kotak API key tersembunyi (server tanpa key)", not page.is_visible("#apiKeyBox"))
    demo = info["ui"]["demo_mode"]
    print("server:", info["corrector"], "|", info["stt"], "| ui:", info["ui"], flush=True)
    check("banner demo sesuai server", page.is_visible("#demoBanner") == demo, f"demo_mode={demo}")
    consent = page.inner_text("#consentText")
    if demo:
        check("teks persetujuan versi demo", "18 tahun" in consent, consent[:60])
    else:
        check("teks persetujuan versi orang tua", "orang tua" in consent, consent[:60])
    check("fon proyek termuat", page.evaluate("document.fonts.check('16px \"DM Sans\"')"))
    return info


def check_typed_text(page: Page, shots: Path, engine: str) -> None:
    s = check_text(page, "My mother cook rice every morning.")
    check("teks salah: saran tampil", "cooks" in s["fixes"], s["corrected"])
    check("teks salah: pujian tidak tampil", not s["praise"])
    check("teks salah: kata salah dicoret", s["marks"] == ["cook"], s["marks"])
    check(
        "teks salah: penjelasan Indonesia",
        any("Kata kerja" in e and "cook" in e for e in s["edits"]),
        s["edits"],
    )
    check("teks salah: waktu proses tampil", s["timing"].startswith("Diproses dalam"), s["timing"])
    page.screenshot(path=str(shots / f"{engine}_desktop_koreksi.png"), full_page=True)

    page.click("#viewClean")
    clean = page.inner_text("#cleanView")
    check("tampilan bersih: kalimat hasil", clean.endswith("cooks rice every morning."), clean)
    page.click("#viewSplit")
    pair = page.inner_text("#original"), page.inner_text("#corrected")
    check("tampilan berdampingan: asli dan saran", "cook " in pair[0] and "cooks" in pair[1], pair)
    page.click("#viewDiff")

    s = check_text(page, "She goes to school every day.")
    check("teks benar: pujian tampil", s["praise"], s["warnings"] or "-")
    check("teks benar: tidak ada coretan", not s["marks"] and not s["fixes"])
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
    check("saran ditahan: saran disembunyikan", not s["fixes"])
    page.screenshot(path=str(shots / f"{engine}_desktop_ragu.png"), full_page=True)

    s = check_text(page, "He don't like apples. All of my friends are here.")
    check("multi-kalimat: dua kalimat diproses", "doesn't" in s["corrected"], s["corrected"])

    check_text(page, "He likes <b>apples</b> and <script>alert(1)</script> cats")
    check(
        "input HTML tidak dieksekusi",
        page.evaluate("document.querySelectorAll('#diffView b, #diffView script').length") == 0,
    )

    page.fill("#textInput", "   ")
    check("teks kosong: tombol periksa nonaktif", page.is_disabled("#checkBtn"))


def check_tools(page: Page) -> None:
    page.click(".preset >> nth=1")
    wait_until(page, NOT_BUSY)
    s = snapshot(page)
    check("contoh kalimat: langsung diperiksa", bool(s["fixes"]), s["corrected"])

    text = page.input_value("#textInput")
    page.click("#clearBtn")
    check("bersihkan: teks kosong", page.input_value("#textInput") == "")
    page.click("#undoBtn")
    check("bersihkan: bisa dibatalkan", page.input_value("#textInput") == text)

    page.fill("#textInput", "They is playing in the park.")
    page.press("#textInput", "Control+Enter")
    wait_until(page, NOT_BUSY)
    s = snapshot(page)
    check("Ctrl+Enter: memeriksa kalimat", "are" in s["fixes"], s["corrected"])

    page.click("nav >> text=Riwayat Koreksi")
    dialog = page.locator("[role=dialog]")
    check("riwayat: kalimat terakhir ada", "They is playing" in dialog.inner_text())
    check("riwayat: tersimpan di browser ini", "browser ini" in dialog.inner_text())
    page.keyboard.press("Escape")
    check("riwayat: Esc menutup", dialog.count() == 0)

    page.click("nav >> text=Panduan Grammar")
    page.click("text=Uji Kalimat Ini >> nth=0")
    wait_until(page, NOT_BUSY)
    s = snapshot(page)
    check("panduan grammar: contoh diuji", "works" in s["fixes"], s["corrected"])

    page.click("nav >> text=Pintasan Keyboard")
    check("pintasan keyboard: terbuka", "Alt + V" in page.locator("[role=dialog]").inner_text())
    page.keyboard.press("Escape")

    words = page.inner_text("body").lower()
    stale = [word for word in ("gemini", "heuristi", "nielsen") if word in words]
    check("halaman tidak menyebut Gemini atau klaim standar", not stale, stale)

    before = page.evaluate("document.documentElement.classList.contains('dark')")
    page.click("[aria-label='Toggle theme']")
    after = page.evaluate("document.documentElement.classList.contains('dark')")
    check("tema: tombol mengganti tema", before != after)
    page.click("[aria-label='Toggle theme']")


def check_consent_and_upload(page: Page, info: dict[str, Any], mp3: Path, shots: Path) -> None:
    page.uncheck("#consent")
    page.click("#recordBtn")
    wait_until(page, MIC_ERROR, 5.0)
    s = snapshot(page)
    check("rekam tanpa persetujuan: ditolak", "persetujuan" in s["error"], s["error"])
    check("rekam tanpa persetujuan: tidak merekam", page.evaluate(STOPPED))

    page.click("#uploadBtn")
    page.set_input_files("#fileInput", str(mp3))
    check("unggah tanpa persetujuan: tombol nonaktif", page.is_disabled("#uploadSubmit"))
    page.check("#consentUpload")
    check("unggah: centang di dialog berlaku", page.is_enabled("#uploadSubmit"))
    page.click("#uploadCancel")
    check("persetujuan sama dengan di halaman", page.is_checked("#consent"))

    s = upload(page, mp3)
    check(
        "unggah audio: berhasil",
        not s["error"] and "are" in s["corrected"],
        f"{s['original']!r} -> {s['corrected']!r}",
    )
    check(
        "unggah audio: penjelasan Indonesia", any("Kata kerja" in e for e in s["edits"]), s["edits"]
    )
    check("unggah audio: transkrip masuk ke editor", bool(page.input_value("#textInput")))

    junk = shots / "bukan_audio.txt"
    junk.write_text("ini bukan audio " * 50, encoding="utf-8")
    s = upload(page, junk)
    check("unggah bukan audio: pesan Indonesia", "Format file" in s["error"], s["error"])

    limit_mb = info["limits"]["max_upload_mb"]
    big = shots / "besar.wav"
    big.write_bytes(b"RIFF" + b"\0" * int((limit_mb + 1) * 1024 * 1024))
    page.click("#uploadBtn")
    page.set_input_files("#fileInput", str(big))
    message = page.inner_text("#uploadError")
    check("unggah file besar: pesan batas server", f"maksimal {limit_mb:g} MB" in message, message)
    check("unggah file besar: tidak terkirim", page.is_disabled("#uploadSubmit"))
    page.click("#uploadCancel")
    big.unlink()


def start_recording(page: Page, seconds: float = 1.5) -> None:
    page.click("#recordBtn")
    wait_until(page, RECORDING, 15.0)
    time.sleep(seconds)


def check_recording(page: Page, shots: Path, engine: str) -> None:
    page.click("#recordBtn")
    wait_until(page, f"{RECORDING} || {MIC_ERROR}", 15.0)
    s = snapshot(page)
    check(
        "rekam: mulai merekam, pesan lama hilang",
        page.evaluate(RECORDING) and not s["error"],
        s["error"],
    )
    time.sleep(6.0)
    timer = page.inner_text("#timer")
    check("rekam: timer berjalan", not timer.startswith("0:00"), timer)
    check("rekam: status terbaca", "Merekam" in snapshot(page)["status"])
    page.screenshot(path=str(shots / f"{engine}_desktop_sedang_rekam.png"), full_page=True)
    mime = page.evaluate("window.__recorder ? window.__recorder.mimeType : ''")
    page.click("#recordBtn")
    wait_until(page, f"{STOPPED} && {NOT_BUSY}")
    s = snapshot(page)
    print(f"   rekaman ({mime}) ->", json.dumps(s, ensure_ascii=False)[:420], flush=True)
    check("rekam: tombol kembali", page.evaluate(STOPPED))
    check("rekam: pemutar audio tampil", page.is_visible("#playback"))
    check("rekam: server menerima format browser", not s["error"], f"{mime} | {s['error']}")
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

    sent: list[str] = []
    page.on("request", lambda r: sent.append(r.url) if "/v1/audio/" in r.url else None)
    start_recording(page)
    page.keyboard.press("Escape")
    wait_until(page, STOPPED, 5.0)
    page.wait_for_timeout(500)
    check("rekam: Esc membatalkan tanpa mengirim", not sent, sent)
    check("rekam: Esc memberi tahu", "dibatalkan" in page.inner_text("#toast"))

    start_recording(page)
    page.evaluate("window.__recorder.stop()")  # a phone locking its screen ends it like this
    wait_until(page, f"{STOPPED} && {NOT_BUSY}")
    before_tick = page.inner_text("#timer")
    time.sleep(1.3)
    check(
        "rekam diakhiri browser: tombol kembali, timer berhenti",
        page.inner_text("#timer") == before_tick,
        before_tick,
    )


def check_recording_limit(context: BrowserContext, url: str, info: dict[str, Any]) -> None:
    page = context.new_page()
    short = {**info, "limits": {**info["limits"], "max_audio_seconds": 4}}
    page.route("**/v1/info", lambda r: r.fulfill(json=short))
    page.goto(url + "/", wait_until="networkidle")
    page.check("#consent")
    page.click("#recordBtn")
    wait_until(page, RECORDING, 15.0)
    wait_until(page, STOPPED, 8.0)
    check("rekam: berhenti sendiri di batas waktu", True, page.inner_text("#timer"))
    wait_until(page, NOT_BUSY)
    s = snapshot(page)
    check("rekam: rekaman batas waktu diterima server", not s["error"], s["error"])
    page.close()


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
        wait_until(page, MIC_ERROR, 5.0)
        message = page.inner_text("#micError")
        check(f"mikrofon {error_name}: pesan Indonesia", expected in message, message[:90])
        check(f"mikrofon {error_name}: tombol tetap siap", page.evaluate(STOPPED))
        page.close()

    page = context.new_page()
    full = {"error": {"code": "resource_exhausted", "message": "Out of memory", "request_id": "x"}}
    page.route("**/v1/text/correct", lambda r: r.fulfill(status=503, json=full))
    page.goto(url + "/", wait_until="networkidle")
    s = check_text(page, "She go to school.")
    check("server penuh: pesan Indonesia", "penuh" in s["error"], s["error"])
    page.unroute("**/v1/text/correct")
    check("gagal: teks tidak hilang", page.input_value("#textInput") == "She go to school.")
    page.click("#retryBtn")
    wait_until(page, NOT_BUSY)
    s = snapshot(page)
    check("Coba Lagi: mengirim ulang", not s["error"] and s["result"], s["corrected"])
    page.close()

    page = context.new_page()
    page.route("**/v1/text/correct", lambda r: r.abort())
    page.goto(url + "/", wait_until="networkidle")
    s = check_text(page, "She go to school.")
    check("koneksi putus: pesan Indonesia", "terputus" in s["error"], s["error"])
    page.close()

    page = context.new_page()
    held: list[Route] = []
    page.route("**/v1/text/correct", lambda route: held.append(route))
    page.goto(url + "/", wait_until="networkidle")
    page.fill("#textInput", "She go to school.")
    page.click("#checkBtn")
    page.wait_for_timeout(8500)
    status = snapshot(page)["status"]
    check("permintaan lama: halaman menjelaskan", "memuat model" in status, status)
    check("permintaan lama: penjelasan terlihat", "memuat model" in page.inner_text("#analyzing"))
    held[0].abort()
    page.close()

    page = context.new_page()
    page.add_init_script("Object.defineProperty(window, 'isSecureContext', {value: false});")
    page.goto(url + "/", wait_until="networkidle")
    page.check("#consent")
    page.click("#recordBtn")
    wait_until(page, MIC_ERROR, 5.0)
    message = page.inner_text("#micError")
    check("tanpa HTTPS: pesan mikrofon jelas", "https://" in message, message[:90])
    page.close()


def check_phone_layout(browser: Browser, url: str, shots: Path, engine: str) -> None:
    for scheme in ("light", "dark"):
        context = browser.new_context(viewport={"width": 360, "height": 800}, color_scheme=scheme)
        page = context.new_page()
        page.goto(url + "/", wait_until="networkidle")
        page.screenshot(path=str(shots / f"{engine}_hp_{scheme}_awal.png"), full_page=True)
        check_text(page, "My mother cook rice every morning.")
        page.wait_for_timeout(900)  # smooth scrolling
        top = page.evaluate("document.getElementById('result').getBoundingClientRect().top")
        check(f"HP 360px ({scheme}): hasil tergulir ke layar", 0 <= top <= 800, f"top={top:.0f}px")
        overflow = page.evaluate("document.documentElement.scrollWidth - window.innerWidth")
        check(
            f"HP 360px ({scheme}): tidak ada scroll horizontal",
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

    page = context.new_page()
    warmups: list[str] = []
    page.route(
        "**/v1/info",
        lambda r: r.fulfill(
            json={
                **info,
                "corrector": None,
                "stt": None,
                "ui": {**info["ui"], "load_models_on_demand": True},
            }
        ),
    )
    page.on("request", lambda r: warmups.append(r.method) if "/v1/warmup" in r.url else None)
    page.goto(url + "/", wait_until="networkidle")
    check("model belum dimuat: halaman memanggil warmup", warmups == ["POST"], warmups)
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
        context = browser.new_context(viewport={"width": 1366, "height": 900})
        context.add_init_script(KEEP_RECORDER)
        page = context.new_page()
        console_errors: list[str] = []
        outside: set[str] = set()
        page.on(
            "console",
            lambda m: console_errors.append(f"{m.type}: {m.text}") if m.type == "error" else None,
        )
        page.on("pageerror", lambda e: console_errors.append(f"pageerror: {e}"))
        own_host = urlsplit(url).netloc
        page.on(
            "request",
            lambda r: (
                outside.add(urlsplit(r.url).netloc)
                if r.url.startswith("http") and urlsplit(r.url).netloc != own_host
                else None
            ),
        )

        info = check_page_load(page, url)
        check_typed_text(page, shots, engine)
        check_tools(page)
        check_consent_and_upload(page, info, args.mp3, shots)
        secure = page.evaluate("window.isSecureContext")
        check("secure context (mikrofon diizinkan)", secure, url)
        if secure and engine == "webkit":
            print("   webkit: tidak ada mikrofon tiruan, alur rekam dilewati", flush=True)
        elif secure:
            check_recording(page, shots, engine)
            check_recording_limit(context, url, info)
        check_failures(context, url)
        check_phone_layout(browser, url, shots, engine)
        check_other_server_settings(context, url, info)

        check("tidak ada permintaan ke pihak luar", not outside, sorted(outside))
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
