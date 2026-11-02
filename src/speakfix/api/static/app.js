"use strict";

let maxUploadMb = 10; // replaced by the server's limit from /v1/info
let maxAudioSeconds = 60; // likewise
const $ = (id) => document.getElementById(id);
let recorder = null;
let chunks = [];
let timerId = null;
let startedAt = 0;

/** Learner-friendly text for the warning codes of the API (internal ones are not shown). */
const NOTES = {
  very_quiet_audio: "Suaranya terlalu pelan. Coba bicara lebih dekat ke mikrofon.",
  little_or_no_speech_detected: "Hampir tidak ada suara yang terdengar.",
  clipping_detected: "Suaranya terlalu keras sehingga sebagian terpotong.",
  empty_transcript: "Tidak ada kata yang terdengar.",
  correction_unavailable: "Koreksi sedang tidak tersedia, jadi hanya transkrip yang ditampilkan.",
  low_confidence_correction_skipped:
    "Sebagian saran tidak ditampilkan karena terlalu berbeda dari kalimat aslinya.",
  uncertain_correction_skipped: "Sebagian saran tidak ditampilkan karena sistem belum cukup yakin.",
};

/** "No mistakes found" would be misleading: the system held a suggestion back or heard too little. */
const NO_PRAISE = [
  "uncertain_correction_skipped",
  "low_confidence_correction_skipped",
  "little_or_no_speech_detected",
];

/** Messages for the API error codes; anything else shows the server's message. */
const ERRORS = {
  consent_required: "Centang persetujuan dulu sebelum mengirim rekaman.",
  unauthorized: "API key salah atau belum diisi.",
  rate_limited: "Terlalu banyak permintaan. Tunggu sebentar, lalu coba lagi.",
  server_busy: "Server sedang sibuk. Coba lagi beberapa detik lagi.",
  queue_full: "Server sedang sibuk. Coba lagi beberapa detik lagi.",
  resource_exhausted: "Server sedang penuh. Coba lagi beberapa detik lagi.",
  inference_failed: "Terjadi gangguan saat memproses. Coba lagi.",
  model_not_ready: "Model masih dimuat. Coba lagi sebentar lagi.",
  inference_timeout: "Pemrosesan terlalu lama. Coba rekaman yang lebih pendek.",
  payload_too_large: "File terlalu besar.",
  unsupported_media_type: "Format file tidak didukung. Gunakan rekaman audio (wav, mp3, m4a, webm).",
  invalid_audio: "Audio tidak bisa dibaca. Coba rekam ulang.",
  invalid_audio_duration: "Durasi rekaman terlalu pendek atau terlalu panjang (maksimal 60 detik).",
  invalid_text: "Teks kosong atau terlalu panjang.",
  network: "Koneksi ke server terputus. Periksa internet, lalu coba lagi.",
};

const MIC_ERRORS = {
  NotAllowedError: "Izin mikrofon ditolak. Izinkan mikrofon di pengaturan browser, atau unggah rekaman.",
  NotFoundError: "Mikrofon tidak ditemukan. Sambungkan mikrofon, atau unggah rekaman.",
  NotReadableError: "Mikrofon sedang dipakai aplikasi lain. Tutup aplikasi itu, lalu coba lagi.",
};

function setStatus(message, isError = false) {
  const el = $("status");
  el.textContent = message;
  el.className = isError ? "error" : "";
}

function headers() {
  const h = {};
  const key = $("apiKey").value.trim();
  if (key) h["X-API-Key"] = key;
  return h;
}

async function callApi(url, options) {
  let response;
  try {
    response = await fetch(url, options);
  } catch {
    throw new Error(ERRORS.network);
  }
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    const err = body.error || {};
    throw new Error(ERRORS[err.code] || `${err.message || response.statusText} (${err.code || response.status})`);
  }
  return body;
}

function showNotes(codes) {
  const notes = [...new Set(codes)].map((code) => NOTES[code]).filter(Boolean);
  $("warnings").textContent = notes.join(" ");
}

/** Render the original text with mistakes highlighted (DOM only, no innerHTML). */
function renderOriginal(text, edits) {
  const el = $("original");
  el.replaceChildren();
  let cursor = 0;
  for (const edit of [...edits].sort((a, b) => a.start - b.start)) {
    if (edit.start > cursor) el.append(text.slice(cursor, edit.start));
    if (edit.end > edit.start) {
      const mark = document.createElement("mark");
      mark.className = "mistake";
      mark.textContent = text.slice(edit.start, edit.end);
      el.append(mark);
    } else {
      const caret = document.createElement("span");
      caret.className = "missing";
      caret.title = "ada kata yang kurang di sini";
      caret.textContent = "^";
      el.append(caret);
    }
    cursor = Math.max(cursor, edit.end);
  }
  el.append(text.slice(cursor));
}

function describeEdit(edit) {
  let change = `"${edit.original}" → "${edit.corrected}"`;
  if (!edit.original) change = `Tambahkan "${edit.corrected}"`;
  if (!edit.corrected) change = `Hapus "${edit.original}"`;
  return edit.description ? `${change} — ${edit.description}` : change;
}

function renderCorrection(correction, extra = {}) {
  $("result").hidden = false;
  renderOriginal(correction.original, correction.edits);
  $("corrected").textContent = correction.changed ? correction.corrected : "";
  $("corrected").hidden = !correction.changed;
  $("correctedLabel").hidden = !correction.changed;
  const list = $("edits");
  list.replaceChildren();
  for (const edit of correction.edits) {
    const li = document.createElement("li");
    li.textContent = describeEdit(edit);
    list.append(li);
  }
  const notes = [...(extra.warnings || []), ...(correction.warnings || [])];
  $("praise").hidden = correction.changed || notes.some((code) => NO_PRAISE.includes(code));
  showNotes(notes);
  $("timing").textContent = extra.timings ? `Diproses dalam ${extra.timings.total_ms ?? "?"} ms` : "";
  $("result").scrollIntoView({ behavior: "smooth", block: "nearest" }); // far below on a phone
}

async function analyzeAudio(blob, filename) {
  if (!$("consent").checked) {
    setStatus(ERRORS.consent_required, true);
    return;
  }
  if (blob.size > maxUploadMb * 1024 * 1024) {
    setStatus(`File terlalu besar (maksimal ${maxUploadMb} MB).`, true);
    return;
  }
  const form = new FormData();
  form.append("file", blob, filename);
  form.append("consent", "true");
  setStatus("Mendengarkan dan memeriksa grammar…");
  try {
    const body = await callApi("/v1/audio/analyze", { method: "POST", headers: headers(), body: form });
    if (body.correction) {
      renderCorrection(body.correction, { warnings: body.warnings, timings: body.timings_ms });
    } else {
      $("result").hidden = false;
      $("original").textContent = body.transcript.text || "(tidak ada suara yang terdeteksi)";
      $("corrected").textContent = "";
      $("corrected").hidden = true;
      $("correctedLabel").hidden = true;
      $("edits").replaceChildren();
      $("praise").hidden = true;
      $("timing").textContent = "";
      showNotes(body.warnings);
      $("result").scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
    setStatus("");
  } catch (err) {
    setStatus(err.message, true);
  }
}

async function checkText() {
  const text = $("textInput").value.trim();
  if (!text) return;
  setStatus("Memeriksa grammar…");
  try {
    const body = await callApi("/v1/text/correct", {
      method: "POST",
      headers: { ...headers(), "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    renderCorrection(body.correction, { timings: body.timings_ms });
    setStatus("");
  } catch (err) {
    setStatus(err.message, true);
  }
}

function pickMimeType() {
  const candidates = ["audio/webm;codecs=opus", "audio/ogg;codecs=opus", "audio/mp4", "audio/webm"];
  return candidates.find((t) => window.MediaRecorder && MediaRecorder.isTypeSupported(t)) || "";
}

function formatTime(seconds) {
  return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`;
}

/** A second under the server's limit: a recording runs slightly longer than the timer shows. */
function recordingLimit() {
  return Math.max(1, Math.floor(maxAudioSeconds) - 1);
}

function updateTimer() {
  const seconds = Math.floor((Date.now() - startedAt) / 1000);
  $("timer").textContent = `${formatTime(seconds)} / ${formatTime(recordingLimit())}`;
  if (seconds >= recordingLimit()) stopRecording();
}

async function startRecording() {
  if (!$("consent").checked) {
    setStatus(ERRORS.consent_required, true);
    return;
  }
  if (!window.isSecureContext) {
    setStatus(
      "Browser hanya mengizinkan mikrofon lewat alamat https:// atau localhost. " +
        "Buka lewat link https, atau unggah rekaman.",
      true
    );
    return;
  }
  if (!navigator.mediaDevices || !window.MediaRecorder) {
    setStatus("Browser ini belum mendukung perekaman. Silakan unggah file rekaman.", true);
    return;
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    });
    const mimeType = pickMimeType();
    recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
    chunks = [];
    recorder.ondataavailable = (e) => e.data.size && chunks.push(e.data);
    recorder.onstop = () => {
      // Also runs when the browser ends the recording itself, e.g. a phone locking its screen.
      clearInterval(timerId);
      $("recordBtn").textContent = "Mulai rekam";
      stream.getTracks().forEach((t) => t.stop());
      const blob = new Blob(chunks, { type: recorder.mimeType || "audio/webm" });
      const player = $("playback");
      player.src = URL.createObjectURL(blob);
      player.hidden = false;
      const ext = blob.type.includes("mp4") ? "m4a" : blob.type.includes("ogg") ? "ogg" : "webm";
      analyzeAudio(blob, `recording.${ext}`);
    };
    recorder.start();
    startedAt = Date.now();
    timerId = setInterval(updateTimer, 250);
    $("recordBtn").textContent = "Berhenti";
    setStatus("Merekam… bicaralah dengan jelas.");
  } catch (err) {
    setStatus(MIC_ERRORS[err.name] || `Mikrofon tidak bisa dipakai: ${err.message}`, true);
  }
}

function stopRecording() {
  if (recorder && recorder.state !== "inactive") recorder.stop();
}

/** Demo mode (public demo): adult voices only, adapted consent text. */
async function applyServerSettings() {
  try {
    const info = await callApi("/v1/info", {});
    maxUploadMb = info.limits?.max_upload_mb ?? maxUploadMb;
    maxAudioSeconds = info.limits?.max_audio_seconds ?? maxAudioSeconds;
    ERRORS.invalid_audio_duration =
      `Durasi rekaman terlalu pendek atau terlalu panjang (maksimal ${maxAudioSeconds} detik).`;
    $("timer").textContent = `0:00 / ${formatTime(recordingLimit())}`;
    $("apiKeyBox").hidden = !info.ui?.api_key_required;
    if (info.ui && info.ui.demo_mode) {
      $("demoBanner").hidden = false;
      $("consentText").textContent =
        "Saya berusia 18 tahun ke atas, merekam suara saya sendiri, dan setuju rekaman ini " +
        "diproses. Audio hanya diproses di memori dan tidak pernah disimpan.";
    }
  } catch {
    // The page still works with the default (parental consent) wording.
  }
}

document.addEventListener("DOMContentLoaded", () => {
  applyServerSettings();
  $("recordBtn").addEventListener("click", () =>
    recorder && recorder.state === "recording" ? stopRecording() : startRecording()
  );
  $("fileInput").addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (file) analyzeAudio(file, file.name);
    e.target.value = "";
  });
  $("checkBtn").addEventListener("click", checkText);
});
