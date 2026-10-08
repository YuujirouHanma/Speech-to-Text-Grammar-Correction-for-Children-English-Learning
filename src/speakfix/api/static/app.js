"use strict";

const MAX_SECONDS = 60;
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

/** Messages for the API error codes; anything else shows the server's message. */
const ERRORS = {
  consent_required: "Centang persetujuan dulu sebelum mengirim rekaman.",
  unauthorized: "API key salah atau belum diisi.",
  rate_limited: "Terlalu banyak permintaan. Tunggu sebentar, lalu coba lagi.",
  server_busy: "Server sedang sibuk. Coba lagi beberapa detik lagi.",
  queue_full: "Server sedang sibuk. Coba lagi beberapa detik lagi.",
  model_not_ready: "Model masih dimuat. Coba lagi sebentar lagi.",
  inference_timeout: "Pemrosesan terlalu lama. Coba rekaman yang lebih pendek.",
  payload_too_large: "File terlalu besar.",
  unsupported_media_type: "Format file tidak didukung. Gunakan rekaman audio (wav, mp3, m4a, webm).",
  invalid_audio: "Audio tidak bisa dibaca. Coba rekam ulang.",
  invalid_audio_duration: "Durasi rekaman terlalu pendek atau terlalu panjang (maksimal 60 detik).",
  invalid_text: "Teks kosong atau terlalu panjang.",
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
  const response = await fetch(url, options);
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
  $("praise").hidden = correction.changed;
  showNotes([...(extra.warnings || []), ...(correction.warnings || [])]);
  $("timing").textContent = extra.timings ? `Diproses dalam ${extra.timings.total_ms ?? "?"} ms` : "";
}

async function analyzeAudio(blob, filename) {
  if (!$("consent").checked) {
    setStatus(ERRORS.consent_required, true);
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
      showNotes(body.warnings);
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

function updateTimer() {
  const seconds = Math.floor((Date.now() - startedAt) / 1000);
  $("timer").textContent = `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")} / 0:60`;
  if (seconds >= MAX_SECONDS) stopRecording();
}

async function startRecording() {
  if (!$("consent").checked) {
    setStatus(ERRORS.consent_required, true);
    return;
  }
  if (!window.isSecureContext || !navigator.mediaDevices) {
    setStatus(
      "Browser hanya mengizinkan mikrofon lewat alamat https:// atau localhost. " +
        "Buka lewat link https, atau unggah rekaman.",
      true
    );
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
    setStatus(`Mikrofon tidak bisa dipakai: ${err.message}`, true);
  }
}

function stopRecording() {
  clearInterval(timerId);
  if (recorder && recorder.state !== "inactive") recorder.stop();
  $("recordBtn").textContent = "Mulai rekam";
}

/** Demo mode (public demo): adult voices only, adapted consent text. */
async function applyServerSettings() {
  try {
    const info = await callApi("/v1/info", {});
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
