"use strict";

const MAX_SECONDS = 60;
const $ = (id) => document.getElementById(id);
let recorder = null;
let chunks = [];
let timerId = null;
let startedAt = 0;

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
    throw new Error(`${err.message || response.statusText} (${err.code || response.status})`);
  }
  return body;
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
      caret.title = "something is missing here";
      caret.textContent = "^";
      el.append(caret);
    }
    cursor = Math.max(cursor, edit.end);
  }
  el.append(text.slice(cursor));
}

function renderCorrection(correction, extra = {}) {
  $("result").hidden = false;
  renderOriginal(correction.original, correction.edits);
  $("corrected").textContent = correction.corrected;
  const list = $("edits");
  list.replaceChildren();
  for (const edit of correction.edits) {
    const li = document.createElement("li");
    const from = edit.original || "(nothing)";
    const to = edit.corrected || "(remove)";
    li.textContent = `"${from}" → "${to}"` + (edit.description ? ` — ${edit.description}` : "");
    list.append(li);
  }
  $("praise").hidden = correction.changed;
  const warnings = [...(extra.warnings || []), ...(correction.warnings || [])];
  $("warnings").textContent = warnings.length ? `Notes: ${[...new Set(warnings)].join(", ")}` : "";
  $("timing").textContent = extra.timings ? `Processed in ${extra.timings.total_ms ?? "?"} ms` : "";
}

async function analyzeAudio(blob, filename) {
  if (!$("consent").checked) {
    setStatus("Please confirm parental consent before sending a recording.", true);
    return;
  }
  const form = new FormData();
  form.append("file", blob, filename);
  form.append("consent", "true");
  setStatus("Listening and checking grammar…");
  try {
    const body = await callApi("/v1/audio/analyze", { method: "POST", headers: headers(), body: form });
    if (body.correction) {
      renderCorrection(body.correction, { warnings: body.warnings, timings: body.timings_ms });
    } else {
      $("result").hidden = false;
      $("original").textContent = body.transcript.text || "(no speech detected)";
      $("corrected").textContent = "";
      $("edits").replaceChildren();
      $("warnings").textContent = `Notes: ${body.warnings.join(", ")}`;
    }
    setStatus("");
  } catch (err) {
    setStatus(err.message, true);
  }
}

async function checkText() {
  const text = $("textInput").value.trim();
  if (!text) return;
  setStatus("Checking grammar…");
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
    setStatus("Please confirm parental consent before recording.", true);
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
    $("recordBtn").textContent = "Stop";
    setStatus("Recording… speak clearly.");
  } catch (err) {
    setStatus(`Microphone unavailable: ${err.message}`, true);
  }
}

function stopRecording() {
  clearInterval(timerId);
  if (recorder && recorder.state !== "inactive") recorder.stop();
  $("recordBtn").textContent = "Start recording";
}

document.addEventListener("DOMContentLoaded", () => {
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
