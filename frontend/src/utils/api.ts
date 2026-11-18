import type {
  AnalysisResult,
  AudioResponse,
  Correction,
  ServerInfo,
  TextResponse,
} from '../types';

/** Learner-friendly text for the warning codes of the API (internal ones are not shown). */
const NOTES: Record<string, string> = {
  very_quiet_audio: 'Suaranya terlalu pelan. Coba bicara lebih dekat ke mikrofon.',
  little_or_no_speech_detected: 'Hampir tidak ada suara yang terdengar.',
  clipping_detected: 'Suaranya terlalu keras sehingga sebagian terpotong.',
  empty_transcript: 'Tidak ada kata yang terdengar.',
  correction_unavailable: 'Koreksi sedang tidak tersedia, jadi hanya transkrip yang ditampilkan.',
  low_confidence_correction_skipped:
    'Sebagian saran tidak ditampilkan karena terlalu berbeda dari kalimat aslinya.',
  uncertain_correction_skipped: 'Sebagian saran tidak ditampilkan karena sistem belum cukup yakin.',
};

/** "No mistakes found" would be misleading: the system held a suggestion back or heard too little. */
const NO_PRAISE = [
  'uncertain_correction_skipped',
  'low_confidence_correction_skipped',
  'little_or_no_speech_detected',
];

/** Messages for the API error codes; anything else shows the server's message. */
export const ERRORS: Record<string, string> = {
  consent_required: 'Centang persetujuan dulu sebelum mengirim rekaman.',
  unauthorized: 'API key salah atau belum diisi.',
  rate_limited: 'Terlalu banyak permintaan. Tunggu sebentar, lalu coba lagi.',
  server_busy: 'Server sedang sibuk. Coba lagi beberapa detik lagi.',
  queue_full: 'Server sedang sibuk. Coba lagi beberapa detik lagi.',
  resource_exhausted: 'Server sedang penuh. Coba lagi beberapa detik lagi.',
  inference_failed: 'Terjadi gangguan saat memproses. Coba lagi.',
  model_not_ready: 'Model masih dimuat. Coba lagi sebentar lagi.',
  inference_timeout: 'Pemrosesan terlalu lama. Coba rekaman yang lebih pendek.',
  payload_too_large: 'File terlalu besar.',
  unsupported_media_type: 'Format file tidak didukung. Gunakan rekaman audio (wav, mp3, m4a, webm).',
  invalid_audio: 'Audio tidak bisa dibaca. Coba rekam ulang.',
  invalid_audio_duration: 'Durasi rekaman terlalu pendek atau terlalu panjang (maksimal 60 detik).',
  invalid_text: 'Teks kosong atau terlalu panjang.',
  network: 'Koneksi ke server terputus. Periksa internet, lalu coba lagi.',
};

async function callApi<T>(url: string, options?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(url, options);
  } catch {
    throw new Error(ERRORS.network);
  }
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    const err = body.error || {};
    throw new Error(
      ERRORS[err.code] || `${err.message || response.statusText} (${err.code || response.status})`
    );
  }
  return body as T;
}

function authHeaders(apiKey: string): Record<string, string> {
  const key = apiKey.trim();
  return key ? { 'X-API-Key': key } : {};
}

function describeNotes(codes: string[]): string {
  return [...new Set(codes)]
    .map((code) => NOTES[code])
    .filter(Boolean)
    .join(' ');
}

function toResult(
  correction: Correction,
  warnings: string[],
  timings: Record<string, number>
): AnalysisResult {
  const codes = [...warnings, ...correction.warnings];
  return {
    original: correction.original,
    corrected: correction.corrected,
    changed: correction.changed,
    edits: correction.edits,
    notes: describeNotes(codes),
    praise: !correction.changed && !codes.some((code) => NO_PRAISE.includes(code)),
    totalMs: timings.total_ms ?? null,
  };
}

export function getInfo(): Promise<ServerInfo> {
  return callApi<ServerInfo>('/v1/info');
}

/** Spends the wait for the models while the visitor is still reading, not after they press a button. */
export function warmUp(apiKey: string): void {
  callApi('/v1/warmup', { method: 'POST', headers: authHeaders(apiKey) }).catch(() => {});
}

export async function sendText(text: string, apiKey: string): Promise<AnalysisResult> {
  const body = await callApi<TextResponse>('/v1/text/correct', {
    method: 'POST',
    headers: { ...authHeaders(apiKey), 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
  });
  return toResult(body.correction, [], body.timings_ms);
}

export async function sendAudio(
  blob: Blob,
  filename: string,
  apiKey: string
): Promise<{ result: AnalysisResult; transcript: string }> {
  const form = new FormData();
  form.append('file', blob, filename);
  form.append('consent', 'true');
  const body = await callApi<AudioResponse>('/v1/audio/analyze', {
    method: 'POST',
    headers: authHeaders(apiKey),
    body: form,
  });
  const transcript = body.transcript.text;
  if (body.correction) {
    return { result: toResult(body.correction, body.warnings, body.timings_ms), transcript };
  }
  return {
    result: {
      original: transcript || '(tidak ada suara yang terdeteksi)',
      corrected: transcript,
      changed: false,
      edits: [],
      notes: describeNotes(body.warnings),
      praise: false,
      totalMs: body.timings_ms.total_ms ?? null,
    },
    transcript,
  };
}
