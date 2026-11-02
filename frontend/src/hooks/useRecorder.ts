import { useCallback, useRef, useState } from 'react';
import { playChime } from '../utils/audio';

const MIC_ERRORS: Record<string, string> = {
  NotAllowedError: 'Izin mikrofon ditolak. Izinkan mikrofon di pengaturan browser, atau unggah rekaman.',
  NotFoundError: 'Mikrofon tidak ditemukan. Sambungkan mikrofon, atau unggah rekaman.',
  NotReadableError: 'Mikrofon sedang dipakai aplikasi lain. Tutup aplikasi itu, lalu coba lagi.',
};

const CHIME_MS = 200;

function pickMimeType(): string {
  const candidates = ['audio/webm;codecs=opus', 'audio/ogg;codecs=opus', 'audio/mp4', 'audio/webm'];
  return candidates.find((t) => MediaRecorder.isTypeSupported?.(t)) || '';
}

function extensionFor(type: string): string {
  if (type.includes('mp4')) return 'm4a';
  return type.includes('ogg') ? 'ogg' : 'webm';
}

/**
 * Records the microphone in the browser. The clip goes to the server only when it ends, so there
 * is no live text while speaking. `start` throws an Error with a message to show the visitor.
 */
export function useRecorder(
  limitSeconds: number,
  onRecorded: (blob: Blob, filename: string) => void
) {
  const [isRecording, setIsRecording] = useState(false);
  const [seconds, setSeconds] = useState(0);
  const [stream, setStream] = useState<MediaStream | null>(null);

  const recorderRef = useRef<MediaRecorder | null>(null);
  const startingRef = useRef(false);
  const discardRef = useRef(false);
  const timerRef = useRef<number | undefined>(undefined);
  const limitRef = useRef(limitSeconds);
  const onRecordedRef = useRef(onRecorded);
  limitRef.current = limitSeconds;
  onRecordedRef.current = onRecorded;

  const stop = useCallback(() => {
    if (recorderRef.current?.state === 'recording') recorderRef.current.stop();
  }, []);

  /** Ends the recording without sending anything. */
  const cancel = useCallback(() => {
    discardRef.current = true;
    stop();
  }, [stop]);

  const start = useCallback(async () => {
    if (startingRef.current || recorderRef.current?.state === 'recording') return;
    if (!window.isSecureContext) {
      throw new Error(
        'Browser hanya mengizinkan mikrofon lewat alamat https:// atau localhost. ' +
          'Buka lewat link https, atau unggah rekaman.'
      );
    }
    if (!navigator.mediaDevices || !window.MediaRecorder) {
      throw new Error('Browser ini belum mendukung perekaman. Silakan unggah file rekaman.');
    }

    startingRef.current = true;
    let micStream: MediaStream | null = null;
    try {
      micStream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
      });
      playChime('start');
      await new Promise((resolve) => setTimeout(resolve, CHIME_MS)); // keeps it out of the recording

      const tracks = micStream.getTracks();
      const mimeType = pickMimeType();
      const recorder = new MediaRecorder(micStream, mimeType ? { mimeType } : undefined);
      const chunks: Blob[] = [];
      recorder.ondataavailable = (e) => {
        if (e.data.size) chunks.push(e.data);
      };
      recorder.onstop = () => {
        // Also runs when the browser ends the recording itself, e.g. a phone locking its screen.
        window.clearInterval(timerRef.current);
        tracks.forEach((t) => t.stop());
        setStream(null);
        setIsRecording(false);
        playChime('stop');
        if (discardRef.current) return;
        const type = recorder.mimeType || 'audio/webm';
        onRecordedRef.current(new Blob(chunks, { type }), `recording.${extensionFor(type)}`);
      };

      discardRef.current = false;
      recorder.start();
      recorderRef.current = recorder;
      const startedAt = Date.now();
      setSeconds(0);
      setStream(micStream);
      setIsRecording(true);
      timerRef.current = window.setInterval(() => {
        const elapsed = Math.floor((Date.now() - startedAt) / 1000);
        setSeconds(elapsed);
        if (elapsed >= limitRef.current) stop();
      }, 250);
    } catch (err) {
      micStream?.getTracks().forEach((t) => t.stop());
      const { name, message } = err as Error;
      throw new Error(MIC_ERRORS[name] || `Mikrofon tidak bisa dipakai: ${message}`);
    } finally {
      startingRef.current = false;
    }
  }, [stop]);

  return { isRecording, seconds, stream, start, stop, cancel };
}
