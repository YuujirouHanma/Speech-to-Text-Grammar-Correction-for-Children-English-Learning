import { useEffect, useLayoutEffect, useRef, useState } from 'react';
import confetti from 'canvas-confetti';
import {
  AlertCircle,
  FileAudio,
  Loader2,
  Mic,
  MicOff,
  ShieldAlert,
  Sparkles,
  Trash2,
} from 'lucide-react';

import type { AnalysisResult, HistoryRecord, InputSource } from './types';
import { useRecorder } from './hooks/useRecorder';
import { ERRORS, getInfo, sendAudio, sendText, warmUp } from './utils/api';
import { isSpeechSynthesisSupported, playChime, speakText, stopSpeaking } from './utils/audio';
import { copyText, readStored, writeStored } from './utils/browser';
import { computeWordDiff, formatWordCount } from './utils/diff';

import { TopBar } from './components/TopBar';
import { AudioWaveform } from './components/AudioWaveform';
import { DiffViewer } from './components/DiffViewer';
import { ExplanationDrawer } from './components/ExplanationDrawer';
import { HistoryModal } from './components/HistoryModal';
import { CheatsheetModal } from './components/CheatsheetModal';
import { ShortcutsModal } from './components/ShortcutsModal';
import { AudioUploadModal } from './components/AudioUploadModal';

const PROJECT_TITLE = 'Speech to Text Grammar Correction for Children English Learning';
const REPOSITORY_URL =
  'https://github.com/YuujirouHanma/Speech-to-Text-Grammar-Correction-for-Children-English-Learning';

const PARENT_CONSENT =
  'Saya orang tua/wali, atau guru yang sudah mendapat izin dari orang tua, dan setuju rekaman ini ' +
  'diproses. Audio hanya diproses di memori dan tidak pernah disimpan.';
const DEMO_CONSENT =
  'Saya berusia 18 tahun ke atas, merekam suara saya sendiri, dan setuju rekaman ini diproses. ' +
  'Audio hanya diproses di memori dan tidak pernah disimpan.';

const RECORDING_STATUS = 'Merekam… bicaralah dengan jelas.';
/** The first request after a server has slept can also load the models: say so instead of looking stuck. */
const SLOW_HINT =
  'Server baru dinyalakan dan sedang memuat model. Permintaan pertama bisa lebih lama, mohon tunggu…';
const SLOW_HINT_AFTER_MS = 8000;

// The page's CSP refuses blob: workers, so the confetti is drawn on the main thread.
const celebrate = confetti.create(undefined, {
  useWorker: false,
  resize: true,
  disableForReducedMotion: true,
});

function formatTime(seconds: number): string {
  return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, '0')}`;
}

// Sample presets for quick testing
const samplePresets = [
  {
    label: 'Job Interview Intro',
    text: 'Hello, I wants to applies for this position because I have many experience in software engineering.',
  },
  {
    label: 'Business Email',
    text: 'Dear Sir, I am write this email to discuss about the contract renewal for next month.',
  },
  {
    label: 'Daily Chat',
    text: 'Yesterday me and my friend goes to the restaurant and eat very delicious food.',
  },
  {
    label: 'Academic Argument',
    text: 'The data clearly shows that climate change impact on agricultural yield significant.',
  },
];

export default function App() {
  // Theme state
  const [darkMode, setDarkMode] = useState<boolean>(() => {
    const saved = readStored('vg_theme');
    if (saved) return saved === 'dark';
    return window.matchMedia('(prefers-color-scheme: dark)').matches;
  });

  // Server settings
  const [limits, setLimits] = useState({ maxUploadMb: 10, maxAudioSeconds: 60, maxTextChars: 2000 });
  const [demoMode, setDemoMode] = useState<boolean>(false);
  const [apiKeyRequired, setApiKeyRequired] = useState<boolean>(false);
  const apiKeyRef = useRef<string>('');

  // Editor states
  const [inputText, setInputText] = useState<string>('');
  const [previousText, setPreviousText] = useState<string | null>(null);
  const [speechRate, setSpeechRate] = useState<number>(1.0);

  // Voice recording states
  const [consent, setConsent] = useState<boolean>(false);
  const [speechError, setSpeechError] = useState<string | null>(null);
  const [playbackUrl, setPlaybackUrl] = useState<string | null>(null);
  const consentRef = useRef<HTMLInputElement | null>(null);

  // Audio Playback (TTS) state
  const [isPlayingAudio, setIsPlayingAudio] = useState<boolean>(false);

  // Analysis Result state
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
  const [progress, setProgress] = useState<string>('');
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const retryRef = useRef<(() => void) | null>(null);
  const resultColumnRef = useRef<HTMLDivElement | null>(null);
  const errorRef = useRef<HTMLDivElement | null>(null);

  // History state: kept in this browser only
  const [history, setHistory] = useState<HistoryRecord[]>(() => {
    try {
      const saved = readStored('vg_history');
      return saved ? JSON.parse(saved) : [];
    } catch (e) {
      return [];
    }
  });

  // Modal states
  const [isHistoryOpen, setIsHistoryOpen] = useState<boolean>(false);
  const [isCheatsheetOpen, setIsCheatsheetOpen] = useState<boolean>(false);
  const [isShortcutsOpen, setIsShortcutsOpen] = useState<boolean>(false);
  const [isUploadOpen, setIsUploadOpen] = useState<boolean>(false);

  // Toast notification for user control (Heuristic #3)
  const [toastMessage, setToastMessage] = useState<string | null>(null);
  const [canUndoClear, setCanUndoClear] = useState<boolean>(false);
  const toastTimerRef = useRef<number | undefined>(undefined);

  // A second under the server's limit: a recording runs slightly longer than the timer shows.
  const recordingLimit = Math.max(1, Math.floor(limits.maxAudioSeconds) - 1);
  const recorder = useRecorder(recordingLimit, (blob, filename) => {
    setPlaybackUrl((previous) => {
      if (previous) URL.revokeObjectURL(previous);
      return URL.createObjectURL(blob);
    });
    analyzeAudio(blob, filename);
  });
  const { isRecording } = recorder;
  const canSpeak = isSpeechSynthesisSupported();
  const consentText = demoMode ? DEMO_CONSENT : PARENT_CONSENT;

  // Apply dark mode class to html element
  useLayoutEffect(() => {
    document.documentElement.classList.toggle('dark', darkMode);
    writeStored('vg_theme', darkMode ? 'dark' : 'light');
  }, [darkMode]);

  // Save history to localStorage
  useEffect(() => {
    writeStored('vg_history', JSON.stringify(history));
  }, [history]);

  // Limits and wording come from the server; the page still works with the defaults.
  useEffect(() => {
    getInfo()
      .then((info) => {
        setLimits({
          maxUploadMb: info.limits.max_upload_mb,
          maxAudioSeconds: info.limits.max_audio_seconds,
          maxTextChars: info.limits.max_text_chars,
        });
        ERRORS.invalid_audio_duration = `Durasi rekaman terlalu pendek atau terlalu panjang (maksimal ${info.limits.max_audio_seconds} detik).`;
        setDemoMode(info.ui.demo_mode);
        setApiKeyRequired(info.ui.api_key_required);
        if (info.ui.load_models_on_demand && !(info.corrector && info.stt)) {
          warmUp(apiKeyRef.current);
        }
      })
      .catch(() => {});
  }, []);

  // The result is far below the buttons on a phone
  useEffect(() => {
    if (isAnalyzing || analysisResult) {
      resultColumnRef.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  }, [isAnalyzing, analysisResult]);

  useEffect(() => {
    if (errorMessage) errorRef.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }, [errorMessage]);

  // Show transient toast
  const triggerToast = (msg: string, undoable: boolean = false) => {
    window.clearTimeout(toastTimerRef.current);
    setToastMessage(msg);
    setCanUndoClear(undoable);
    toastTimerRef.current = window.setTimeout(() => {
      setToastMessage(null);
      setCanUndoClear(false);
    }, 4500);
  };

  // The server refuses a voice recording without consent, so the page asks first.
  const hasConsent = (): boolean => {
    if (consent) return true;
    setSpeechError(ERRORS.consent_required);
    consentRef.current?.focus();
    return false;
  };

  // Toggle Recording
  const toggleRecording = async () => {
    if (isRecording) {
      recorder.stop();
      return;
    }
    if (isAnalyzing || !hasConsent()) return;
    setSpeechError(null);
    setErrorMessage(null);
    stopSpeaking();
    setIsPlayingAudio(false);
    try {
      await recorder.start();
    } catch (err) {
      setSpeechError((err as Error).message);
    }
  };

  // Shared by typed text and voice: progress, result, history and retry
  const runAnalysis = async (
    source: InputSource,
    progressText: string,
    send: () => Promise<AnalysisResult>
  ) => {
    retryRef.current = () => runAnalysis(source, progressText, send);
    setIsAnalyzing(true);
    setErrorMessage(null);
    setProgress(progressText);
    celebrate.reset(); // the praise of the previous sentence must not fall onto the next result
    stopSpeaking();
    setIsPlayingAudio(false);
    const hint =window.setTimeout(() => setProgress(SLOW_HINT), SLOW_HINT_AFTER_MS);

    try {
      const result = await send();
      setAnalysisResult(result);
      playChime('success');

      // Celebration confetti if nothing is wrong and no suggestion was held back
      if (result.praise) {
        celebrate({
          particleCount: 90,
          spread: 75,
          origin: { y: 0.6 },
          colors: ['#dc2626', '#ef4444', '#b91c1c', '#ffffff'],
        });
      }

      // Add to history
      if (result.corrected) {
        const newRecord: HistoryRecord = {
          id: Date.now().toString(),
          timestamp: Date.now(),
          source,
          originalText: result.original,
          correctedText: result.corrected,
          changesCount: result.edits.length,
        };
        setHistory((prev) => [newRecord, ...prev.slice(0, 49)]); // keep latest 50
      }
    } catch (err) {
      setErrorMessage((err as Error).message);
    } finally {
      window.clearTimeout(hint);
      setIsAnalyzing(false);
      setProgress('');
    }
  };

  // Analyze typed text
  const analyzeText = (textToAnalyze?: string) => {
    const targetText = (textToAnalyze ?? inputText).trim();
    if (!targetText) {
      triggerToast('Silakan masukkan kalimat terlebih dahulu melalui suara atau ketikan.');
      return;
    }
    if (isAnalyzing) return;

    // A recording in progress is dropped: stopping it normally would send it too
    recorder.cancel();

    void runAnalysis('text', 'Memeriksa grammar…', () => sendText(targetText, apiKeyRef.current));
  };

  // Analyze a recording or an uploaded file
  const analyzeAudio = (blob: Blob, filename: string) => {
    if (!hasConsent()) return;
    if (blob.size > limits.maxUploadMb * 1024 * 1024) {
      setSpeechError(`File terlalu besar (maksimal ${limits.maxUploadMb} MB).`);
      return;
    }
    setSpeechError(null);
    void runAnalysis('voice', 'Mendengarkan dan memeriksa grammar…', async () => {
      const { result, transcript } = await sendAudio(blob, filename, apiKeyRef.current);
      if (transcript) setInputText(transcript);
      return result;
    });
  };

  // Clear Text with Undo capability (Heuristic #3: User Control & Freedom)
  const handleClearText = () => {
    if (!inputText) return;
    setPreviousText(inputText);
    setInputText('');
    setAnalysisResult(null);
    triggerToast('Teks berhasil dibersihkan.', true);
  };

  // Undo Clear
  const handleUndoClear = () => {
    if (previousText) {
      setInputText(previousText);
      setPreviousText(null);
      setToastMessage(null);
      setCanUndoClear(false);
    }
  };

  // Revert back to original spoken text
  const handleRevert = () => {
    if (analysisResult?.corrected) {
      setInputText(analysisResult.original);
      triggerToast('Teks dikembalikan ke kalimat asli.');
    }
  };

  // Text to Speech playback
  const handleSpeak = (text: string) => {
    if (!canSpeak || !text) return;
    if (isPlayingAudio) {
      stopSpeaking();
      setIsPlayingAudio(false);
      return;
    }

    setIsPlayingAudio(true);
    speakText(text, speechRate, () => setIsPlayingAudio(false));
  };

  // Sample prompt insertion (Heuristic #6: Recognition rather than recall)
  const handleInsertPreset = (presetText: string) => {
    setInputText(presetText);
    setAnalysisResult(null);
    analyzeText(presetText);
  };

  // Restore history record
  const handleRestoreRecord = (rec: HistoryRecord) => {
    setInputText(rec.originalText);
    analyzeText(rec.originalText);
    triggerToast('Riwayat berhasil dipulihkan ke editor.');
  };

  // Delete single history item
  const handleDeleteRecord = (id: string) => {
    setHistory((prev) => prev.filter((item) => item.id !== id));
  };

  // Clear all history
  const handleClearAllHistory = () => {
    if (window.confirm('Apakah Anda yakin ingin menghapus seluruh riwayat?')) {
      setHistory([]);
      triggerToast('Semua riwayat telah dihapus.');
    }
  };

  // Keyboard Shortcuts (Heuristic #7: Flexibility and Efficiency)
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setIsHistoryOpen(false);
        setIsCheatsheetOpen(false);
        setIsShortcutsOpen(false);
        setIsUploadOpen(false);
        if (isRecording) {
          recorder.cancel();
          triggerToast('Rekaman dibatalkan dan tidak dikirim.');
        }
      }

      // Alt + V : Toggle voice recording
      if (e.altKey && (e.key === 'v' || e.key === 'V')) {
        e.preventDefault();
        void toggleRecording();
      }

      // Ctrl + Enter or Cmd + Enter : Analyze grammar
      if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault();
        if (inputText.trim() && !isAnalyzing) {
          analyzeText();
        }
      }

      // Ctrl + Shift + C : Copy corrected text
      if ((e.ctrlKey || e.metaKey) && e.shiftKey && (e.key === 'c' || e.key === 'C')) {
        if (analysisResult?.corrected) {
          e.preventDefault();
          void copyText(analysisResult.corrected).then((copied) => {
            if (copied) triggerToast('Teks hasil koreksi tersalin ke clipboard!');
          });
        }
      }

      // Ctrl + Shift + P : Play/Stop speech TTS
      if ((e.ctrlKey || e.metaKey) && e.shiftKey && (e.key === 'p' || e.key === 'P')) {
        if (analysisResult?.corrected) {
          e.preventDefault();
          handleSpeak(analysisResult.corrected);
        }
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  });

  // Compute live word & char counts
  const inputStats = formatWordCount(inputText);

  // Compute Diff tokens if analysis is available
  const diffTokens = analysisResult
    ? computeWordDiff(analysisResult.original, analysisResult.corrected || analysisResult.original)
    : [];

  return (
    <div className="min-h-screen flex flex-col bg-neutral-100 text-neutral-900 dark:bg-black dark:text-neutral-100 transition-colors">
      {/* 1. Top Bar Navigation (Strict 3-Zone Contract) */}
      <TopBar
        darkMode={darkMode}
        onToggleTheme={() => setDarkMode(!darkMode)}
        onOpenHistory={() => setIsHistoryOpen(true)}
        onOpenCheatsheet={() => setIsCheatsheetOpen(true)}
        onOpenShortcuts={() => setIsShortcutsOpen(true)}
      />

      {/* Main Container */}
      <main className="flex-1 mx-auto w-full max-w-7xl px-4 sm:px-6 lg:px-8 py-8 space-y-8">
        {/* Hero & Workspace Header */}
        <section className="relative flex flex-col md:flex-row md:items-end justify-between gap-6 pb-4 border-b border-neutral-200 dark:border-red-950/80">
          <div className="space-y-1.5 max-w-3xl">
            <div className="text-xs font-bold uppercase tracking-widest text-red-600 dark:text-red-500 font-brand">
              {PROJECT_TITLE}
            </div>
            <h1 className="text-3xl sm:text-4xl lg:text-5xl font-black tracking-tight font-brand uppercase text-neutral-900 dark:text-white">
              Bicara Alami, <span className="text-red-600 dark:text-red-500">Sempurnakan</span> Grammar
            </h1>
            <p className="text-sm text-neutral-600 dark:text-neutral-400 leading-relaxed font-medium">
              Ucapkan atau ketik kalimat bahasa Inggris, lalu lihat saran perbaikannya. Suara
              ditranskripsi oleh Whisper, dan grammar dikoreksi oleh model T5 yang dilatih untuk proyek ini.
            </p>
          </div>

          {/* TTS Speech Rate */}
          {canSpeak && (
            <div className="flex flex-col gap-1">
              <label
                htmlFor="speechRate"
                className="text-[11px] font-bold uppercase tracking-wider text-neutral-500 dark:text-neutral-400 font-brand"
              >
                Kecepatan Audio
              </label>
              <select
                id="speechRate"
                value={speechRate}
                onChange={(e) => setSpeechRate(parseFloat(e.target.value))}
                className="h-9 rounded-lg border border-neutral-200 bg-white px-3 text-xs font-bold text-neutral-800 shadow-xs focus:border-red-500 focus:outline-none dark:border-neutral-800 dark:bg-neutral-950 dark:text-neutral-200 cursor-pointer"
              >
                <option value="0.75">0.75x (Lambat)</option>
                <option value="1">1.0x (Normal)</option>
                <option value="1.25">1.25x (Cepat)</option>
              </select>
            </div>
          )}
        </section>

        {/* Public demo: adult voices only */}
        {demoMode && (
          <p
            id="demoBanner"
            className="flex items-start gap-3 rounded-xl border border-amber-300 bg-amber-50 p-3.5 text-xs leading-relaxed text-amber-900 dark:border-amber-900/60 dark:bg-amber-950/40 dark:text-amber-200"
          >
            <ShieldAlert className="h-4 w-4 shrink-0 mt-0.5" />
            <span>
              <strong className="font-bold">Mode demo.</strong> Gunakan suara orang dewasa saja, jangan
              merekam suara anak. Rekaman hanya diproses di memori dan tidak disimpan.
            </span>
          </p>
        )}

        {/* Live Audio Recording Central Console (Heuristic #1: Visibility of System Status) */}
        <section className="relative overflow-hidden rounded-2xl border border-neutral-200 bg-white p-6 shadow-sm dark:border-red-950/70 dark:bg-neutral-950 dark:glow-red transition-all">
          <div className="flex flex-col sm:flex-row items-center justify-between gap-6">
            {/* Left: Mic Activation and Real-Time State */}
            <div className="flex items-center gap-5">
              <button
                id="recordBtn"
                type="button"
                onClick={toggleRecording}
                disabled={isAnalyzing}
                className={`relative flex h-16 w-16 shrink-0 items-center justify-center rounded-2xl font-bold transition-all cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed ${
                  isRecording
                    ? 'bg-red-600 text-white shadow-xl shadow-red-600/50 scale-105 animate-pulse glow-red-lg'
                    : 'bg-red-600 text-white hover:bg-red-500 shadow-lg shadow-red-600/30 hover:scale-102'
                }`}
                title={isRecording ? 'Klik untuk Menghentikan Rekaman' : 'Klik untuk Mulai Bicara'}
                aria-label={isRecording ? 'Berhenti' : 'Mulai rekam'}
              >
                {isRecording ? <MicOff className="h-7 w-7" /> : <Mic className="h-7 w-7" />}
              </button>

              <div className="space-y-1">
                <div className="flex flex-wrap items-center gap-2">
                  <h2 className="text-lg font-black text-neutral-900 dark:text-white font-brand uppercase tracking-wide">
                    {isRecording ? 'Mendengarkan Suara Anda...' : 'Siap Merekam Suara'}
                  </h2>
                  <span
                    className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-bold font-mono tracking-wider ${
                      isRecording
                        ? 'bg-red-100 text-red-700 dark:bg-red-950 dark:text-red-300'
                        : 'bg-neutral-100 text-neutral-500 dark:bg-neutral-900 dark:text-neutral-400'
                    }`}
                  >
                    {isRecording && <span>LIVE ·</span>}
                    <span id="timer">
                      {formatTime(isRecording ? recorder.seconds : 0)} / {formatTime(recordingLimit)}
                    </span>
                  </span>
                </div>
                <p className="text-xs text-neutral-500 dark:text-neutral-400">
                  {isRecording
                    ? 'Bicaralah dengan jelas, lalu tekan tombol lagi untuk mengirim. Tekan Esc untuk membatalkan.'
                    : 'Centang persetujuan di bawah, lalu tekan tombol mikrofon merah atau gunakan pintasan [Alt + V].'}
                </p>
              </div>
            </div>

            {/* Middle: Canvas Waveform Visualizer */}
            <div className="hidden lg:flex items-center justify-center px-4">
              <AudioWaveform
                isRecording={isRecording}
                isPlaying={isPlayingAudio}
                stream={recorder.stream}
              />
            </div>

            {/* Right: Alternative Audio File Upload */}
            <div className="flex items-center gap-2">
              <button
                id="uploadBtn"
                type="button"
                onClick={() => setIsUploadOpen(true)}
                disabled={isRecording || isAnalyzing}
                className="inline-flex items-center gap-1.5 px-4 py-2 text-xs font-bold uppercase tracking-wider text-neutral-800 bg-neutral-100 hover:bg-neutral-200 rounded-lg dark:bg-neutral-900 dark:text-neutral-200 dark:hover:border-red-900/60 dark:hover:bg-neutral-800 border dark:border-neutral-800 transition-all cursor-pointer font-brand disabled:opacity-50 disabled:cursor-not-allowed"
                title="Unggah file rekaman audio"
              >
                <FileAudio className="h-4 w-4 text-red-600 dark:text-red-500" />
                <span>Unggah Audio</span>
              </button>
            </div>
          </div>

          <div className="mt-5 space-y-3 border-t border-neutral-100 pt-4 dark:border-red-950/60">
            <label className="flex items-start gap-2.5 text-xs leading-relaxed text-neutral-600 dark:text-neutral-300 cursor-pointer">
              <input
                ref={consentRef}
                id="consent"
                type="checkbox"
                checked={consent}
                onChange={(e) => setConsent(e.target.checked)}
                className="mt-0.5 h-4 w-4 shrink-0 accent-red-600 cursor-pointer"
              />
              <span id="consentText">{consentText}</span>
            </label>

            <details id="apiKeyBox" hidden={!apiKeyRequired} className="text-xs text-neutral-600 dark:text-neutral-300">
              <summary className="cursor-pointer">API key (hanya jika server memintanya)</summary>
              <input
                id="apiKey"
                type="password"
                autoComplete="off"
                placeholder="X-API-Key"
                onChange={(e) => {
                  apiKeyRef.current = e.target.value;
                }}
                className="mt-2 w-full max-w-sm rounded-lg border border-neutral-200 bg-white px-3 py-2 text-xs focus:border-red-500 focus:outline-none dark:border-neutral-800 dark:bg-neutral-900"
              />
            </details>

            {playbackUrl && <audio id="playback" controls src={playbackUrl} className="w-full" />}
          </div>

          {/* Speech error banner (Heuristic #9: Help users recognize & recover) */}
          {speechError && (
            <div
              id="micError"
              className="mt-4 flex items-start gap-3 p-3.5 rounded-xl bg-red-50 border border-red-200 text-red-900 dark:bg-red-950/60 dark:border-red-900 dark:text-red-200 text-xs"
            >
              <AlertCircle className="h-4 w-4 shrink-0 text-red-600 mt-0.5" />
              <div className="flex-1">
                <span className="font-bold uppercase tracking-wider font-brand">Perhatian Audio: </span>
                <span>{speechError}</span>
              </div>
              <button
                type="button"
                onClick={() => setSpeechError(null)}
                aria-label="Tutup pesan"
                className="text-red-700 hover:text-red-900 dark:text-red-400 text-xs font-bold cursor-pointer"
              >
                ✕
              </button>
            </div>
          )}
        </section>

        {/* Preset Sentence Inspiration Chips (Heuristic #6: Recognition Rather than Recall) */}
        <section className="flex flex-wrap items-center gap-2">
          <span className="text-xs font-bold uppercase tracking-wider text-neutral-500 dark:text-neutral-400 mr-1 font-brand">
            Contoh Kalimat:
          </span>
          {samplePresets.map((preset, idx) => (
            <button
              key={idx}
              type="button"
              onClick={() => handleInsertPreset(preset.text)}
              disabled={isAnalyzing}
              className="preset text-xs font-medium px-3.5 py-1.5 rounded-lg border border-neutral-200 bg-white text-neutral-800 hover:border-red-500 hover:text-red-600 dark:border-neutral-800 dark:bg-neutral-950 dark:text-neutral-300 dark:hover:border-red-600 dark:hover:text-red-400 transition-all cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {preset.label}
            </button>
          ))}
        </section>

        {/* 2-Column Core Grammar Workspace (Heuristic #4: Consistency & Standards) */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 items-stretch">
          {/* Column 1: Input / Spoken Words */}
          <div className="flex flex-col h-full rounded-xl border border-neutral-200 bg-white shadow-sm dark:border-red-950/70 dark:bg-neutral-950 transition-colors">
            {/* Header */}
            <div className="flex flex-wrap items-center justify-between gap-2 border-b border-neutral-100 px-4 py-3 dark:border-red-950/60">
              <div className="flex flex-wrap items-center gap-2">
                <span className="text-xs font-bold uppercase tracking-wider text-neutral-500 dark:text-neutral-400 font-brand">
                  KALIMAT ASLI (INPUT SUARA / KETIK)
                </span>
                <span className="text-xs text-neutral-400 dark:text-neutral-600">·</span>
                <span className="text-xs font-mono tabular-nums text-neutral-500 dark:text-neutral-400">
                  {inputStats.words} kata ({inputStats.chars}/{limits.maxTextChars} karakter)
                </span>
              </div>

              {inputText && (
                <button
                  id="clearBtn"
                  type="button"
                  onClick={handleClearText}
                  className="inline-flex items-center gap-1 text-xs text-neutral-400 hover:text-red-600 dark:hover:text-red-400 transition-colors cursor-pointer"
                  title="Bersihkan teks input"
                >
                  <Trash2 className="h-3.5 w-3.5" />
                  <span>Bersihkan</span>
                </button>
              )}
            </div>

            {/* Input Textarea */}
            <div className="flex-1 p-4 relative">
              <textarea
                id="textInput"
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                maxLength={limits.maxTextChars}
                aria-label="Kalimat bahasa Inggris yang akan diperiksa"
                placeholder={
                  'Bicaralah melalui mikrofon atau ketik kalimat bahasa Inggris di sini...\n' +
                  "Contoh ucapan yang sering salah: 'She don't like apples' atau 'They is playing in the park'."
                }
                className="w-full h-full min-h-[220px] resize-none border-0 bg-transparent text-base leading-relaxed text-neutral-800 placeholder-neutral-400 focus:outline-none dark:text-neutral-100 dark:placeholder-neutral-600 font-medium"
              />
            </div>

            {/* Action Bar */}
            <div className="flex items-center justify-between border-t border-neutral-100 px-4 py-3 bg-neutral-50/70 dark:border-red-950/60 dark:bg-neutral-950 rounded-b-xl">
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={toggleRecording}
                  disabled={isAnalyzing}
                  className={`inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-bold uppercase tracking-wider rounded-lg border transition-all cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed ${
                    isRecording
                      ? 'border-red-500 bg-red-600 text-white shadow-md shadow-red-600/30'
                      : 'border-neutral-200 bg-white text-neutral-800 hover:border-red-400 hover:text-red-600 dark:border-neutral-800 dark:bg-neutral-900 dark:text-neutral-200 dark:hover:border-red-900/80'
                  }`}
                >
                  <Mic className={`h-3.5 w-3.5 ${isRecording ? 'text-white' : 'text-red-600 dark:text-red-500'}`} />
                  <span>{isRecording ? 'Hentikan' : 'Bicara'}</span>
                </button>
              </div>

              <button
                id="checkBtn"
                type="button"
                onClick={() => analyzeText()}
                disabled={!inputText.trim() || isAnalyzing}
                className="inline-flex items-center gap-2 px-5 py-2 text-xs font-bold uppercase tracking-wider rounded-lg bg-red-600 text-white hover:bg-red-500 disabled:opacity-40 disabled:cursor-not-allowed shadow-md shadow-red-600/30 transition-all cursor-pointer font-brand"
              >
                {isAnalyzing ? (
                  <>
                    <Loader2 className="h-3.5 w-3.5 animate-spin" />
                    <span>Menganalisis...</span>
                  </>
                ) : (
                  <>
                    <Sparkles className="h-3.5 w-3.5" />
                    <span>Periksa Tata Bahasa</span>
                  </>
                )}
              </button>
            </div>
          </div>

          {/* Column 2: Correction Output & Diff Viewer */}
          <div
            ref={resultColumnRef}
            id="resultColumn"
            aria-busy={isAnalyzing}
            className="flex flex-col h-full scroll-mt-20"
          >
            {isAnalyzing ? (
              <div
                id="analyzing"
                className="flex-1 flex flex-col items-center justify-center p-8 rounded-xl border border-neutral-200 bg-white dark:border-red-950/70 dark:bg-neutral-950 text-center min-h-[300px]"
              >
                <div className="relative mb-4">
                  <div className="h-12 w-12 rounded-full border-4 border-neutral-200 border-t-red-600 animate-spin dark:border-neutral-800 dark:border-t-red-500" />
                  <Sparkles className="absolute inset-0 m-auto h-5 w-5 text-red-600 dark:text-red-500" />
                </div>
                <h3 className="text-base font-black text-neutral-900 dark:text-white font-brand uppercase tracking-wide">
                  Menganalisis Kalimat...
                </h3>
                <p className="text-xs text-neutral-500 dark:text-neutral-400 mt-1 max-w-sm">{progress}</p>
              </div>
            ) : analysisResult ? (
              <div id="result" className="h-full">
                <DiffViewer
                  tokens={diffTokens}
                  originalText={analysisResult.original}
                  correctedText={analysisResult.corrected}
                  changed={analysisResult.changed}
                  changesCount={analysisResult.edits.length}
                  canSpeak={canSpeak}
                  onSpeak={handleSpeak}
                  isPlaying={isPlayingAudio}
                  onRevert={handleRevert}
                />
              </div>
            ) : (
              <div className="flex-1 flex flex-col items-center justify-center p-8 rounded-xl border border-dashed border-neutral-300 bg-white/60 dark:border-neutral-800 dark:bg-neutral-950/40 text-center min-h-[300px]">
                <div className="h-12 w-12 rounded-full bg-neutral-100 dark:bg-neutral-900 flex items-center justify-center text-neutral-400 mb-3 border dark:border-neutral-800">
                  <Sparkles className="h-6 w-6 text-red-600/70" />
                </div>
                <h3 className="text-base font-black text-neutral-700 dark:text-neutral-300 font-brand uppercase tracking-wide">
                  Belum Ada Analisis Tata Bahasa
                </h3>
                <p className="text-xs text-neutral-500 dark:text-neutral-400 mt-1 max-w-xs leading-relaxed">
                  Bicaralah melalui mikrofon di atas atau pilih salah satu contoh kalimat untuk melihat perbandingan kata demi kata.
                </p>
              </div>
            )}
          </div>
        </div>

        {/* Error notification banner if API failed */}
        {errorMessage && (
          <div
            ref={errorRef}
            id="apiError"
            className="scroll-mt-20 flex items-center justify-between gap-3 p-4 rounded-xl bg-red-50 border border-red-200 text-red-900 dark:bg-red-950/60 dark:border-red-900 dark:text-red-200 text-xs"
          >
            <div className="flex items-center gap-2.5">
              <AlertCircle className="h-4 w-4 shrink-0 text-red-600" />
              <span>{errorMessage}</span>
            </div>
            <button
              id="retryBtn"
              type="button"
              onClick={() => retryRef.current?.()}
              className="shrink-0 px-3 py-1 font-bold rounded-md bg-red-600 text-white hover:bg-red-500 transition-colors cursor-pointer uppercase tracking-wider font-brand text-xs"
            >
              Coba Lagi
            </button>
          </div>
        )}

        {/* Detailed Explanation Breakdown */}
        {analysisResult && !isAnalyzing && (
          <section className="pt-2">
            <ExplanationDrawer result={analysisResult} />
          </section>
        )}

      </main>

      <p id="status" role="status" aria-live="polite" className="sr-only">
        {isRecording ? RECORDING_STATUS : progress}
      </p>

      {/* Floating Undo Toast (Heuristic #3: User Control and Freedom) */}
      {toastMessage && (
        <div
          id="toast"
          className="fixed bottom-6 left-6 right-6 sm:left-auto z-50 flex items-center gap-3 rounded-xl bg-neutral-900 px-4 py-3 text-xs text-white shadow-xl dark:bg-white dark:text-neutral-900"
        >
          <span>{toastMessage}</span>
          {canUndoClear && (
            <button
              id="undoBtn"
              type="button"
              onClick={handleUndoClear}
              className="font-bold underline text-red-400 dark:text-red-600 hover:opacity-80 cursor-pointer"
            >
              Batalkan (Undo)
            </button>
          )}
        </div>
      )}

      {/* Footer */}
      <footer className="border-t border-neutral-200/80 bg-white py-6 dark:border-red-950/80 dark:bg-black transition-colors">
        <div className="mx-auto flex max-w-7xl flex-col lg:flex-row items-center justify-between gap-4 px-4 sm:px-6 lg:px-8 text-xs text-neutral-500 dark:text-neutral-400 font-medium">
          <div className="text-center lg:text-left">
            <span className="font-brand uppercase font-bold text-neutral-700 dark:text-neutral-300">Voice Grammar.AI</span> · {PROJECT_TITLE}
          </div>
          <div className="flex flex-wrap items-center justify-center gap-x-4 gap-y-2">
            <button
              type="button"
              onClick={() => setIsHistoryOpen(true)}
              className="hover:text-red-600 dark:hover:text-red-400 transition-colors cursor-pointer"
            >
              Riwayat Koreksi
            </button>
            <button
              type="button"
              onClick={() => setIsShortcutsOpen(true)}
              className="hover:text-red-600 dark:hover:text-red-400 transition-colors cursor-pointer"
            >
              Pintasan Keyboard
            </button>
            <button
              type="button"
              onClick={() => setIsCheatsheetOpen(true)}
              className="hover:text-red-600 dark:hover:text-red-400 transition-colors cursor-pointer"
            >
              Panduan Tata Bahasa
            </button>
            <a href="/docs" className="hover:text-red-600 dark:hover:text-red-400 transition-colors">
              Dokumentasi API
            </a>
            <a href="/readyz" className="hover:text-red-600 dark:hover:text-red-400 transition-colors">
              Status Server
            </a>
            <a href={REPOSITORY_URL} className="hover:text-red-600 dark:hover:text-red-400 transition-colors">
              Kode Sumber
            </a>
          </div>
        </div>
      </footer>

      {/* Interactive Modals */}
      <HistoryModal
        isOpen={isHistoryOpen}
        onClose={() => setIsHistoryOpen(false)}
        history={history}
        onSelectRecord={handleRestoreRecord}
        onClearHistory={handleClearAllHistory}
        onDeleteRecord={handleDeleteRecord}
      />
      <CheatsheetModal
        isOpen={isCheatsheetOpen}
        onClose={() => setIsCheatsheetOpen(false)}
        onInsertExample={handleInsertPreset}
      />
      <ShortcutsModal
        isOpen={isShortcutsOpen}
        onClose={() => setIsShortcutsOpen(false)}
      />
      <AudioUploadModal
        isOpen={isUploadOpen}
        onClose={() => setIsUploadOpen(false)}
        onSubmit={(file) => analyzeAudio(file, file.name)}
        maxUploadMb={limits.maxUploadMb}
        consent={consent}
        onConsentChange={setConsent}
        consentText={consentText}
      />
    </div>
  );
}
