import React, { useState } from 'react';
import { DiffToken } from '../types';
import { copyText } from '../utils/browser';
import { Check, Copy, Volume2, RotateCcw } from 'lucide-react';

interface DiffViewerProps {
  tokens: DiffToken[];
  originalText: string;
  correctedText: string;
  changed: boolean;
  changesCount: number;
  canSpeak: boolean;
  onSpeak: (text: string) => void;
  isPlaying: boolean;
  onRevert: () => void;
}

export const DiffViewer: React.FC<DiffViewerProps> = ({
  tokens,
  originalText,
  correctedText,
  changed,
  changesCount,
  canSpeak,
  onSpeak,
  isPlaying,
  onRevert,
}) => {
  const [viewMode, setViewMode] = useState<'diff' | 'clean' | 'split'>('diff');
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    if (await copyText(correctedText)) {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  return (
    <div className="flex flex-col h-full rounded-xl border border-neutral-200 bg-white shadow-sm dark:border-red-950/70 dark:bg-neutral-950 dark:glow-red transition-all">
      {/* Header Bar */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-neutral-100 px-4 py-3 dark:border-red-950/60">
        <div className="flex items-center gap-2">
          <span className="text-xs font-bold uppercase tracking-wider text-neutral-500 dark:text-neutral-400 font-brand">
            HASIL KOREKSI TATA BAHASA
          </span>
          <span className="text-xs text-neutral-400 dark:text-neutral-600">·</span>
          <span className="text-xs font-bold text-red-600 dark:text-red-500 uppercase tracking-wide">
            {changesCount} perbaikan
          </span>
        </div>

        {/* View Mode Segmented Controls */}
        <div className="flex items-center gap-1 rounded-lg bg-neutral-100 p-0.5 dark:bg-neutral-900 border dark:border-neutral-800">
          <button
            id="viewDiff"
            type="button"
            onClick={() => setViewMode('diff')}
            className={`px-2.5 py-1 text-xs font-bold rounded-md transition-colors cursor-pointer ${
              viewMode === 'diff'
                ? 'bg-red-600 text-white shadow-sm'
                : 'text-neutral-600 hover:text-neutral-900 dark:text-neutral-400 dark:hover:text-white'
            }`}
          >
            Diff Warna
          </button>
          <button
            id="viewClean"
            type="button"
            onClick={() => setViewMode('clean')}
            className={`px-2.5 py-1 text-xs font-bold rounded-md transition-colors cursor-pointer ${
              viewMode === 'clean'
                ? 'bg-red-600 text-white shadow-sm'
                : 'text-neutral-600 hover:text-neutral-900 dark:text-neutral-400 dark:hover:text-white'
            }`}
          >
            Teks Bersih
          </button>
          <button
            id="viewSplit"
            type="button"
            onClick={() => setViewMode('split')}
            className={`px-2.5 py-1 text-xs font-bold rounded-md transition-colors cursor-pointer ${
              viewMode === 'split'
                ? 'bg-red-600 text-white shadow-sm'
                : 'text-neutral-600 hover:text-neutral-900 dark:text-neutral-400 dark:hover:text-white'
            }`}
          >
            Berdampingan
          </button>
        </div>
      </div>

      {/* Main Content Area */}
      <div className="flex-1 p-5 overflow-y-auto min-h-[220px]">
        {viewMode === 'diff' && (
          <div
            id="diffView"
            className="text-base leading-relaxed tracking-normal text-neutral-800 dark:text-neutral-200"
          >
            {tokens.map((token, idx) => {
              if (token.type === 'removed') {
                if (!token.text.trim()) return null;
                return (
                  <del
                    key={idx}
                    className="inline-block bg-neutral-200 text-neutral-500 line-through decoration-red-500 decoration-2 px-1 mx-0.5 rounded-xs dark:bg-neutral-900 dark:text-neutral-500"
                    title="Kata awal sebelum dikoreksi"
                  >
                    {token.text}
                  </del>
                );
              }
              if (token.type === 'added' && token.text.trim()) {
                return (
                  <ins
                    key={idx}
                    className="inline-block no-underline bg-emerald-600/10 text-emerald-700 font-bold px-1.5 mx-0.5 rounded-xs dark:bg-emerald-950/60 dark:text-emerald-300 border-b-2 border-emerald-600"
                    title="Perbaikan tata bahasa"
                  >
                    {token.text}
                  </ins>
                );
              }
              return <span key={idx}>{token.text}</span>;
            })}
          </div>
        )}

        {viewMode === 'clean' && (
          <div
            id="cleanView"
            className="text-base leading-relaxed text-neutral-800 dark:text-neutral-100 whitespace-pre-wrap select-text font-medium"
          >
            {correctedText || originalText}
          </div>
        )}

        {viewMode === 'split' && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 h-full">
            <div className="p-3.5 rounded-lg border border-neutral-200 bg-neutral-50 dark:border-neutral-800/80 dark:bg-black">
              <div className="text-xs font-bold uppercase tracking-wider text-neutral-500 dark:text-neutral-400 mb-2 font-brand">
                Asli (Teks Ucapan)
              </div>
              <p id="original" className="text-sm leading-relaxed text-neutral-700 dark:text-neutral-300">
                {originalText}
              </p>
            </div>
            <div className="p-3.5 rounded-lg border border-red-200 bg-red-50/30 dark:border-red-950 dark:bg-red-950/20">
              <div className="text-xs font-bold uppercase tracking-wider text-red-600 dark:text-red-400 mb-2 font-brand">
                Saran Perbaikan
              </div>
              <p id="corrected" className="text-sm leading-relaxed text-neutral-900 dark:text-white font-medium">
                {changed ? correctedText : 'Tidak ada saran perbaikan.'}
              </p>
            </div>
          </div>
        )}
      </div>

      {/* Action Footer Bar */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-t border-neutral-100 px-4 py-3 bg-neutral-50/70 dark:border-red-950/60 dark:bg-neutral-950 rounded-b-xl">
        <div className="flex items-center gap-2">
          {/* Read aloud pronunciation TTS */}
          {canSpeak && (
            <button
              id="speakBtn"
              type="button"
              onClick={() => onSpeak(correctedText)}
              disabled={!correctedText}
              className={`inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-bold uppercase tracking-wider rounded-lg border transition-all cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed ${
                isPlaying
                  ? 'border-red-500 bg-red-600 text-white shadow-md shadow-red-600/30'
                  : 'border-neutral-200 bg-white text-neutral-800 hover:border-red-300 dark:border-neutral-800 dark:bg-neutral-900 dark:text-neutral-200 dark:hover:border-red-900/80'
              }`}
              title="Dengarkan pengucapan kalimat hasil"
            >
              <Volume2 className={`h-3.5 w-3.5 ${isPlaying ? 'animate-pulse text-white' : 'text-red-600 dark:text-red-400'}`} />
              <span>{isPlaying ? 'Memutar Suara...' : 'Dengarkan Pelafalan'}</span>
            </button>
          )}

          {/* Revert to original */}
          <button
            id="revertBtn"
            type="button"
            onClick={onRevert}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold rounded-lg border border-neutral-200 bg-white text-neutral-700 hover:bg-neutral-100 dark:border-neutral-800 dark:bg-neutral-900 dark:text-neutral-300 dark:hover:bg-neutral-800 transition-colors cursor-pointer"
            title="Kembalikan teks ke versi sebelum dikoreksi"
          >
            <RotateCcw className="h-3.5 w-3.5" />
            <span className="hidden sm:inline">Kembalikan Asli</span>
          </button>
        </div>

        <div className="flex items-center gap-2">
          {/* Copy Button */}
          <button
            id="copyBtn"
            type="button"
            onClick={handleCopy}
            className="inline-flex items-center gap-1.5 px-4 py-2 text-xs font-bold uppercase tracking-wider rounded-lg bg-red-600 text-white hover:bg-red-500 shadow-md shadow-red-600/30 transition-all cursor-pointer"
          >
            {copied ? (
              <>
                <Check className="h-3.5 w-3.5" />
                <span>Tersalin!</span>
              </>
            ) : (
              <>
                <Copy className="h-3.5 w-3.5" />
                <span>Salin Teks</span>
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
};
