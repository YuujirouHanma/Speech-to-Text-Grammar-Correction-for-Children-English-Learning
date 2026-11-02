import React from 'react';
import { AnalysisResult } from '../types';
import { CheckCircle2, Info, BookCheck, ArrowRight } from 'lucide-react';

interface ExplanationDrawerProps {
  result: AnalysisResult;
}

/** A short label for an ERRANT error type such as R:VERB:SVA. */
function categoryOf(errorType: string | null): string {
  if (errorType?.endsWith('SPELL')) return 'Ejaan';
  if (errorType?.endsWith('PUNCT')) return 'Tanda Baca';
  if (errorType?.endsWith('ORTH')) return 'Penulisan';
  if (errorType?.endsWith('OTHER')) return 'Pilihan Kata';
  return 'Grammar';
}

export const ExplanationDrawer: React.FC<ExplanationDrawerProps> = ({ result }) => {
  const { changed, edits, notes, praise, totalMs } = result;

  const summary = changed
    ? `Model memberi ${edits.length} saran perbaikan untuk kalimat ini.`
    : praise
      ? 'Model tidak menemukan kesalahan pada kalimat ini.'
      : 'Model belum memberi saran untuk kalimat ini.';

  return (
    <div className="space-y-6">
      {/* 1. Summary */}
      <div className="rounded-xl border border-neutral-200 bg-white p-5 shadow-sm dark:border-red-950/70 dark:bg-neutral-950 transition-colors">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-neutral-100 dark:border-red-950/60">
          <div>
            <span className="text-xs font-bold uppercase tracking-wider text-neutral-500 dark:text-neutral-400 font-brand">
              HASIL DARI MODEL
            </span>
            <h3 className="text-xl font-black text-neutral-900 dark:text-white mt-0.5 font-brand uppercase tracking-wide">
              Ringkasan Koreksi
            </h3>
          </div>

          <div className="flex items-center gap-3">
            <div className="text-right text-xs text-neutral-500 dark:text-neutral-400 font-brand uppercase">
              Jumlah Saran
            </div>
            <div className="flex h-14 w-14 items-center justify-center rounded-xl border-2 border-red-500 font-brand text-2xl font-black tabular-nums shadow-sm text-red-500 dark:text-red-400">
              {edits.length}
            </div>
          </div>
        </div>

        <p className="mt-4 text-sm leading-relaxed text-neutral-700 dark:text-neutral-300">
          {summary}
        </p>
        {notes && (
          <p id="warnings" className="mt-2 flex items-start gap-2 text-sm leading-relaxed text-amber-800 dark:text-amber-300">
            <Info className="h-4 w-4 shrink-0 mt-0.5" />
            <span>{notes}</span>
          </p>
        )}
        <p className="mt-3 text-xs text-neutral-500 dark:text-neutral-400">
          Saran dibuat otomatis dan bisa keliru. Anggap sebagai bantuan, bukan nilai.
          {totalMs !== null && <span id="timing"> Diproses dalam {Math.round(totalMs)} ms.</span>}
        </p>
      </div>

      {/* 2. Detailed Rule Breakdown */}
      <div className="rounded-xl border border-neutral-200 bg-white p-5 shadow-sm dark:border-red-950/70 dark:bg-neutral-950 transition-colors">
        <div className="flex items-center justify-between gap-3 mb-4">
          <div className="flex items-center gap-2">
            <BookCheck className="h-4 w-4 text-red-600 dark:text-red-500" />
            <h4 className="text-base font-black text-neutral-900 dark:text-white font-brand uppercase tracking-wide">
              Rincian Koreksi ({edits.length})
            </h4>
          </div>
        </div>

        {edits.length === 0 ? (
          praise ? (
            <div
              id="praise"
              className="flex items-center gap-3 p-4 rounded-lg bg-emerald-50 border border-emerald-200 text-emerald-800 dark:bg-emerald-950/40 dark:border-emerald-900 dark:text-emerald-300"
            >
              <CheckCircle2 className="h-5 w-5 shrink-0" />
              <p className="text-sm">Bagus! Tidak ada kesalahan yang terdeteksi.</p>
            </div>
          ) : (
            <p className="text-sm text-neutral-500 dark:text-neutral-400">
              Tidak ada saran yang bisa ditampilkan.
            </p>
          )
        ) : (
          <ul id="edits" className="space-y-3">
            {edits.map((edit, index) => (
              <li
                key={index}
                className="p-3.5 rounded-lg border border-neutral-200 bg-neutral-50 dark:border-red-950/60 dark:bg-black/60 space-y-2"
              >
                <div className="flex flex-wrap items-center gap-2">
                  <span className="text-xs font-bold uppercase tracking-wider px-2 py-0.5 rounded-sm bg-red-600/10 text-red-600 dark:bg-red-950/60 dark:text-red-400 border dark:border-red-900/50 font-brand">
                    {categoryOf(edit.error_type)}
                  </span>
                  <div className="flex flex-wrap items-center gap-1.5 text-sm font-mono">
                    {!edit.original && <span className="text-neutral-500 dark:text-neutral-400">Tambahkan</span>}
                    {!edit.corrected && <span className="text-neutral-500 dark:text-neutral-400">Hapus</span>}
                    {edit.original && (
                      <span className="line-through text-neutral-400 dark:text-neutral-500">{edit.original}</span>
                    )}
                    {edit.original && edit.corrected && (
                      <ArrowRight className="h-3 w-3 text-red-600 dark:text-red-500" />
                    )}
                    {edit.corrected && (
                      <span className="font-bold text-emerald-700 dark:text-emerald-400">{edit.corrected}</span>
                    )}
                  </div>
                </div>

                {edit.description && (
                  <p className="text-xs leading-relaxed text-neutral-600 dark:text-neutral-300 pl-0.5">
                    {edit.description}
                  </p>
                )}
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
};
