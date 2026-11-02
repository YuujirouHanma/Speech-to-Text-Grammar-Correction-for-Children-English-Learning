import React from 'react';
import { X, Keyboard } from 'lucide-react';

interface ShortcutsModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const ShortcutsModal: React.FC<ShortcutsModalProps> = ({ isOpen, onClose }) => {
  if (!isOpen) return null;

  const shortcuts = [
    { key: 'Alt + V', desc: 'Mulai / Hentikan Perekaman Suara (Toggle Record)' },
    { key: 'Ctrl + Enter / ⌘ + Enter', desc: 'Jalankan Pemeriksaan Tata Bahasa (Analyze)' },
    { key: 'Ctrl + Shift + C / ⌘ + ⇧ + C', desc: 'Salin Teks Hasil Koreksi ke Clipboard' },
    { key: 'Ctrl + Shift + P / ⌘ + ⇧ + P', desc: 'Putar / Hentikan Pelafalan Suara (TTS)' },
    { key: 'Esc', desc: 'Tutup Dialog / Batalkan Rekaman (Tidak Dikirim)' },
  ];

  return (
    <div role="dialog" aria-modal="true" aria-label="Pintasan keyboard" className="fixed inset-0 z-50 flex items-center justify-center p-4 sm:p-6 bg-black/80 backdrop-blur-xs">
      <div className="relative w-full max-w-lg flex flex-col rounded-2xl bg-white shadow-2xl border border-neutral-200 dark:bg-neutral-950 dark:border-red-950/80 transition-colors">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-neutral-100 px-6 py-4 dark:border-red-950/60">
          <div className="flex items-center gap-2.5">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-red-600 text-white shadow-sm shadow-red-600/30">
              <Keyboard className="h-4 w-4" />
            </div>
            <div>
              <h2 className="text-base font-black text-neutral-900 dark:text-white font-brand uppercase tracking-wide">
                Pintasan Keyboard (Shortcuts)
              </h2>
              <p className="text-xs text-neutral-500 dark:text-neutral-400">
                Akselerator navigasi cepat untuk produktivitas maksimal
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="rounded-lg p-1.5 text-neutral-400 hover:bg-neutral-100 hover:text-red-600 dark:hover:bg-neutral-900 dark:hover:text-red-400 transition-colors cursor-pointer"
            aria-label="Tutup pintasan"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* List */}
        <div className="p-6 space-y-3">
          {shortcuts.map((sc, idx) => (
            <div
              key={idx}
              className="flex items-center justify-between gap-4 p-3 rounded-xl border border-neutral-200 bg-neutral-50 dark:border-neutral-800/80 dark:bg-black/60"
            >
              <span className="text-xs text-neutral-700 dark:text-neutral-300 font-medium">
                {sc.desc}
              </span>
              <kbd className="px-2.5 py-1 text-xs font-mono font-bold text-neutral-800 bg-white border border-neutral-200 rounded-md shadow-xs dark:bg-neutral-900 dark:border-neutral-800 dark:text-red-400 whitespace-nowrap">
                {sc.key}
              </kbd>
            </div>
          ))}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-end border-t border-neutral-100 px-6 py-3.5 bg-neutral-50/70 dark:border-red-950/60 dark:bg-neutral-950 rounded-b-2xl">
          <button
            onClick={onClose}
            className="px-4 py-2 text-xs font-bold uppercase tracking-wider text-white bg-red-600 rounded-lg hover:bg-red-500 shadow-md shadow-red-600/30 transition-all cursor-pointer font-brand"
          >
            Mengerti
          </button>
        </div>
      </div>
    </div>
  );
};
