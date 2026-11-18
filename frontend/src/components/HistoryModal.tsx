import React, { useState } from 'react';
import { HistoryRecord } from '../types';
import { X, Search, Clock, Trash2, RotateCcw } from 'lucide-react';

interface HistoryModalProps {
  isOpen: boolean;
  onClose: () => void;
  history: HistoryRecord[];
  onSelectRecord: (record: HistoryRecord) => void;
  onClearHistory: () => void;
  onDeleteRecord: (id: string) => void;
}

export const HistoryModal: React.FC<HistoryModalProps> = ({
  isOpen,
  onClose,
  history,
  onSelectRecord,
  onClearHistory,
  onDeleteRecord,
}) => {
  const [searchQuery, setSearchQuery] = useState('');

  if (!isOpen) return null;

  const filtered = history.filter(
    (item) =>
      item.originalText.toLowerCase().includes(searchQuery.toLowerCase()) ||
      item.correctedText.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <div role="dialog" aria-modal="true" aria-label="Riwayat koreksi" className="fixed inset-0 z-50 flex items-center justify-center p-4 sm:p-6 bg-black/80 backdrop-blur-xs">
      <div className="relative w-full max-w-2xl max-h-[85vh] flex flex-col rounded-2xl bg-white shadow-2xl border border-neutral-200 dark:bg-neutral-950 dark:border-red-950/80 transition-colors">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-neutral-100 px-6 py-4 dark:border-red-950/60">
          <div className="flex items-center gap-2.5">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-red-600 text-white shadow-sm shadow-red-600/30">
              <Clock className="h-4 w-4" />
            </div>
            <div>
              <h2 className="text-base font-black text-neutral-900 dark:text-white font-brand uppercase tracking-wide">
                Riwayat Koreksi
              </h2>
              <p className="text-xs text-neutral-500 dark:text-neutral-400">
                Tersimpan hanya di browser ini, tanpa audio. Pilih kalimat untuk diperiksa lagi.
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="rounded-lg p-1.5 text-neutral-400 hover:bg-neutral-100 hover:text-red-600 dark:hover:bg-neutral-900 dark:hover:text-red-400 transition-colors cursor-pointer"
            aria-label="Tutup riwayat"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* Search Bar */}
        <div className="px-6 py-3 border-b border-neutral-100 dark:border-red-950/60 bg-neutral-50 dark:bg-black/60">
          <div className="relative">
            <Search className="absolute left-3 top-2.5 h-4 w-4 text-neutral-400" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Cari dalam riwayat kalimat..."
              className="w-full rounded-lg border border-neutral-200 bg-white py-1.5 pl-9 pr-4 text-xs text-neutral-900 placeholder-neutral-400 focus:border-red-500 focus:outline-none dark:border-neutral-800 dark:bg-neutral-900 dark:text-white"
            />
          </div>
        </div>

        {/* Records List */}
        <div className="flex-1 overflow-y-auto p-6 space-y-3">
          {history.length === 0 ? (
            <div className="text-center py-12">
              <Clock className="mx-auto h-8 w-8 text-neutral-400 dark:text-neutral-600 mb-2" />
              <p className="text-sm font-bold text-neutral-600 dark:text-neutral-400 font-brand uppercase tracking-wide">
                Belum ada riwayat koreksi
              </p>
              <p className="text-xs text-neutral-400 dark:text-neutral-500 mt-1">
                Lakukan pemeriksaan suara atau teks untuk menyimpan histori secara otomatis.
              </p>
            </div>
          ) : filtered.length === 0 ? (
            <div className="text-center py-8 text-xs text-neutral-500">
              Tidak ada hasil yang cocok dengan pencarian "{searchQuery}"
            </div>
          ) : (
            filtered.map((item) => (
              <div
                key={item.id}
                className="group relative p-3.5 rounded-xl border border-neutral-200 bg-white hover:border-red-500 hover:shadow-sm dark:border-neutral-800 dark:bg-black/60 dark:hover:border-red-900/80 transition-all"
              >
                <div className="flex items-center justify-between gap-2 mb-2">
                  <div className="flex items-center gap-2 text-xs text-neutral-400 dark:text-neutral-500 font-medium">
                    <span>{new Date(item.timestamp).toLocaleString('id-ID', { dateStyle: 'medium', timeStyle: 'short' })}</span>
                    <span>·</span>
                    <span>{item.source === 'voice' ? 'Suara' : 'Ketik'}</span>
                    <span>·</span>
                    <span className="font-bold text-red-600 dark:text-red-400 font-brand">
                      {item.changesCount} PERBAIKAN
                    </span>
                  </div>

                  <div className="flex items-center gap-1">
                    <button
                      onClick={() => {
                        onSelectRecord(item);
                        onClose();
                      }}
                      className="inline-flex items-center gap-1 px-3 py-1 text-xs font-bold uppercase tracking-wider rounded-md bg-red-600/10 text-red-600 hover:bg-red-600 hover:text-white dark:bg-red-950/60 dark:text-red-400 dark:hover:bg-red-600 dark:hover:text-white transition-all cursor-pointer font-brand"
                    >
                      <RotateCcw className="h-3 w-3" />
                      <span>Buka</span>
                    </button>
                    <button
                      onClick={() => onDeleteRecord(item.id)}
                      className="p-1 text-neutral-400 hover:text-red-600 dark:hover:text-red-400 transition-colors cursor-pointer"
                      title="Hapus dari riwayat"
                    >
                      <Trash2 className="h-3.5 w-3.5" />
                    </button>
                  </div>
                </div>

                <div className="space-y-1">
                  <div className="text-xs text-neutral-500 dark:text-neutral-400 line-clamp-2">
                    <span className="font-bold text-neutral-700 dark:text-neutral-300 font-brand uppercase">Asli: </span>
                    {item.originalText}
                  </div>
                  <div className="text-xs text-neutral-800 dark:text-neutral-200 font-medium line-clamp-2">
                    <span className="font-bold text-red-600 dark:text-red-400 font-brand uppercase">Koreksi: </span>
                    {item.correctedText}
                  </div>
                </div>
              </div>
            ))
          )}
        </div>

        {/* Footer */}
        {history.length > 0 && (
          <div className="flex items-center justify-between border-t border-neutral-100 px-6 py-3.5 bg-neutral-50/70 dark:border-red-950/60 dark:bg-neutral-950 rounded-b-2xl">
            <button
              onClick={onClearHistory}
              className="text-xs text-red-600 hover:text-red-700 dark:text-red-400 font-bold uppercase tracking-wider font-brand cursor-pointer"
            >
              Hapus Semua Riwayat
            </button>
            <button
              onClick={onClose}
              className="px-4 py-1.5 text-xs font-semibold text-neutral-700 bg-white border border-neutral-200 rounded-lg hover:bg-neutral-50 dark:border-neutral-800 dark:bg-neutral-900 dark:text-neutral-200 cursor-pointer"
            >
              Tutup
            </button>
          </div>
        )}
      </div>
    </div>
  );
};
