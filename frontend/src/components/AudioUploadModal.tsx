import React, { useState, useRef } from 'react';
import { X, UploadCloud, FileAudio, AlertCircle } from 'lucide-react';

interface AudioUploadModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSubmit: (file: File) => void;
  maxUploadMb: number;
  consent: boolean;
  onConsentChange: (consent: boolean) => void;
  consentText: string;
}

export const AudioUploadModal: React.FC<AudioUploadModalProps> = ({
  isOpen,
  onClose,
  onSubmit,
  maxUploadMb,
  consent,
  onConsentChange,
  consentText,
}) => {
  const [file, setFile] = useState<File | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  if (!isOpen) return null;

  const pickFile = (selected: File | undefined) => {
    if (!selected) return;
    if (selected.size > maxUploadMb * 1024 * 1024) {
      setFile(null);
      setErrorMsg(`File terlalu besar (maksimal ${maxUploadMb} MB).`);
      return;
    }
    setFile(selected);
    setErrorMsg(null);
  };

  const handleSubmit = () => {
    if (!file || !consent) return;
    onSubmit(file);
    setFile(null);
    onClose();
  };

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label="Unggah file rekaman suara"
      className="fixed inset-0 z-50 flex items-center justify-center p-4 sm:p-6 bg-black/80 backdrop-blur-xs"
    >
      <div className="relative w-full max-w-md flex flex-col rounded-2xl bg-white shadow-2xl border border-neutral-200 dark:bg-neutral-950 dark:border-red-950/80 transition-colors">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-neutral-100 px-6 py-4 dark:border-red-950/60">
          <div className="flex items-center gap-2.5">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-red-600 text-white shadow-sm shadow-red-600/30">
              <UploadCloud className="h-4 w-4" />
            </div>
            <div>
              <h2 className="text-base font-black text-neutral-900 dark:text-white font-brand uppercase tracking-wide">
                Unggah File Rekaman Suara
              </h2>
              <p className="text-xs text-neutral-500 dark:text-neutral-400">
                Format: MP3, WAV, M4A, WebM, OGG, FLAC (Maks. {maxUploadMb} MB)
              </p>
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="rounded-lg p-1.5 text-neutral-400 hover:bg-neutral-100 hover:text-red-600 dark:hover:bg-neutral-900 dark:hover:text-red-400 transition-colors cursor-pointer"
            aria-label="Tutup upload"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* Upload Zone */}
        <div className="p-6 space-y-4">
          <input
            id="fileInput"
            type="file"
            ref={fileInputRef}
            onChange={(e) => {
              pickFile(e.target.files?.[0]);
              e.target.value = '';
            }}
            accept="audio/*"
            className="hidden"
          />

          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            onDragOver={(e) => e.preventDefault()}
            onDrop={(e) => {
              e.preventDefault();
              pickFile(e.dataTransfer.files[0]);
            }}
            className="w-full flex flex-col items-center justify-center p-8 border-2 border-dashed border-neutral-300 rounded-xl hover:border-red-500 hover:bg-red-50/10 dark:border-neutral-800 dark:hover:border-red-600 dark:hover:bg-red-950/20 transition-all cursor-pointer text-center"
          >
            <FileAudio className="h-10 w-10 text-red-600 mb-2" />
            <span id="fileName" className="text-xs font-bold text-neutral-800 dark:text-neutral-200 uppercase font-brand tracking-wider break-all">
              {file ? file.name : 'Klik untuk memilih file audio'}
            </span>
            <span className="text-[11px] text-neutral-400 dark:text-neutral-500 mt-1">
              {file ? `${(file.size / (1024 * 1024)).toFixed(2)} MB` : 'atau seret file rekaman ke sini'}
            </span>
          </button>

          <label className="flex items-start gap-2.5 text-xs leading-relaxed text-neutral-600 dark:text-neutral-300 cursor-pointer">
            <input
              id="consentUpload"
              type="checkbox"
              checked={consent}
              onChange={(e) => onConsentChange(e.target.checked)}
              className="mt-0.5 h-4 w-4 shrink-0 accent-red-600 cursor-pointer"
            />
            <span>{consentText}</span>
          </label>

          {errorMsg && (
            <div
              id="uploadError"
              className="flex items-center gap-2 p-3 rounded-lg bg-red-50 text-red-700 dark:bg-red-950/60 dark:text-red-300 text-xs border border-red-200 dark:border-red-900"
            >
              <AlertCircle className="h-4 w-4 shrink-0" />
              <span>{errorMsg}</span>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-end gap-2 border-t border-neutral-100 px-6 py-3.5 bg-neutral-50/70 dark:border-red-950/60 dark:bg-neutral-950 rounded-b-2xl">
          <button
            id="uploadCancel"
            type="button"
            onClick={onClose}
            className="px-4 py-2 text-xs font-semibold text-neutral-600 hover:text-neutral-900 dark:text-neutral-400 dark:hover:text-white cursor-pointer"
          >
            Batal
          </button>
          <button
            id="uploadSubmit"
            type="button"
            onClick={handleSubmit}
            disabled={!file || !consent}
            className="inline-flex items-center gap-1.5 px-4 py-2 text-xs font-bold uppercase tracking-wider text-white bg-red-600 rounded-lg hover:bg-red-500 disabled:opacity-50 disabled:cursor-not-allowed shadow-md shadow-red-600/30 transition-all cursor-pointer font-brand"
          >
            <span>Periksa Rekaman</span>
          </button>
        </div>
      </div>
    </div>
  );
};
