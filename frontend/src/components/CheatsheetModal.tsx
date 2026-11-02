import React, { useState } from 'react';
import { X, BookOpen } from 'lucide-react';

interface CheatsheetModalProps {
  isOpen: boolean;
  onClose: () => void;
  onInsertExample: (text: string) => void;
}

export const CheatsheetModal: React.FC<CheatsheetModalProps> = ({
  isOpen,
  onClose,
  onInsertExample,
}) => {
  const [activeTab, setActiveTab] = useState<'rules' | 'pitfalls'>('rules');

  if (!isOpen) return null;

  const grammarRules = [
    {
      title: 'Subject-Verb Agreement',
      rule: 'Subjek tunggal (he, she, it) memerlukan kata kerja tunggal (akhiran -s/-es). Subjek jamak (they, we) menggunakan bentuk dasar.',
      wrong: 'She work at multinational tech company.',
      right: 'She works at a multinational tech company.',
    },
    {
      title: 'Articles: A vs. An vs. The',
      rule: 'Gunakan "an" sebelum bunyi vokal (an hour, an apple), "a" sebelum bunyi konsonan (a university, a book), dan "the" untuk hal spesifik.',
      wrong: 'I have an unique opportunity today.',
      right: 'I have a unique opportunity today.',
    },
    {
      title: 'Prepositions: At, On, In',
      rule: 'AT untuk waktu/titik presisi (at 5 PM, at the door). ON untuk hari/tanggal/permukaan (on Monday, on the table). IN untuk bulan/tahun/ruang tertutup (in July, in Jakarta).',
      wrong: 'We will meet in Monday at July.',
      right: 'We will meet on Monday in July.',
    },
    {
      title: 'Past Tense Consistency',
      rule: 'Saat menceritakan peristiwa di masa lalu, jaga konsistensi bentuk lampau (Past Simple) di seluruh klausa.',
      wrong: 'Yesterday I went to office and finish my report.',
      right: 'Yesterday I went to the office and finished my report.',
    },
  ];

  const commonPitfalls = [
    {
      title: 'Discuss about → Discuss',
      note: 'Kata "discuss" sudah berarti "membahas tentang", tidak perlu ditambahkan "about".',
      wrong: 'We need to discuss about the voice grammar project.',
      right: 'We need to discuss the voice grammar project.',
    },
    {
      title: 'Join with us → Join us',
      note: 'Kata "join" langsung diikuti objek tanpa preposisi "with".',
      wrong: 'Please join with our team for lunch.',
      right: 'Please join our team for lunch.',
    },
    {
      title: 'Explain me → Explain to me',
      note: 'Kata "explain" membutuhkan preposisi "to" sebelum penerima penjelasan.',
      wrong: 'Can you explain me this grammar mistake?',
      right: 'Can you explain this grammar mistake to me?',
    },
    {
      title: 'Their vs. There vs. They\'re',
      note: 'Their (kepemilikan), There (tempat/keberadaan), They\'re (singkatan "they are").',
      wrong: 'There working on they\'re grammar skills over their.',
      right: 'They\'re working on their grammar skills over there.',
    },
  ];

  return (
    <div role="dialog" aria-modal="true" aria-label="Panduan cepat tata bahasa" className="fixed inset-0 z-50 flex items-center justify-center p-4 sm:p-6 bg-black/80 backdrop-blur-xs">
      <div className="relative w-full max-w-2xl max-h-[85vh] flex flex-col rounded-2xl bg-white shadow-2xl border border-neutral-200 dark:bg-neutral-950 dark:border-red-950/80 transition-colors">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-neutral-100 px-6 py-4 dark:border-red-950/60">
          <div className="flex items-center gap-2.5">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-red-600 text-white shadow-sm shadow-red-600/30">
              <BookOpen className="h-4 w-4" />
            </div>
            <div>
              <h2 className="text-base font-black text-neutral-900 dark:text-white font-brand uppercase tracking-wide">
                Panduan Cepat Tata Bahasa
              </h2>
              <p className="text-xs text-neutral-500 dark:text-neutral-400">
                Kaidah esensial bahasa Inggris dan kekeliruan umum saat berbicara
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="rounded-lg p-1.5 text-neutral-400 hover:bg-neutral-100 hover:text-red-600 dark:hover:bg-neutral-900 dark:hover:text-red-400 transition-colors cursor-pointer"
            aria-label="Tutup panduan"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* Tab switcher */}
        <div className="flex border-b border-neutral-100 dark:border-red-950/60 px-6 pt-3 gap-6 bg-neutral-50 dark:bg-black/40">
          <button
            onClick={() => setActiveTab('rules')}
            className={`pb-2.5 text-xs font-bold uppercase tracking-wider font-brand border-b-2 transition-colors cursor-pointer ${
              activeTab === 'rules'
                ? 'border-red-600 text-red-600 dark:border-red-500 dark:text-red-400'
                : 'border-transparent text-neutral-500 hover:text-neutral-800 dark:hover:text-neutral-300'
            }`}
          >
            Aturan Pokok Grammar
          </button>
          <button
            onClick={() => setActiveTab('pitfalls')}
            className={`pb-2.5 text-xs font-bold uppercase tracking-wider font-brand border-b-2 transition-colors cursor-pointer ${
              activeTab === 'pitfalls'
                ? 'border-red-600 text-red-600 dark:border-red-500 dark:text-red-400'
                : 'border-transparent text-neutral-500 hover:text-neutral-800 dark:hover:text-neutral-300'
            }`}
          >
            Kekeliruan Umum (Common Pitfalls)
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6 space-y-4">
          {activeTab === 'rules' ? (
            <div className="space-y-3.5">
              {grammarRules.map((r, i) => (
                <div
                  key={i}
                  className="p-4 rounded-xl border border-neutral-200 bg-neutral-50 dark:border-neutral-800/80 dark:bg-black/60"
                >
                  <h3 className="text-sm font-black text-neutral-900 dark:text-white mb-1 font-brand uppercase tracking-wide">
                    {r.title}
                  </h3>
                  <p className="text-xs text-neutral-600 dark:text-neutral-300 mb-2.5 leading-relaxed">
                    {r.rule}
                  </p>
                  <div className="space-y-1 text-xs font-mono">
                    <div className="text-neutral-400 line-through">
                      ❌ {r.wrong}
                    </div>
                    <div className="text-red-600 dark:text-red-400 font-bold flex items-center justify-between">
                      <span>✓ {r.right}</span>
                      <button
                        onClick={() => {
                          onInsertExample(r.wrong);
                          onClose();
                        }}
                        className="text-[11px] font-sans font-bold text-red-600 hover:underline dark:text-red-400 cursor-pointer uppercase font-brand tracking-wider"
                      >
                        Uji Kalimat Ini →
                      </button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="space-y-3.5">
              {commonPitfalls.map((p, i) => (
                <div
                  key={i}
                  className="p-4 rounded-xl border border-neutral-200 bg-neutral-50 dark:border-neutral-800/80 dark:bg-black/60"
                >
                  <h3 className="text-sm font-black text-neutral-900 dark:text-white mb-1 font-brand uppercase tracking-wide">
                    {p.title}
                  </h3>
                  <p className="text-xs text-neutral-600 dark:text-neutral-300 mb-2.5 leading-relaxed">
                    {p.note}
                  </p>
                  <div className="space-y-1 text-xs font-mono">
                    <div className="text-neutral-400 line-through">
                      ❌ {p.wrong}
                    </div>
                    <div className="text-red-600 dark:text-red-400 font-bold flex items-center justify-between">
                      <span>✓ {p.right}</span>
                      <button
                        onClick={() => {
                          onInsertExample(p.wrong);
                          onClose();
                        }}
                        className="text-[11px] font-sans font-bold text-red-600 hover:underline dark:text-red-400 cursor-pointer uppercase font-brand tracking-wider"
                      >
                        Uji Kalimat Ini →
                      </button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-end border-t border-neutral-100 px-6 py-3.5 bg-neutral-50/70 dark:border-red-950/60 dark:bg-neutral-950 rounded-b-2xl">
          <button
            onClick={onClose}
            className="px-4 py-2 text-xs font-bold uppercase tracking-wider text-white bg-red-600 rounded-lg hover:bg-red-500 shadow-md shadow-red-600/30 transition-all cursor-pointer font-brand"
          >
            Tutup
          </button>
        </div>
      </div>
    </div>
  );
};
