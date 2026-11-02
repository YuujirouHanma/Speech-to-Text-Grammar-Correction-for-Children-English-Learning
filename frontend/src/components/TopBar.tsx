import React from 'react';
import { Sun, Moon, Sparkles } from 'lucide-react';

interface TopBarProps {
  darkMode: boolean;
  onToggleTheme: () => void;
  onOpenHistory: () => void;
  onOpenCheatsheet: () => void;
  onOpenShortcuts: () => void;
}

export const TopBar: React.FC<TopBarProps> = ({
  darkMode,
  onToggleTheme,
  onOpenHistory,
  onOpenCheatsheet,
  onOpenShortcuts,
}) => {
  return (
    <header className="sticky top-0 z-40 w-full border-b border-neutral-200/80 bg-white/90 backdrop-blur-md dark:border-red-950/80 dark:bg-black/90 transition-colors">
      <div className="mx-auto flex h-16 max-w-7xl items-center justify-between gap-3 md:gap-8 px-4 sm:px-6 lg:px-8">
        {/* Zone 1: Brand Wordmark (Single text element) */}
        <div className="flex items-center gap-2.5 shrink-0">
          <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-red-600 text-white shadow-md shadow-red-600/40">
            <Sparkles className="h-4 w-4" />
          </div>
          <span className="font-brand text-2xl font-black tracking-wider uppercase text-neutral-900 dark:text-white whitespace-nowrap shrink-0">
            Voice Grammar<span className="text-red-600 dark:text-red-500">.AI</span>
          </span>
        </div>

        {/* Zone 2: 4-5 Clean Nav Links */}
        <nav className="hidden md:flex items-center gap-7 text-xs font-bold tracking-wider uppercase text-neutral-600 dark:text-neutral-300">
          <button
            onClick={onOpenHistory}
            className="hover:text-red-600 dark:hover:text-red-400 transition-colors whitespace-nowrap shrink-0 cursor-pointer text-left"
          >
            Riwayat Koreksi
          </button>
          <button
            onClick={onOpenCheatsheet}
            className="hover:text-red-600 dark:hover:text-red-400 transition-colors whitespace-nowrap shrink-0 cursor-pointer text-left"
          >
            Panduan Grammar
          </button>
          <button
            onClick={onOpenShortcuts}
            className="hover:text-red-600 dark:hover:text-red-400 transition-colors whitespace-nowrap shrink-0 cursor-pointer text-left"
          >
            Pintasan Keyboard
          </button>
        </nav>

        {/* Zone 3: Theme Switch */}
        <div className="flex items-center gap-3 shrink-0">
          <button
            onClick={onToggleTheme}
            className="flex h-9 w-9 items-center justify-center rounded-lg border border-neutral-200 text-neutral-700 hover:bg-neutral-100 hover:text-red-600 dark:border-neutral-800 dark:text-neutral-300 dark:hover:border-red-900/60 dark:hover:bg-neutral-900 dark:hover:text-red-400 transition-colors cursor-pointer"
            title={darkMode ? 'Beralih ke Mode Terang' : 'Beralih ke Mode Gelap'}
            aria-label="Toggle theme"
          >
            {darkMode ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
          </button>
        </div>
      </div>
    </header>
  );
};
