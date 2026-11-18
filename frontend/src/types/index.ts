/** A single change, with offsets into the original text (see EditOut in the API schemas). */
export interface GrammarEdit {
  op: string;
  original: string;
  corrected: string;
  start: number;
  end: number;
  error_type: string | null;
  description: string | null;
}

export interface Correction {
  original: string;
  corrected: string;
  changed: boolean;
  edits: GrammarEdit[];
  warnings: string[];
}

export interface TextResponse {
  correction: Correction;
  timings_ms: Record<string, number>;
}

export interface AudioResponse {
  transcript: { text: string };
  correction: Correction | null;
  warnings: string[];
  timings_ms: Record<string, number>;
}

export interface ServerInfo {
  corrector: string | null;
  stt: string | null;
  limits: {
    max_upload_mb: number;
    max_audio_seconds: number;
    max_text_chars: number;
  };
  ui: {
    demo_mode: boolean;
    api_key_required: boolean;
    load_models_on_demand?: boolean;
  };
}

/** What the page shows for one analysed sentence. */
export interface AnalysisResult {
  original: string;
  corrected: string;
  changed: boolean;
  edits: GrammarEdit[];
  notes: string;
  praise: boolean;
  totalMs: number | null;
}

export type InputSource = 'voice' | 'text';

export interface HistoryRecord {
  id: string;
  timestamp: number;
  source: InputSource;
  originalText: string;
  correctedText: string;
  changesCount: number;
}

export interface DiffToken {
  type: 'added' | 'removed' | 'unchanged';
  text: string;
}
