import type { DiffToken } from '../types';

/**
 * Tokenize a sentence into words and whitespace/punctuation preserving chunks.
 */
function tokenize(text: string): string[] {
  // Matches words or punctuation or whitespace groups
  const matches = text.match(/[\w']+|[^\w\s]+|\s+/g);
  return matches || [];
}

/**
 * Compute word-level LCS diff between original and corrected text.
 * The comparison is exact, so that a fix of capitalization shows as a change too.
 */
export function computeWordDiff(original: string, corrected: string): DiffToken[] {
  const origTokens = tokenize(original);
  const corrTokens = tokenize(corrected);

  const n = origTokens.length;
  const m = corrTokens.length;

  if (n === 0 && m === 0) return [];
  if (n === 0) {
    return corrTokens.map((t) => ({ type: 'added', text: t }));
  }
  if (m === 0) {
    return origTokens.map((t) => ({ type: 'removed', text: t }));
  }

  // Build DP table for LCS
  const dp: number[][] = Array.from({ length: n + 1 }, () => Array(m + 1).fill(0));

  for (let i = 1; i <= n; i++) {
    for (let j = 1; j <= m; j++) {
      if (origTokens[i - 1] === corrTokens[j - 1]) {
        dp[i][j] = dp[i - 1][j - 1] + 1;
      } else {
        dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
      }
    }
  }

  // Backtrack to build diff tokens
  const diffs: DiffToken[] = [];
  let i = n;
  let j = m;

  while (i > 0 || j > 0) {
    if (i > 0 && j > 0 && origTokens[i - 1] === corrTokens[j - 1]) {
      diffs.unshift({ type: 'unchanged', text: corrTokens[j - 1] });
      i--;
      j--;
    } else if (j > 0 && (i === 0 || dp[i][j - 1] >= dp[i - 1][j])) {
      diffs.unshift({ type: 'added', text: corrTokens[j - 1] });
      j--;
    } else {
      diffs.unshift({ type: 'removed', text: origTokens[i - 1] });
      i--;
    }
  }

  return diffs;
}

/**
 * Clean up extra punctuation or normalize spaces
 */
export function formatWordCount(text: string): { words: number; chars: number } {
  const trimmed = text.trim();
  const words = trimmed ? trimmed.split(/\s+/).filter(Boolean).length : 0;
  const chars = text.length;
  return { words, chars };
}
