/** Make whitespace visible in token strings. */
export function visualizeToken(s: string): string {
  return s
    .replace(/ /g, "\u00B7") // middle dot for space
    .replace(/\n/g, "\u21B5") // return symbol for newline
    .replace(/\t/g, "\u21E5"); // rightwards arrow to bar for tab
}

/** Normalize a token string for near-duplicate detection.
 *  Strips leading whitespace (including SentencePiece ▁ U+2581) and lowercases. */
export function normalizeToken(s: string): string {
  return s.replace(/^[\s\u2581]+/, "").toLowerCase();
}
