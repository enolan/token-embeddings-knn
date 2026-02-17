/** Make whitespace visible in token strings. */
export function visualizeToken(s: string): string {
  return s
    .replace(/ /g, "\u00B7") // middle dot for space
    .replace(/\n/g, "\u21B5") // return symbol for newline
    .replace(/\t/g, "\u21E5"); // rightwards arrow to bar for tab
}
