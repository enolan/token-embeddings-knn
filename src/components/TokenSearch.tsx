import { useState, useRef, useEffect } from "react";
import type { SearchResult } from "../types";
import { visualizeToken } from "../util";

interface Props {
  onSelect: (tokenId: number) => void;
  search: (query: string, limit?: number) => SearchResult[];
}

export function TokenSearch({ onSelect, search }: Props) {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [open, setOpen] = useState(false);
  const [activeIndex, setActiveIndex] = useState(-1);
  const inputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLUListElement>(null);

  useEffect(() => {
    if (query.length === 0) {
      setResults([]);
      setOpen(false);
      return;
    }
    const r = search(query);
    setResults(r);
    setOpen(r.length > 0);
    setActiveIndex(-1);
  }, [query, search]);

  function select(id: number) {
    setOpen(false);
    setQuery("");
    onSelect(id);
  }

  function handleKeyDown(e: React.KeyboardEvent) {
    if (!open) return;
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setActiveIndex((i) => Math.min(i + 1, results.length - 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setActiveIndex((i) => Math.max(i - 1, 0));
    } else if (e.key === "Enter" && activeIndex >= 0) {
      e.preventDefault();
      select(results[activeIndex].id);
    } else if (e.key === "Escape") {
      setOpen(false);
    }
  }

  // Scroll active item into view
  useEffect(() => {
    if (activeIndex >= 0 && listRef.current) {
      const item = listRef.current.children[activeIndex] as HTMLElement;
      item?.scrollIntoView({ block: "nearest" });
    }
  }, [activeIndex]);

  return (
    <div className="token-search">
      <input
        ref={inputRef}
        type="text"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        onKeyDown={handleKeyDown}
        onFocus={() => results.length > 0 && setOpen(true)}
        onBlur={() => setTimeout(() => setOpen(false), 150)}
        placeholder="Search tokens by text or ID..."
        className="search-input"
        autoFocus
      />
      {open && (
        <ul ref={listRef} className="search-results">
          {results.map((r, i) => (
            <li
              key={r.id}
              className={`search-result ${i === activeIndex ? "active" : ""}`}
              onMouseDown={() => select(r.id)}
            >
              <span className="token-text">{visualizeToken(r.text)}</span>
              <span className="token-id">#{r.id}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
