import { useEffect, useRef, useState } from "react";
import { getLifter, searchLifters } from "../api";
import type { LifterDetail, LifterSummary } from "../api";

interface Props {
  onSelect: (lifter: LifterDetail) => void;
}

export default function LifterSearch({ onSelect }: Props) {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<LifterSummary[]>([]);
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const timer = useRef<ReturnType<typeof setTimeout>>(undefined);
  const containerRef = useRef<HTMLDivElement>(null);

  // Debounced search
  useEffect(() => {
    if (query.length < 2) {
      setResults([]);
      setOpen(false);
      return;
    }
    setLoading(true);
    clearTimeout(timer.current);
    timer.current = setTimeout(async () => {
      try {
        const data = await searchLifters(query);
        setResults(data);
        setOpen(true);
        setError("");
      } catch (e) {
        setError(e instanceof Error ? e.message : "Search failed");
        setResults([]);
      } finally {
        setLoading(false);
      }
    }, 300);
    return () => clearTimeout(timer.current);
  }, [query]);

  // Close dropdown on outside click
  useEffect(() => {
    function handleClick(e: MouseEvent) {
      if (
        containerRef.current &&
        !containerRef.current.contains(e.target as Node)
      ) {
        setOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, []);

  async function handleSelect(name: string) {
    setOpen(false);
    setQuery(name);
    setLoading(true);
    try {
      const detail = await getLifter(name);
      onSelect(detail);
      setError("");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load lifter");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="search-container" ref={containerRef}>
      <input
        type="text"
        placeholder="Search lifters by name (e.g. Ray Williams, John Haack)..."
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        onFocus={() => results.length > 0 && setOpen(true)}
      />
      {loading && <div className="loading">Searching...</div>}
      {error && <div className="error">{error}</div>}
      {open && results.length > 0 && (
        <div className="search-results">
          {results.map((r) => (
            <div
              key={r.name}
              className="search-item"
              onClick={() => handleSelect(r.name)}
            >
              <span className="search-item-name">{r.name}</span>
              <span className="search-item-meta">
                {r.competition_count} meets
                {r.best_total_kg ? ` · ${r.best_total_kg} kg total` : ""}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
