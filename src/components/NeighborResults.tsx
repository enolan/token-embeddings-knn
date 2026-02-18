import { useMemo, useState } from "react";
import type { TokenEntry } from "../types";
import { normalizeToken, visualizeToken } from "../util";

interface Props {
  tokenId: number;
  token: TokenEntry;
  getToken: (id: number) => TokenEntry | undefined;
  onNavigate: (tokenId: number) => void;
}

export function NeighborResults({
  tokenId,
  token,
  getToken,
  onNavigate,
}: Props) {
  const [hideDuplicates, setHideDuplicates] = useState(false);

  const filteredNeighbors = useMemo(() => {
    if (!hideDuplicates) return token.n;
    const seen = new Map<string, number>();
    // Seed with the selected token so neighbors that are near-duplicates of it get hidden too
    seen.set(normalizeToken(token.s), -1);
    const result: typeof token.n = [];
    for (const entry of token.n) {
      const neighbor = getToken(entry[0]);
      const key = neighbor ? normalizeToken(neighbor.s) : String(entry[0]);
      if (!seen.has(key)) {
        seen.set(key, result.length);
        result.push(entry);
      }
    }
    return result;
  }, [token.n, token.s, hideDuplicates, getToken]);

  const similarities = filteredNeighbors.map(([, s]) => s);
  const maxSim = Math.max(...similarities);
  const minSim = Math.min(...similarities);
  const range = maxSim - minSim || 1;

  return (
    <div className="neighbor-results">
      <div className="selected-token">
        <span className="label">Selected</span>
        <span className="token-text">{visualizeToken(token.s)}</span>
        <span className="token-id">#{tokenId}</span>
      </div>
      <label className="dedup-checkbox">
        <input
          type="checkbox"
          checked={hideDuplicates}
          onChange={(e) => setHideDuplicates(e.target.checked)}
        />
        Hide near-duplicates
      </label>
      <table className="neighbor-table">
        <thead>
          <tr>
            <th>Rank</th>
            <th>Token</th>
            <th>ID</th>
            <th>Similarity</th>
          </tr>
        </thead>
        <tbody>
          {filteredNeighbors.map(([neighborId, similarity], i) => {
            const neighbor = getToken(neighborId);
            const barWidth = 15 + ((similarity - minSim) / range) * 85;
            return (
              <tr
                key={neighborId}
                className="neighbor-row"
                onClick={() => onNavigate(neighborId)}
                style={{ animationDelay: `${i * 0.025}s` }}
              >
                <td className="rank">{i + 1}</td>
                <td className="token-text">
                  {neighbor ? visualizeToken(neighbor.s) : "?"}
                </td>
                <td className="token-id">#{neighborId}</td>
                <td className="similarity-cell">
                  <span
                    className="similarity-bar"
                    style={{ width: `${barWidth}%` }}
                  />
                  <span className="similarity-value">
                    {similarity.toFixed(4)}
                  </span>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
