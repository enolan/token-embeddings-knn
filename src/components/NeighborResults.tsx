import type { TokenEntry } from "../types";
import { visualizeToken } from "../util";

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
  const similarities = token.n.map(([, s]) => s);
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
          {token.n.map(([neighborId, similarity], i) => {
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
