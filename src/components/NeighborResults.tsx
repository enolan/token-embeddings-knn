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
  return (
    <div className="neighbor-results">
      <div className="selected-token">
        <span className="label">Selected token:</span>
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
            return (
              <tr
                key={neighborId}
                className="neighbor-row"
                onClick={() => onNavigate(neighborId)}
              >
                <td className="rank">{i + 1}</td>
                <td className="token-text">
                  {neighbor ? visualizeToken(neighbor.s) : "?"}
                </td>
                <td className="token-id">#{neighborId}</td>
                <td className="similarity">{similarity.toFixed(4)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
