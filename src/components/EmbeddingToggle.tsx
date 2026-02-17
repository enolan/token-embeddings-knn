import type { EmbeddingType } from "../hooks/useModelData";

interface Props {
  value: EmbeddingType;
  onChange: (type: EmbeddingType) => void;
  availableTypes: EmbeddingType[];
}

export function EmbeddingToggle({ value, onChange, availableTypes }: Props) {
  const hasOutput = availableTypes.includes("output");

  if (!hasOutput) {
    return (
      <div className="embedding-toggle tied">
        <span className="tied-label">Tied embeddings</span>
      </div>
    );
  }

  return (
    <div className="embedding-toggle">
      <span
        className={`embedding-slider${value === "output" ? " right" : ""}`}
      />
      <button
        className={value === "input" ? "active" : ""}
        onClick={() => onChange("input")}
      >
        Input
      </button>
      <button
        className={value === "output" ? "active" : ""}
        onClick={() => onChange("output")}
      >
        Output
      </button>
    </div>
  );
}
