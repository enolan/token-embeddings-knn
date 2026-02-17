import type { EmbeddingType } from "../hooks/useModelData";

interface Props {
  value: EmbeddingType;
  onChange: (type: EmbeddingType) => void;
}

export function EmbeddingToggle({ value, onChange }: Props) {
  return (
    <div className="embedding-toggle">
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
