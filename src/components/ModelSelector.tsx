import { availableModels } from "../hooks/useModelData";

interface Props {
  value: string;
  onChange: (modelId: string) => void;
}

export function ModelSelector({ value, onChange }: Props) {
  const models = availableModels();
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="model-selector"
    >
      {models.map((m) => (
        <option key={m} value={m}>
          {m}
        </option>
      ))}
    </select>
  );
}
