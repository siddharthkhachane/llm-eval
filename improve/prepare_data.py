from __future__ import annotations

import json
from pathlib import Path

from datasets import load_dataset


OUTPUT_PATH = Path(__file__).resolve().parent / "hellaswag_sample.json"
SAMPLE_SIZE = 100


def build_context(row: dict) -> str:
    context = row.get("ctx")
    if isinstance(context, str) and context.strip():
        return context.strip()

    ctx_a = row.get("ctx_a", "")
    ctx_b = row.get("ctx_b", "")
    combined = f"{ctx_a} {ctx_b}".strip()
    return combined


def main() -> None:
    dataset = load_dataset("hellaswag", split="validation")
    sample = dataset.select(range(min(SAMPLE_SIZE, len(dataset))))

    simplified: list[dict[str, object]] = []
    for row in sample:
        endings = row.get("endings", [])
        label_raw = row.get("label", -1)

        try:
            label_idx = int(label_raw)
        except (TypeError, ValueError):
            label_idx = -1

        correct_answer = endings[label_idx] if 0 <= label_idx < len(endings) else ""
        simplified.append(
            {
                "context": build_context(row),
                "ending_options": endings,
                "correct_answer": correct_answer,
            }
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(simplified, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(simplified)} examples to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
