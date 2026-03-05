from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


INPUT_PATH = Path(__file__).resolve().parent / "hellaswag_sample.json"
OUTPUT_PATH = Path(__file__).resolve().parent / "optimized_prompts.json"
LETTER_MAP = ["A", "B", "C", "D"]


def detect_answer_letter(example: dict[str, Any]) -> str:
    options = example.get("ending_options", [])
    correct = example.get("correct_answer", "")
    if not isinstance(options, list) or not isinstance(correct, str):
        return "A"
    for idx, option in enumerate(options[:4]):
        if option == correct:
            return LETTER_MAP[idx]
    return "A"


def format_question_block(context: str, options: list[str]) -> str:
    lines = [
        "Context:",
        context,
        "",
        "Options:",
        f"A) {options[0]}",
        f"B) {options[1]}",
        f"C) {options[2]}",
        f"D) {options[3]}",
        "",
        "Think step-by-step then answer with a single letter.",
        "",
        "Return:",
        "Answer: <LETTER>",
    ]
    return "\n".join(lines)


def format_fewshot_block(example: dict[str, Any], index: int) -> str:
    context = str(example.get("context", "")).strip()
    options = example.get("ending_options", [])
    if not isinstance(options, list) or len(options) < 4:
        return ""
    answer = detect_answer_letter(example)
    return (
        f"Example {index}:\n"
        f"{format_question_block(context, options[:4])}\n"
        f"Answer: {answer}"
    )


def format_prompt(target: dict[str, Any], fewshots: list[dict[str, Any]]) -> str:
    context = str(target.get("context", "")).strip()
    options = target.get("ending_options", [])
    if not isinstance(options, list) or len(options) < 4:
        raise ValueError("Each example must include at least 4 ending options.")

    parts = [
        "You are a reasoning assistant.",
        "",
        "Choose the most logical ending.",
    ]

    if fewshots:
        parts.append("")
        parts.append("Few-shot examples:")
        for i, shot in enumerate(fewshots, start=1):
            shot_block = format_fewshot_block(shot, i)
            if shot_block:
                parts.append("")
                parts.append(shot_block)

    parts.append("")
    parts.append("Now solve this item:")
    parts.append("")
    parts.append(format_question_block(context, options[:4]))

    return "\n".join(parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate optimized HellaSwag prompts.")
    parser.add_argument("--input", type=Path, default=INPUT_PATH, help="Path to HellaSwag sample JSON")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH, help="Path to save formatted prompts JSON")
    parser.add_argument("--fewshot", type=int, default=0, help="Number of few-shot examples to include")
    parser.add_argument(
        "--max-items",
        type=int,
        default=100,
        help="Maximum number of prompts to generate",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.input.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of examples.")

    max_items = min(args.max_items, len(data))
    fewshot_count = max(0, args.fewshot)

    output_rows: list[dict[str, Any]] = []
    for idx in range(max_items):
        target = data[idx]
        available_fewshots = data[:idx]
        fewshots = available_fewshots[-fewshot_count:] if fewshot_count else []
        prompt = format_prompt(target, fewshots)
        output_rows.append(
            {
                "id": idx,
                "prompt": prompt,
                "answer": detect_answer_letter(target),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(output_rows, f, indent=2, ensure_ascii=False)

    print(f"Generated {len(output_rows)} prompts at {args.output}")


if __name__ == "__main__":
    main()
