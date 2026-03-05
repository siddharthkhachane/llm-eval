from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import requests


INPUT_PATH = Path(__file__).resolve().parent / "hellaswag_sample.json"
OUTPUT_PATH = Path(__file__).resolve().parent / "results.json"
ENDPOINT = "http://localhost:8000/generate"
ANSWER_RE = re.compile(r"\b([ABCD])\b", re.IGNORECASE)
LETTER_TO_INDEX = {"A": 0, "B": 1, "C": 2, "D": 3}


def format_prompt(context: str, options: list[str]) -> str:
    return (
        "You are a reasoning assistant.\n\n"
        "Choose the most logical ending.\n\n"
        f"Context:\n{context}\n\n"
        "Options:\n"
        f"A) {options[0]}\n"
        f"B) {options[1]}\n"
        f"C) {options[2]}\n"
        f"D) {options[3]}\n\n"
        "Think step-by-step then answer with a single letter.\n\n"
        "Return:\n"
        "Answer: <LETTER>"
    )


def parse_prediction(text: str) -> str | None:
    match = ANSWER_RE.search(text)
    if not match:
        return None
    return match.group(1).upper()


def call_model(prompt: str) -> str:
    response = requests.post(ENDPOINT, json={"prompt": prompt}, timeout=120)
    response.raise_for_status()
    payload = response.json()
    output = payload.get("response")
    if not isinstance(output, str):
        raise RuntimeError("Model response missing 'response' field.")
    return output


def main() -> None:
    with INPUT_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Input file must be a list of examples.")

    total = 0
    correct = 0
    rows: list[dict[str, Any]] = []

    for idx, row in enumerate(data):
        context = str(row.get("context", "")).strip()
        options = row.get("ending_options", [])
        gold_answer = row.get("correct_answer", "")
        if not isinstance(options, list) or len(options) < 4:
            continue

        prompt = format_prompt(context, options[:4])
        output = call_model(prompt)
        pred_letter = parse_prediction(output)

        pred_text = None
        is_correct = False
        if pred_letter in LETTER_TO_INDEX:
            pred_text = options[LETTER_TO_INDEX[pred_letter]]
            is_correct = pred_text == gold_answer

        total += 1
        if is_correct:
            correct += 1

        rows.append(
            {
                "id": idx,
                "prediction_letter": pred_letter,
                "prediction_text": pred_text,
                "correct_answer": gold_answer,
                "is_correct": is_correct,
            }
        )

    accuracy = (correct / total) if total else 0.0
    results = {
        "accuracy": accuracy,
        "total_questions": total,
        "correct_predictions": correct,
        "predictions": rows,
    }

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Accuracy: {accuracy:.2f}")


if __name__ == "__main__":
    main()
