from __future__ import annotations

import json
import re
from typing import Any

import requests


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3"
RUNS = 10
MC_REGEX = re.compile(r"\b([ABCD])\b", re.IGNORECASE)


def generate_deterministic(prompt: str) -> str:
    payload: dict[str, Any] = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0,
            "top_p": 1,
            "seed": 42,
        },
    }
    response = requests.post(OLLAMA_URL, json=payload, timeout=120)
    response.raise_for_status()
    data = response.json()
    output = data.get("response")
    if not isinstance(output, str):
        raise RuntimeError("Ollama response missing 'response' field.")
    return output.strip()


def validate_multiple_choice(text: str) -> dict[str, Any]:
    match = MC_REGEX.search(text)
    if match:
        answer = match.group(1).upper()
        return {"valid": True, "answer": answer}
    return {"valid": False, "answer": None}


def main() -> None:
    prompt = (
        "Which option is a primary color? "
        "A) Purple B) Green C) Red D) Brown. "
        "Reply with only A, B, C, or D."
    )

    outputs: list[str] = []
    for _ in range(RUNS):
        outputs.append(generate_deterministic(prompt))

    first = outputs[0] if outputs else ""
    all_identical = all(item == first for item in outputs)

    if all_identical:
        print("Deterministic mode confirmed")
        print("All outputs identical")
    else:
        print("WARNING: Deterministic outputs mismatch")

    validation = validate_multiple_choice(first)
    print(json.dumps(validation))


if __name__ == "__main__":
    main()
