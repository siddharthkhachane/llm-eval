from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any

import requests
from lm_eval.api.model import LM


class OllamaLM(LM):
    """lm-eval wrapper that calls a local FastAPI endpoint backed by Ollama."""

    def __init__(self, endpoint: str = "http://localhost:8000/generate", timeout: int = 120):
        super().__init__()
        self.endpoint = endpoint
        self.timeout = timeout

    def _call_generate(self, prompt: str) -> str:
        try:
            response = requests.post(
                self.endpoint,
                json={"prompt": prompt},
                timeout=self.timeout,
            )
            response.raise_for_status()
            payload = response.json()
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(f"Failed calling local generation endpoint {self.endpoint}: {exc}") from exc
        except ValueError as exc:
            raise RuntimeError("Generation endpoint returned invalid JSON.") from exc

        output = payload.get("response")
        if not isinstance(output, str):
            raise RuntimeError("Generation endpoint response missing 'response' string field.")
        return output

    @staticmethod
    def _score_continuation(generated: str, continuation: str) -> tuple[float, bool]:
        target = continuation.strip()
        predicted = generated.strip()
        if not target:
            return 0.0, True

        predicted_prefix = predicted[: len(target)]
        ratio = SequenceMatcher(None, predicted_prefix, target).ratio()
        score = float((ratio * 2.0) - 1.0)
        is_greedy = predicted_prefix == target
        return score, is_greedy

    def loglikelihood(self, requests_list) -> list[tuple[float, bool]]:
        results: list[tuple[float, bool]] = []
        for req in requests_list:
            context, continuation = req.args
            generated = self._call_generate(context)
            score, is_greedy = self._score_continuation(generated, continuation)
            self.cache_hook.add_partial("loglikelihood", (context, continuation), (score, is_greedy))
            results.append((score, is_greedy))
        return results

    def loglikelihood_rolling(self, requests_list) -> list[float]:
        results: list[float] = []
        for req in requests_list:
            (text,) = req.args
            generated = self._call_generate(text)
            ratio = SequenceMatcher(None, generated.strip(), text.strip()).ratio()
            score = float((ratio * 2.0) - 1.0)
            self.cache_hook.add_partial("loglikelihood_rolling", (text,), score)
            results.append(score)
        return results

    def generate_until(self, requests_list) -> list[str]:
        outputs: list[str] = []
        for req in requests_list:
            context, gen_kwargs = req.args
            generated = self._call_generate(context)

            until: Any = gen_kwargs.get("until", []) if isinstance(gen_kwargs, dict) else []
            if isinstance(until, str):
                until = [until]

            cut_idx = len(generated)
            for stop in until:
                if not stop:
                    continue
                idx = generated.find(stop)
                if idx != -1:
                    cut_idx = min(cut_idx, idx)
            trimmed = generated[:cut_idx]
            self.cache_hook.add_partial("generate_until", (context, gen_kwargs), trimmed)
            outputs.append(trimmed)

        return outputs
