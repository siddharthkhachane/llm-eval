from __future__ import annotations

from difflib import SequenceMatcher
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import requests
from lm_eval.api.model import LM


class OllamaLM(LM):
    """lm-eval wrapper that calls a local FastAPI endpoint backed by Ollama."""

    def __init__(self, endpoint: str = "http://localhost:8000/generate", timeout: int = 120):
        super().__init__()
        self.endpoint = endpoint
        self.timeout = timeout
        self.logger = logging.getLogger("eval_runner.cache")
        self.cache_path = Path(__file__).resolve().parent / "cache.json"
        self.prompt_cache: dict[str, str] = self._load_cache()

    def _load_cache(self) -> dict[str, str]:
        if not self.cache_path.exists():
            return {}

        try:
            with self.cache_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            self.logger.warning("Cache file unreadable. Starting with empty cache.")
            return {}

        if not isinstance(data, dict):
            self.logger.warning("Cache file format invalid. Starting with empty cache.")
            return {}

        cleaned: dict[str, str] = {}
        for key, value in data.items():
            if isinstance(key, str) and isinstance(value, str):
                cleaned[key] = value
        return cleaned

    def _save_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.cache_path.with_suffix(".tmp")
        with temp_path.open("w", encoding="utf-8") as f:
            json.dump(self.prompt_cache, f, ensure_ascii=False, indent=2)
        temp_path.replace(self.cache_path)

    @staticmethod
    def _prompt_hash(prompt: str) -> str:
        return hashlib.md5(prompt.encode("utf-8")).hexdigest()

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

    def _generate_with_cache(self, prompt: str) -> str:
        prompt_hash = self._prompt_hash(prompt)
        cached = self.prompt_cache.get(prompt_hash)
        if isinstance(cached, str):
            self.logger.info("CACHE HIT")
            return cached

        self.logger.info("CACHE MISS")
        output = self._call_generate(prompt)
        self.prompt_cache[prompt_hash] = output
        self._save_cache()
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
            generated = self._generate_with_cache(context)
            score, is_greedy = self._score_continuation(generated, continuation)
            self.cache_hook.add_partial("loglikelihood", (context, continuation), (score, is_greedy))
            results.append((score, is_greedy))
        return results

    def loglikelihood_rolling(self, requests_list) -> list[float]:
        results: list[float] = []
        for req in requests_list:
            (text,) = req.args
            generated = self._generate_with_cache(text)
            ratio = SequenceMatcher(None, generated.strip(), text.strip()).ratio()
            score = float((ratio * 2.0) - 1.0)
            self.cache_hook.add_partial("loglikelihood_rolling", (text,), score)
            results.append(score)
        return results

    def generate_until(self, requests_list) -> list[str]:
        outputs: list[str] = []
        for req in requests_list:
            context, gen_kwargs = req.args
            generated = self._generate_with_cache(context)

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
