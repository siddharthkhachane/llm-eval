# LLM Evaluation Pipeline Report

## Overview
This project implements a local LLM evaluation pipeline in five parts:
- Part A: Local serving API over Ollama
- Part B: `lm-evaluation-harness` integration
- Part C: Performance and load testing
- Part D: Determinism and output guardrails
- Part E: HellaSwag benchmark improvement workflow

Current Ollama model used by the server: `phi3:mini`.

## Part A - Serving (Ollama Endpoint)
Implemented files:
- `serve/serve.py`
- `serve/client.py`

Details:
- FastAPI server on port `8000`
- Endpoint: `POST /generate`
- Input: `{ "prompt": "..." }`
- Output: `{ "response": "..." }`
- Backend call: `http://localhost:11434/api/generate`
- Includes input validation and error handling

Run:
```bash
ollama run phi3:mini
python serve/serve.py
python serve/client.py
```

## Part B - Evaluation Integration
Implemented files:
- `eval_runner/model.py`
- `eval_runner/run_eval.py`
- `eval_runner/results/results.json`

Details:
- Custom `OllamaLM` wrapper for `lm_eval`
- Tasks configured: `hellaswag`, `mmlu`
- Logging added for task progress
- Output written to `eval_runner/results/results.json`

Run:
```bash
python eval_runner/run_eval.py --mode quick --limit 1
```
or full:
```bash
python eval_runner/run_eval.py --mode full
```

### Part B2 - Caching
Added prompt caching in:
- `eval_runner/model.py`
- Cache file: `eval_runner/cache.json`

Behavior:
- MD5 hash per prompt (`hashlib`)
- `CACHE HIT`/`CACHE MISS` logging
- Persistent across runs

## Part C - Performance and Load Testing
Implemented files:
- `perf/load_test.py`
- `perf/metrics.csv`

Details:
- Async load testing with `asyncio + aiohttp`
- Concurrency levels supported (`1,5,20` default)
- Prompt types: short and long (200+ token-style prompt)
- Metrics collected:
  - `latency`
  - `ttfb` (approx time-to-first-token via first response chunk)
  - `tokens_generated`
  - `tokens_per_second`
  - `response_length`
  - `concurrency`
  - `gpu_utilization`

Run:
```bash
python perf/load_test.py
```

### Part C2 - GPU Monitoring
Added NVML-based GPU utilization sampling using `pynvml` in `perf/load_test.py`.

## Part D - Guardrails
Implemented file:
- `guardrails/validate.py`

Details:
- Sends identical prompt 10 times
- Deterministic params:
  - `temperature=0`
  - `top_p=1`
  - `seed=42`
- Confirms identical outputs or prints warning
- Regex validation for MC answers (`A/B/C/D`)
- Structured validation output:
  - `{"valid": true/false, "answer": "A"|null}`

Run:
```bash
python guardrails/validate.py
```

## Part E - Benchmark Improvement (HellaSwag)
Implemented files:
- `improve/prepare_data.py`
- `improve/optimize_prompt.py`
- `improve/infer.py`
- `improve/eval.sh`
- Outputs:
  - `improve/hellaswag_sample.json`
  - `improve/optimized_prompts.json`
  - `improve/results.json`

Details:
- Loads HellaSwag validation data and samples 100 examples
- Simplified fields:
  - `context`
  - `ending_options`
  - `correct_answer`
- Prompt optimization supports few-shot formatting
- Inference script predicts `A/B/C/D`, scores accuracy, saves results
- Shell pipeline runs full Part E flow and prints final accuracy

Run:
```bash
bash improve/eval.sh
```

## Notes on Runtime
- `mmlu` is expensive (many sub-tasks), so full runs can take hours.
- For deadline-safe verification, quick runs with small `--limit` were used.
- Quick-run scores are integration checks, not final benchmark-quality numbers.

## Key Output Artifacts
- `eval_runner/results/results.json`
- `eval_runner/cache.json`
- `perf/metrics.csv`
- `improve/hellaswag_sample.json`
- `improve/optimized_prompts.json`
- `improve/results.json`
