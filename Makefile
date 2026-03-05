.PHONY: help install sanity serve client eval eval-full guardrails perf part-e-data part-e-prompt part-e-infer part-e-full clean-results

PYTHON ?= python
MODE ?= quick
LIMIT ?= 1
MAX_ITEMS ?= 5
CONCURRENCY ?= 1,5,20
REQUESTS_PER_LEVEL ?= 1
FEWSHOT ?= 1

help:
	@echo "Targets:"
	@echo "  install        Install Python dependencies"
	@echo "  sanity         Single very fast sanity run"
	@echo "  serve          Run FastAPI serving endpoint"
	@echo "  client         Run sample client prompts"
	@echo "  eval           Run eval_runner with MODE and LIMIT"
	@echo "  eval-full      Shortcut for full eval"
	@echo "  guardrails     Run deterministic/output validation"
	@echo "  perf           Run load test with configurable concurrency"
	@echo "  part-e-data    Run Part E data preparation"
	@echo "  part-e-prompt  Run Part E prompt generation"
	@echo "  part-e-infer   Run Part E inference"
	@echo "  part-e-full    Run full Part E pipeline"
	@echo ""
	@echo "Mode toggles:"
	@echo "  make eval MODE=quick LIMIT=1"
	@echo "  make eval MODE=full"
	@echo "  make part-e-infer MAX_ITEMS=10"

install:
	$(PYTHON) -m pip install -r requirements.txt

sanity:
	$(PYTHON) eval_runner/run_eval.py --mode quick --limit 1

serve:
	$(PYTHON) serve/serve.py

client:
	$(PYTHON) serve/client.py

eval:
	$(PYTHON) eval_runner/run_eval.py --mode $(MODE) --limit $(LIMIT)

eval-full:
	$(PYTHON) eval_runner/run_eval.py --mode full

guardrails:
	$(PYTHON) guardrails/validate.py

perf:
	$(PYTHON) perf/load_test.py --concurrency $(CONCURRENCY) --requests-per-level $(REQUESTS_PER_LEVEL)

part-e-data:
	$(PYTHON) improve/prepare_data.py

part-e-prompt:
	$(PYTHON) improve/optimize_prompt.py --fewshot $(FEWSHOT) --max-items $(MAX_ITEMS)

part-e-infer:
	$(PYTHON) improve/infer.py --max-items $(MAX_ITEMS)

part-e-full:
	$(PYTHON) improve/prepare_data.py
	$(PYTHON) improve/optimize_prompt.py --fewshot $(FEWSHOT)
	$(PYTHON) improve/infer.py

clean-results:
	@echo "Cleaning generated result artifacts"
	@rm -f eval_runner/results/results.json perf/metrics.csv improve/results.json improve/optimized_prompts.json || true
