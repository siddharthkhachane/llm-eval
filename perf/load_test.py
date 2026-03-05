from __future__ import annotations

import argparse
import asyncio
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import aiohttp
from pynvml import (
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetUtilizationRates,
    nvmlInit,
    nvmlShutdown,
)


DEFAULT_URL = "http://localhost:8000/generate"
DEFAULT_CONCURRENCY_LEVELS = [1, 5, 20]
DEFAULT_REQUESTS_PER_LEVEL = 6
METRICS_PATH = Path(__file__).resolve().parent / "metrics.csv"


SHORT_PROMPT = "Summarize why clean APIs matter in one sentence."
LONG_PROMPT = (
    "You are a precise technical writer. Read the following repeated notes and then produce a concise summary. "
    + " ".join(["Latency throughput reliability observability reproducibility."] * 45)
)


@dataclass
class RequestMetric:
    timestamp: str
    prompt_type: str
    latency: float
    ttfb: float
    tokens_generated: int
    tokens_per_second: float
    response_length: int
    concurrency: int
    gpu_utilization: float


def count_tokens(text: str) -> int:
    return len(text.split())


async def single_request(
    session: aiohttp.ClientSession,
    url: str,
    prompt: str,
    prompt_type: str,
    concurrency: int,
    gpu_handle,
) -> RequestMetric:
    started = perf_counter()
    first_chunk_time: float | None = None
    response_text = ""
    gpu_samples: list[float] = []
    stop_sampling = asyncio.Event()

    async def sample_gpu() -> None:
        while not stop_sampling.is_set():
            gpu_samples.append(float(nvmlDeviceGetUtilizationRates(gpu_handle).gpu))
            await asyncio.sleep(0.05)

    sampler_task = asyncio.create_task(sample_gpu())

    try:
        async with session.post(url, json={"prompt": prompt}) as response:
            response.raise_for_status()

            chunks: list[bytes] = []
            async for chunk in response.content.iter_chunked(1024):
                if first_chunk_time is None:
                    first_chunk_time = perf_counter() - started
                chunks.append(chunk)

            if first_chunk_time is None:
                first_chunk_time = perf_counter() - started

            raw_body = b"".join(chunks).decode("utf-8", errors="replace")
            parsed = json.loads(raw_body)
            candidate = parsed.get("response", "")
            if isinstance(candidate, str):
                response_text = candidate
    except Exception as exc:
        response_text = f"ERROR: {exc}"
        if first_chunk_time is None:
            first_chunk_time = perf_counter() - started
    finally:
        stop_sampling.set()
        await sampler_task

    latency = perf_counter() - started
    tokens = count_tokens(response_text)
    tps = (tokens / latency) if latency > 0 else 0.0
    avg_gpu_util = (sum(gpu_samples) / len(gpu_samples)) if gpu_samples else 0.0

    return RequestMetric(
        timestamp=datetime.now(timezone.utc).isoformat(),
        prompt_type=prompt_type,
        latency=latency,
        ttfb=first_chunk_time,
        tokens_generated=tokens,
        tokens_per_second=tps,
        response_length=len(response_text),
        concurrency=concurrency,
        gpu_utilization=avg_gpu_util,
    )


async def run_batch(
    session: aiohttp.ClientSession,
    url: str,
    prompt: str,
    prompt_type: str,
    concurrency: int,
    requests_per_level: int,
    gpu_handle,
) -> list[RequestMetric]:
    semaphore = asyncio.Semaphore(concurrency)

    async def guarded_request() -> RequestMetric:
        async with semaphore:
            return await single_request(
                session=session,
                url=url,
                prompt=prompt,
                prompt_type=prompt_type,
                concurrency=concurrency,
                gpu_handle=gpu_handle,
            )

    tasks = [asyncio.create_task(guarded_request()) for _ in range(requests_per_level)]
    return await asyncio.gather(*tasks)


def write_metrics(path: Path, metrics: list[RequestMetric]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "timestamp",
                "prompt_type",
                "latency",
                "ttfb",
                "tokens_generated",
                "tokens_per_second",
                "response_length",
                "concurrency",
                "gpu_utilization",
            ]
        )
        for metric in metrics:
            writer.writerow(
                [
                    metric.timestamp,
                    metric.prompt_type,
                    f"{metric.latency:.6f}",
                    f"{metric.ttfb:.6f}",
                    metric.tokens_generated,
                    f"{metric.tokens_per_second:.6f}",
                    metric.response_length,
                    metric.concurrency,
                    f"{metric.gpu_utilization:.2f}",
                ]
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load test local Ollama FastAPI endpoint.")
    parser.add_argument("--url", default=DEFAULT_URL, help="Generation endpoint URL")
    parser.add_argument(
        "--concurrency",
        default="1,5,20",
        help="Comma-separated concurrency levels, e.g. 1,5,20",
    )
    parser.add_argument(
        "--requests-per-level",
        type=int,
        default=DEFAULT_REQUESTS_PER_LEVEL,
        help="Number of requests per concurrency level and prompt type",
    )
    parser.add_argument(
        "--gpu-index",
        type=int,
        default=0,
        help="GPU index to monitor via NVML",
    )
    return parser.parse_args()


async def main_async() -> None:
    args = parse_args()
    levels = [int(x.strip()) for x in args.concurrency.split(",") if x.strip()]

    nvmlInit()
    gpu_handle = nvmlDeviceGetHandleByIndex(args.gpu_index)

    timeout = aiohttp.ClientTimeout(total=300)
    all_metrics: list[RequestMetric] = []

    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            for level in levels:
                print(f"Running load test with concurrency={level} (short prompt)")
                all_metrics.extend(
                    await run_batch(
                        session=session,
                        url=args.url,
                        prompt=SHORT_PROMPT,
                        prompt_type="short",
                        concurrency=level,
                        requests_per_level=args.requests_per_level,
                        gpu_handle=gpu_handle,
                    )
                )

                print(f"Running load test with concurrency={level} (long prompt)")
                all_metrics.extend(
                    await run_batch(
                        session=session,
                        url=args.url,
                        prompt=LONG_PROMPT,
                        prompt_type="long",
                        concurrency=level,
                        requests_per_level=args.requests_per_level,
                        gpu_handle=gpu_handle,
                    )
                )
    finally:
        nvmlShutdown()

    write_metrics(METRICS_PATH, all_metrics)
    print(f"Saved metrics to {METRICS_PATH}")


if __name__ == "__main__":
    asyncio.run(main_async())
