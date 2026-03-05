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


def count_tokens(text: str) -> int:
    return len(text.split())


async def single_request(
    session: aiohttp.ClientSession,
    url: str,
    prompt: str,
    prompt_type: str,
    concurrency: int,
) -> RequestMetric:
    started = perf_counter()
    first_chunk_time: float | None = None
    response_text = ""

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

    latency = perf_counter() - started
    tokens = count_tokens(response_text)
    tps = (tokens / latency) if latency > 0 else 0.0

    return RequestMetric(
        timestamp=datetime.now(timezone.utc).isoformat(),
        prompt_type=prompt_type,
        latency=latency,
        ttfb=first_chunk_time,
        tokens_generated=tokens,
        tokens_per_second=tps,
        response_length=len(response_text),
        concurrency=concurrency,
    )


async def run_batch(
    session: aiohttp.ClientSession,
    url: str,
    prompt: str,
    prompt_type: str,
    concurrency: int,
    requests_per_level: int,
) -> list[RequestMetric]:
    semaphore = asyncio.Semaphore(concurrency)

    async def guarded_request() -> RequestMetric:
        async with semaphore:
            return await single_request(session, url, prompt, prompt_type, concurrency)

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
    return parser.parse_args()


async def main_async() -> None:
    args = parse_args()
    levels = [int(x.strip()) for x in args.concurrency.split(",") if x.strip()]

    timeout = aiohttp.ClientTimeout(total=300)
    all_metrics: list[RequestMetric] = []

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
                )
            )

    write_metrics(METRICS_PATH, all_metrics)
    print(f"Saved metrics to {METRICS_PATH}")


if __name__ == "__main__":
    asyncio.run(main_async())
