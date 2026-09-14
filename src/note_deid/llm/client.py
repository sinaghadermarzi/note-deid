"""LiteLLM-backed completion client: model aliases from configs/litellm.models.yaml, disk cache, retries, per-call
cost/latency log, and a concurrency limit (default 1, rate-limit friendly). API keys come from the environment.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path("configs/litellm.models.yaml")


@dataclass
class LLMCall:
    alias: str
    model: str
    prompt_hash: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_s: float
    cached: bool
    timestamp: float = field(default_factory=time.time)


@dataclass
class LLMResponse:
    text: str
    call: LLMCall


def prompt_hash(messages: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(json.dumps(list(messages), sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


def load_model_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


class LLMClient:
    def __init__(
        self,
        config: str | Path | Mapping[str, Any] = DEFAULT_CONFIG,
        cache_dir: str | Path | None = None,
        cache: bool = True,
        concurrency: int | None = None,
        retries: int = 3,
        mock_response: str | None = None,
    ) -> None:
        cfg = load_model_config(config) if isinstance(config, (str, Path)) else dict(config)
        self.aliases: dict[str, dict[str, Any]] = {
            m["model_name"]: dict(m.get("litellm_params", {})) for m in cfg.get("model_list", [])
        }
        settings = cfg.get("litellm_settings", {}) or {}
        nd = cfg.get("note_deid", {}) or {}
        cache_params = settings.get("cache_params", {}) or {}
        self.cache_dir = Path(cache_dir or cache_params.get("disk_cache_dir", ".cache/litellm"))
        self._cache = None
        if cache and (settings.get("cache", True)):
            import diskcache

            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._cache = diskcache.Cache(str(self.cache_dir))
        self.concurrency = int(concurrency if concurrency is not None else nd.get("concurrency", 1))
        self._sem = threading.Semaphore(self.concurrency)
        self.retries = int(settings.get("num_retries", retries))
        self.timeout = settings.get("request_timeout", 120)
        self.drop_params = bool(settings.get("drop_params", True))
        self.mock_response = mock_response
        self.log: list[LLMCall] = []

    # -- configuration -------------------------------------------------------------------------------------------
    def resolve(self, alias: str) -> dict[str, Any]:
        if alias in self.aliases:
            return dict(self.aliases[alias])
        if "/" in alias:  # a raw litellm model string
            return {"model": alias}
        raise KeyError(f"unknown model alias {alias!r}; known: {sorted(self.aliases)}")

    # -- completion ----------------------------------------------------------------------------------------------
    def complete(
        self,
        alias: str,
        messages: Sequence[Mapping[str, Any]],
        json_mode: bool = False,
        max_tokens: int | None = None,
        temperature: float | None = None,
        mock_response: str | None = None,
        **extra: Any,
    ) -> LLMResponse:
        params = self.resolve(alias)
        if temperature is not None:
            params["temperature"] = temperature
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        if json_mode:
            params["response_format"] = {"type": "json_object"}
        params.update(extra)
        mock = mock_response if mock_response is not None else self.mock_response
        cache_key_src = {k: v for k, v in params.items() if k != "api_key"}
        cache_key_src["messages"] = list(messages)
        cache_key_src["mock"] = mock
        key = hashlib.sha256(json.dumps(cache_key_src, sort_keys=True, default=str).encode()).hexdigest()
        p_hash = prompt_hash(messages)

        if self._cache is not None and key in self._cache:
            cached = self._cache[key]
            call = LLMCall(alias, params["model"], p_hash, cached["in"], cached["out"], 0.0, 0.0, True)
            self.log.append(call)
            return LLMResponse(cached["text"], call)

        text, usage_in, usage_out, cost, latency = self._call(params, list(messages), mock)
        if self._cache is not None:
            self._cache[key] = {"text": text, "in": usage_in, "out": usage_out}
        call = LLMCall(alias, params["model"], p_hash, usage_in, usage_out, cost, latency, False)
        self.log.append(call)
        return LLMResponse(text, call)

    def _call(
        self, params: dict[str, Any], messages: list[Mapping[str, Any]], mock: str | None
    ) -> tuple[str, int, int, float, float]:
        import litellm
        from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

        retryable = (
            litellm.exceptions.RateLimitError,
            litellm.exceptions.APIConnectionError,
            litellm.exceptions.Timeout,
            litellm.exceptions.ServiceUnavailableError,
            litellm.exceptions.InternalServerError,
        )

        @retry(
            stop=stop_after_attempt(max(1, self.retries)),
            wait=wait_exponential(multiplier=1, min=2, max=30),
            retry=retry_if_exception_type(retryable),
            reraise=True,
        )
        def once() -> Any:
            kwargs: dict[str, Any] = dict(params)
            kwargs["messages"] = messages
            kwargs.setdefault("timeout", self.timeout)
            # Providers reject params they do not support (e.g. temperature on reasoning models): drop them.
            kwargs.setdefault("drop_params", self.drop_params)
            if mock is not None:
                kwargs["mock_response"] = mock
            return litellm.completion(**kwargs)

        with self._sem:
            t0 = time.perf_counter()
            resp = once()
            latency = time.perf_counter() - t0
        text = resp.choices[0].message.content or ""
        usage = getattr(resp, "usage", None)
        usage_in = int(getattr(usage, "prompt_tokens", 0) or 0)
        usage_out = int(getattr(usage, "completion_tokens", 0) or 0)
        try:
            cost = float(litellm.completion_cost(completion_response=resp) or 0.0)
        except Exception:  # unknown model pricing, mock responses, ...
            cost = 0.0
        return text, usage_in, usage_out, cost, latency

    # -- accounting ----------------------------------------------------------------------------------------------
    def cost_summary(self) -> dict[str, float | int]:
        return {
            "calls": len(self.log),
            "cached_calls": sum(1 for c in self.log if c.cached),
            "input_tokens": sum(c.input_tokens for c in self.log),
            "output_tokens": sum(c.output_tokens for c in self.log),
            "cost_usd": round(sum(c.cost_usd for c in self.log), 6),
            "latency_s": round(sum(c.latency_s for c in self.log), 3),
        }

    def write_log(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for c in self.log:
                fh.write(json.dumps(asdict(c)) + "\n")
        return path
