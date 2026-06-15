"""
Groq API adapters for RAPTOR — drop-in replacements for OllamaSummarizer / OllamaQA.

Rate-limit strategy for llama-3.3-70b-versatile (free tier):
  - Hard limit : 30 req/min, 6000 tokens/min
  - We target  : 20 req/min (3-sec minimum gap) to leave headroom
  - On 429     : read Retry-After header, sleep that many seconds, then retry
  - Max retries: 6 with exponential back-off (30s → 60s → 120s … cap 300s)

Embedding still uses Ollama (nomic-embed-text) — Groq has no embedding endpoint.

Usage:
    from groq_models import GroqSummarizer, GroqQA
    from ollama_models import OllamaEmbedding
    from raptor import RetrievalAugmentation, RetrievalAugmentationConfig

    config = RetrievalAugmentationConfig(
        summarization_model=GroqSummarizer(api_key="gsk_..."),
        qa_model=GroqQA(api_key="gsk_..."),
        embedding_model=OllamaEmbedding(model="nomic-embed-text"),
    )
"""

import os
import re
import time
import logging
import threading
import requests
from raptor import BaseSummarizationModel, BaseQAModel

log = logging.getLogger(__name__)

GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"

# ─────────────────────────────────────────────────────────────────────────────
# Shared rate limiter (one instance across all Groq calls in the process)
# ─────────────────────────────────────────────────────────────────────────────

class _GroqRateLimiter:
    """
    Token-bucket style rate limiter.
    Enforces a minimum gap between successive Groq calls and backs off on 429.
    Thread-safe via a lock (safe for future parallel use).
    """

    def __init__(self, min_gap_seconds: float = 3.0):
        self._min_gap = min_gap_seconds   # 3 s → max 20 req/min (safely under 30)
        self._last_call = 0.0
        self._lock = threading.Lock()

    def wait_for_slot(self):
        """Block until the minimum inter-request gap has elapsed."""
        with self._lock:
            now = time.time()
            wait = self._min_gap - (now - self._last_call)
            if wait > 0:
                log.debug("Rate limiter: sleeping %.1f s", wait)
                time.sleep(wait)
            self._last_call = time.time()


_LIMITER = _GroqRateLimiter(min_gap_seconds=3.0)


# ─────────────────────────────────────────────────────────────────────────────
# Core call with retry
# ─────────────────────────────────────────────────────────────────────────────

def _groq_chat(
    messages: list,
    model: str,
    api_key: str,
    temperature: float = 0.1,
    max_tokens: int = 512,
    max_retries: int = 6,
) -> str:
    """
    Call Groq /chat/completions with rate-limit-aware retry.
    Returns the assistant message text.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    backoff = 30.0   # start with 30 s on first 429
    for attempt in range(1, max_retries + 1):
        _LIMITER.wait_for_slot()
        try:
            resp = requests.post(GROQ_CHAT_URL, json=payload, headers=headers, timeout=120)

            # Log remaining quota from response headers
            remaining_req = resp.headers.get("x-ratelimit-remaining-requests", "?")
            remaining_tok = resp.headers.get("x-ratelimit-remaining-tokens", "?")
            log.debug("Groq quota: req_remaining=%s tok_remaining=%s",
                      remaining_req, remaining_tok)

            if resp.status_code == 429:
                retry_after = _parse_retry_after(resp)
                sleep_sec = max(retry_after, backoff)
                log.warning(
                    "Groq 429 (attempt %d/%d) — sleeping %.0f s "
                    "(req_remaining=%s, tok_remaining=%s)",
                    attempt, max_retries, sleep_sec, remaining_req, remaining_tok,
                )
                time.sleep(sleep_sec)
                backoff = min(backoff * 2, 300)   # cap at 5 min
                continue

            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            return content.strip()

        except requests.exceptions.Timeout:
            log.warning("Groq timeout (attempt %d/%d) — retrying in %.0f s",
                        attempt, max_retries, backoff)
            time.sleep(backoff)
            backoff = min(backoff * 2, 300)

        except requests.exceptions.RequestException as e:
            log.warning("Groq request error (attempt %d/%d): %s", attempt, max_retries, e)
            if attempt == max_retries:
                raise
            time.sleep(backoff)
            backoff = min(backoff * 2, 300)

    raise RuntimeError(f"Groq call failed after {max_retries} retries")


def _parse_retry_after(resp) -> float:
    """Extract seconds from Retry-After header (may be int seconds or 'Xs' string)."""
    ra = resp.headers.get("retry-after", "")
    if not ra:
        # Try to parse from JSON body
        try:
            body = resp.json()
            msg = body.get("error", {}).get("message", "")
            m = re.search(r"try again in ([\d.]+)s", msg, re.IGNORECASE)
            if m:
                return float(m.group(1)) + 1.0
        except Exception:
            pass
        return 30.0
    try:
        return float(re.sub(r"[^\d.]", "", ra)) + 1.0
    except ValueError:
        return 30.0


# ─────────────────────────────────────────────────────────────────────────────
# RAPTOR adapters
# ─────────────────────────────────────────────────────────────────────────────

class GroqSummarizer(BaseSummarizationModel):
    """Groq-backed summarizer — replaces OllamaSummarizer in RetrievalAugmentationConfig."""

    SYSTEM_PROMPT = (
        "You are a precise summarization assistant. "
        "Produce a concise, factually accurate summary that preserves all key entities, "
        "relationships, dates, and numerical values. "
        "Do not add information not present in the passage."
    )

    def __init__(
        self,
        api_key: str = "",
        model: str = "llama-3.3-70b-versatile",
        temperature: float = 0.1,
    ):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "Groq API key required. Pass api_key= or set GROQ_API_KEY env var."
            )
        self.model = model
        self.temperature = temperature

    def summarize(self, context: str, max_tokens: int = 150) -> str:
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"Summarize the following passage in at most {max_tokens} tokens:\n\n{context}"
            )},
        ]
        return _groq_chat(
            messages, self.model, self.api_key,
            temperature=self.temperature, max_tokens=max_tokens,
        )


class GroqQA(BaseQAModel):
    """Groq-backed QA model — replaces OllamaQA in RetrievalAugmentationConfig."""

    SYSTEM_PROMPT = (
        "You are an expert research assistant specialising in multi-step reasoning. "
        "You will be given context (one or more passages) and a question that may require "
        "chaining several facts together, resolving indirect references, or arithmetic.\n\n"
        "Instructions:\n"
        "1. Read the context carefully and identify all relevant facts.\n"
        "2. Think step-by-step to connect those facts and answer the question.\n"
        "3. Give a concise, direct final answer.\n"
        "4. If a piece of information is genuinely absent, state what is missing and give "
        "the best partial answer you can from what is available.\n"
        "Do NOT refuse to answer just because the question is complex."
    )

    def __init__(
        self,
        api_key: str = "",
        model: str = "llama-3.3-70b-versatile",
        temperature: float = 0.1,
        max_tokens: int = 512,
    ):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "Groq API key required. Pass api_key= or set GROQ_API_KEY env var."
            )
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def answer_question(self, context: str, question: str) -> str:
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"Context:\n{context}\n\n"
                f"Question: {question}\n\n"
                f"Think step-by-step, then give your final answer:"
            )},
        ]
        return _groq_chat(
            messages, self.model, self.api_key,
            temperature=self.temperature, max_tokens=self.max_tokens,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Standalone helper for baseline scripts (frames_baseline, quality_baseline)
# ─────────────────────────────────────────────────────────────────────────────

def groq_ask(
    context: str,
    question: str,
    api_key: str = "",
    model: str = "llama-3.3-70b-versatile",
    system_prompt: str = "",
    max_tokens: int = 512,
) -> str:
    """
    One-off Groq call for baseline scripts that don't use RAPTOR adapters.
    api_key defaults to GROQ_API_KEY env var if not provided.
    """
    api_key = api_key or os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set")

    if not system_prompt:
        system_prompt = (
            "You are an expert reading comprehension and reasoning assistant. "
            "Read the provided context carefully, think step-by-step, "
            "and give a concise, direct final answer."
        )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"},
    ]
    return _groq_chat(messages, model, api_key, max_tokens=max_tokens)
