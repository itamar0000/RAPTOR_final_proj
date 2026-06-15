"""
Google Gemini API adapters for RAPTOR — drop-in replacements for OllamaSummarizer / OllamaQA.

Why Gemini for this project:
  - Groq free tier caps at 131,072 tokens/DAY → RAPTOR tree building (many summary
    calls per article) exhausts it fast and triggers 1400 s sleeps.
  - Gemini free tier (gemini-2.5-flash) has a much larger per-minute token budget
    (~250K TPM) so the tree-building summary calls don't stall on token limits.

NOTE: gemini-2.0-flash has ZERO free-tier quota on newer projects (Google phased
it out) — use gemini-2.5-flash (default) or gemini-2.5-flash-lite instead.

Rate-limit strategy (free tier):
  - We target ~9 req/min (6.5-sec minimum gap) to stay under the 10 RPM cap.
  - On 429: read the RetryInfo delay from the error body, sleep, then retry.
  - Max retries: 6 with exponential back-off (15s → 30s → 60s … cap 120s).

The daily REQUEST cap is the real constraint, not tokens — so pair this with a
larger RAPTOR chunk size (--tb_max_tokens 500) to cut summary calls ~5x. If you
hit the daily request cap, switch to gemini-2.5-flash-lite (higher RPD) via
--llm_model gemini-2.5-flash-lite, or resume the next day with --resume.

Embedding still uses Ollama (nomic-embed-text) — local, unlimited.

Usage:
    from gemini_models import GeminiSummarizer, GeminiQA
    from ollama_models import OllamaEmbedding
    from raptor import RetrievalAugmentation, RetrievalAugmentationConfig

    config = RetrievalAugmentationConfig(
        summarization_model=GeminiSummarizer(api_key="AIza..."),
        qa_model=GeminiQA(api_key="AIza..."),
        embedding_model=OllamaEmbedding(model="nomic-embed-text"),
        tb_max_tokens=500,   # fewer leaf nodes → fewer summary calls
    )
"""

import os
import re
import time
import logging
import threading
import requests
from pathlib import Path
from raptor import BaseSummarizationModel, BaseQAModel

# Auto-load .env from the project root (works without python-dotenv installed)
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

log = logging.getLogger(__name__)

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

# ─────────────────────────────────────────────────────────────────────────────
# Shared rate limiter (one instance across all Gemini calls in the process)
# ─────────────────────────────────────────────────────────────────────────────

class _GeminiRateLimiter:
    """
    Enforces a minimum gap between successive Gemini calls and backs off on 429.
    Thread-safe via a lock (safe for future parallel use).
    """

    def __init__(self, min_gap_seconds: float = 6.5):
        self._min_gap = min_gap_seconds   # 6.5 s → ~9 req/min (safely under 10 RPM)
        self._last_call = 0.0
        self._lock = threading.Lock()

    def wait_for_slot(self):
        with self._lock:
            now = time.time()
            wait = self._min_gap - (now - self._last_call)
            if wait > 0:
                log.debug("Rate limiter: sleeping %.1f s", wait)
                time.sleep(wait)
            self._last_call = time.time()


_LIMITER = _GeminiRateLimiter(min_gap_seconds=6.5)


# ─────────────────────────────────────────────────────────────────────────────
# Core call with retry
# ─────────────────────────────────────────────────────────────────────────────

def _gemini_chat(
    system_prompt: str,
    user_prompt: str,
    model: str,
    api_key: str,
    temperature: float = 0.1,
    max_tokens: int = 512,
    max_retries: int = 6,
    thinking_budget: int = 0,
) -> str:
    """
    Call Gemini :generateContent with rate-limit-aware retry.
    Returns the assistant message text.

    thinking_budget: Gemini 2.5 models "think" by default, and those thinking
      tokens are charged against maxOutputTokens — so a small budget (e.g. 100
      for summaries) gets entirely consumed by thinking and returns an EMPTY
      response (finishReason=MAX_TOKENS). We set thinkingBudget=0 to disable
      thinking, which is correct for summarization and direct QA. Use -1 for
      dynamic thinking, or a positive token count to cap it.
    """
    url = f"{GEMINI_BASE_URL}/{model}:generateContent"
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    }
    payload = {
        "contents": [
            {"role": "user", "parts": [{"text": user_prompt}]},
        ],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
            "thinkingConfig": {"thinkingBudget": thinking_budget},
        },
    }
    if system_prompt:
        payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}

    backoff = 15.0   # start with 15 s on first 429
    for attempt in range(1, max_retries + 1):
        _LIMITER.wait_for_slot()
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=120)

            if resp.status_code == 429:
                retry_after = _parse_retry_after(resp)
                sleep_sec = max(retry_after, backoff)
                log.warning(
                    "Gemini 429 rate limit (attempt %d/%d) — sleeping %.0f s",
                    attempt, max_retries, sleep_sec,
                )
                time.sleep(sleep_sec)
                backoff = min(backoff * 2, 120)   # cap at 2 min
                continue

            resp.raise_for_status()
            return _extract_text(resp.json())

        except requests.exceptions.Timeout:
            log.warning("Gemini timeout (attempt %d/%d) — retrying in %.0f s",
                        attempt, max_retries, backoff)
            time.sleep(backoff)
            backoff = min(backoff * 2, 120)

        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else None
            if status in (400, 401, 403):
                # Auth/quota errors won't fix themselves — fail fast, don't retry.
                body = ""
                try:
                    body = e.response.json().get("error", {}).get("message", "")
                except Exception:
                    pass
                raise RuntimeError(
                    f"Gemini API error {status}: {body or e}. "
                    "Check GEMINI_API_KEY in your .env file or pass --gemini_api_key."
                ) from e
            log.warning("Gemini HTTP error (attempt %d/%d): %s", attempt, max_retries, e)
            if attempt == max_retries:
                raise
            time.sleep(backoff)
            backoff = min(backoff * 2, 120)

        except requests.exceptions.RequestException as e:
            log.warning("Gemini request error (attempt %d/%d): %s", attempt, max_retries, e)
            if attempt == max_retries:
                raise
            time.sleep(backoff)
            backoff = min(backoff * 2, 120)

    raise RuntimeError(f"Gemini call failed after {max_retries} retries")


def _extract_text(body: dict) -> str:
    """Pull the assistant text out of a generateContent response."""
    candidates = body.get("candidates", [])
    if not candidates:
        # Often means the prompt was blocked or the response was empty.
        feedback = body.get("promptFeedback", {})
        log.warning("Gemini returned no candidates (promptFeedback=%s)", feedback)
        return ""
    parts = candidates[0].get("content", {}).get("parts", [])
    text = "".join(p.get("text", "") for p in parts)
    return text.strip()


def _parse_retry_after(resp) -> float:
    """Extract the suggested retry delay (seconds) from a 429 error body."""
    try:
        body = resp.json()
        for detail in body.get("error", {}).get("details", []):
            if detail.get("@type", "").endswith("RetryInfo"):
                delay = detail.get("retryDelay", "")   # e.g. "37s"
                m = re.search(r"([\d.]+)s", delay)
                if m:
                    return float(m.group(1)) + 1.0
    except Exception:
        pass
    # Fall back to the standard Retry-After header if present
    ra = resp.headers.get("retry-after", "")
    if ra:
        try:
            return float(re.sub(r"[^\d.]", "", ra)) + 1.0
        except ValueError:
            pass
    return 15.0


# ─────────────────────────────────────────────────────────────────────────────
# RAPTOR adapters
# ─────────────────────────────────────────────────────────────────────────────

class GeminiSummarizer(BaseSummarizationModel):
    """Gemini-backed summarizer — replaces OllamaSummarizer in RetrievalAugmentationConfig."""

    SYSTEM_PROMPT = (
        "You are a precise summarization assistant. "
        "Produce a concise, factually accurate summary that preserves all key entities, "
        "relationships, dates, and numerical values. "
        "Do not add information not present in the passage."
    )

    def __init__(
        self,
        api_key: str = "",
        model: str = "gemini-2.5-flash",
        temperature: float = 0.1,
        thinking_budget: int = 0,
    ):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "Gemini API key required. Pass api_key= or set GEMINI_API_KEY env var."
            )
        self.model = model
        self.temperature = temperature
        self.thinking_budget = thinking_budget

    def summarize(self, context: str, max_tokens: int = 150) -> str:
        user_prompt = (
            f"Summarize the following passage in at most {max_tokens} tokens:\n\n{context}"
        )
        return _gemini_chat(
            self.SYSTEM_PROMPT, user_prompt, self.model, self.api_key,
            temperature=self.temperature, max_tokens=max_tokens,
            thinking_budget=self.thinking_budget,
        )


class GeminiQA(BaseQAModel):
    """Gemini-backed QA model — replaces OllamaQA in RetrievalAugmentationConfig."""

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
        model: str = "gemini-2.5-flash",
        temperature: float = 0.1,
        max_tokens: int = 1024,   # MC reasoning + final letter; 512 truncated some answers
        thinking_budget: int = 0,
    ):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "Gemini API key required. Pass api_key= or set GEMINI_API_KEY env var."
            )
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.thinking_budget = thinking_budget

    def answer_question(self, context: str, question: str) -> str:
        user_prompt = (
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            f"Think step-by-step, then give your final answer:"
        )
        return _gemini_chat(
            self.SYSTEM_PROMPT, user_prompt, self.model, self.api_key,
            temperature=self.temperature, max_tokens=self.max_tokens,
            thinking_budget=self.thinking_budget,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Standalone helper for baseline scripts (frames_baseline, quality_baseline)
# ─────────────────────────────────────────────────────────────────────────────

def gemini_ask(
    context: str,
    question: str,
    api_key: str = "",
    model: str = "gemini-2.5-flash",
    system_prompt: str = "",
    max_tokens: int = 512,
) -> str:
    """
    One-off Gemini call for baseline scripts that don't use RAPTOR adapters.
    api_key defaults to GEMINI_API_KEY env var if not provided.
    """
    api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set")

    if not system_prompt:
        system_prompt = (
            "You are an expert reading comprehension and reasoning assistant. "
            "Read the provided context carefully, think step-by-step, "
            "and give a concise, direct final answer."
        )
    user_prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    return _gemini_chat(system_prompt, user_prompt, model, api_key, max_tokens=max_tokens)
