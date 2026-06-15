"""
Ollama adapters for RAPTOR — drop-in replacements for the OpenAI defaults.
Implements BaseSummarizationModel, BaseQAModel, and BaseEmbeddingModel
using a local Ollama server (default: http://localhost:11434).

Usage:
    from ollama_models import OllamaSummarizer, OllamaQA, OllamaEmbedding
    from raptor import RetrievalAugmentation, RetrievalAugmentationConfig

    config = RetrievalAugmentationConfig(
        summarization_model=OllamaSummarizer("llama3"),
        qa_model=OllamaQA("llama3"),
        embedding_model=OllamaEmbedding("nomic-embed-text"),
    )
    RA = RetrievalAugmentation(config=config)
"""

import requests
import numpy as np
from typing import Optional
from raptor import BaseSummarizationModel, BaseQAModel, BaseEmbeddingModel


OLLAMA_BASE_URL = "http://localhost:11434"


# ──────────────────────────────────────────────────────────────────────────────
# Summarization
# ──────────────────────────────────────────────────────────────────────────────

class OllamaSummarizer(BaseSummarizationModel):
    """
    Uses an Ollama chat model to produce cluster summaries.
    Recommended models: llama3, mistral, gemma2, phi3
    """

    SYSTEM_PROMPT = (
        "You are a precise summarization assistant. "
        "When given a passage, produce a concise, factually accurate summary "
        "that preserves all key entities, relationships, dates, and numerical values. "
        "Do not add information not present in the passage."
    )

    def __init__(
        self,
        model: str = "llama3",
        base_url: str = OLLAMA_BASE_URL,
        temperature: float = 0.1,
    ):
        self.model = model
        self.base_url = base_url
        self.temperature = temperature

    def summarize(self, context: str, max_tokens: int = 150) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Summarize the following passage in at most {max_tokens} tokens:\n\n"
                        f"{context}"
                    ),
                },
            ],
            "options": {
                "temperature": self.temperature,
                "num_predict": max_tokens,
            },
            "stream": False,
        }
        resp = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()


# ──────────────────────────────────────────────────────────────────────────────
# QA
# ──────────────────────────────────────────────────────────────────────────────

class OllamaQA(BaseQAModel):
    """
    Uses an Ollama chat model for the final answer-generation step.

    FIX #4: The original system prompt told the model to give up if the context
    lacked information.  FRAMES questions require multi-hop reasoning — chaining
    facts, doing arithmetic, resolving indirect references.  The new prompt
    encourages the model to reason step-by-step over the retrieved context
    rather than immediately refusing.
    """

    SYSTEM_PROMPT = (
        "You are an expert research assistant specialising in multi-step reasoning. "
        "You will be given a context (one or more Wikipedia passages) and a question "
        "that may require chaining several facts together, resolving indirect references, "
        "or performing simple arithmetic/comparisons.\n\n"
        "Instructions:\n"
        "1. Read the context carefully and identify all relevant facts.\n"
        "2. Think step-by-step to connect those facts and answer the question.\n"
        "3. Give a concise, direct final answer.\n"
        "4. If a piece of information is genuinely absent from the context, state what "
        "is missing and give the best partial answer you can from what is available.\n"
        "Do NOT refuse to answer just because the question is complex."
    )

    def __init__(
        self,
        model: str = "llama3",
        base_url: str = OLLAMA_BASE_URL,
        temperature: float = 0.1,
        max_tokens: int = 512,
    ):
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens

    def answer_question(self, context: str, question: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Context:\n{context}\n\n"
                        f"Question: {question}\n\n"
                        f"Think step-by-step, then give your final answer:"
                    ),
                },
            ],
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
            "stream": False,
        }
        resp = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()


# ──────────────────────────────────────────────────────────────────────────────
# Embeddings
# ──────────────────────────────────────────────────────────────────────────────

class OllamaEmbedding(BaseEmbeddingModel):
    """
    Uses Ollama's /api/embed endpoint.
    Recommended models: nomic-embed-text, mxbai-embed-large, all-minilm
    Pull first: `ollama pull nomic-embed-text`
    """

    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = OLLAMA_BASE_URL,
    ):
        self.model = model
        self.base_url = base_url

    def create_embedding(self, text: str) -> list[float]:
        payload = {"model": self.model, "input": text}
        resp = requests.post(f"{self.base_url}/api/embed", json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        # Ollama returns {"embeddings": [[...]], ...}
        embeddings = data.get("embeddings") or data.get("embedding")
        if not embeddings:
            raise ValueError(f"Ollama embed returned no embeddings for model '{self.model}'")
        if isinstance(embeddings[0], list):
            return embeddings[0]
        return embeddings


# ──────────────────────────────────────────────────────────────────────────────
# Standalone one-shot chat helper (mirrors groq_ask / gemini_ask)
# ──────────────────────────────────────────────────────────────────────────────

def ollama_ask(
    context: str,
    question: str,
    model: str,
    system_prompt: str = "",
    max_tokens: int = 256,
    temperature: float = 0.1,
    base_url: str = OLLAMA_BASE_URL,
) -> str:
    """One-off Ollama chat call with a caller-supplied system prompt."""
    if not system_prompt:
        system_prompt = "You are a helpful assistant. Answer using only the context."
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"},
        ],
        "options": {"temperature": temperature, "num_predict": max_tokens},
        "stream": False,
    }
    resp = requests.post(f"{base_url}/api/chat", json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["message"]["content"].strip()