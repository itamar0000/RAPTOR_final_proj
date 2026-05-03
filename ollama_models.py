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
        "that preserves all key entities, relationships, and claims. "
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
    Same model as summarization is fine; can also use a larger model here.
    """

    SYSTEM_PROMPT = (
        "You are a helpful and precise question-answering assistant. "
        "Answer the question using ONLY the information in the provided context. "
        "If the context does not contain enough information, say so clearly."
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
                        f"Answer:"
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
        if isinstance(embeddings[0], list):
            return embeddings[0]
        return embeddings
