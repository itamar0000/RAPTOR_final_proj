"""
Concise QA models for the free-form benchmarks (QASPER, NarrativeQA).

The default QA adapters prompt the model to "think step-by-step", which yields a
long reasoning paragraph. For free-form metrics (token-F1, BLEU, ROUGE-L, METEOR)
that verbosity tanks precision: a 2-word gold answer vs a 100-word reasoning dump
scores ~0.02 F1 even when correct.

These wrappers instead instruct the model to answer in a SHORT phrase with no
reasoning, so the scored output matches the short reference answers — the same
behaviour the RAPTOR paper's readers had. They implement BaseQAModel, so they
work both as RAPTOR's qa_model and directly in the baselines.
"""

import os
from raptor import BaseQAModel
from groq_models import groq_ask
from gemini_models import gemini_ask
from ollama_models import ollama_ask

SHORT_ANSWER_PROMPT = (
    "You are a precise question-answering assistant. Using ONLY the provided "
    "context, answer the question as briefly as possible — a short phrase or a "
    "single short sentence. Do NOT explain, do NOT reason out loud, do NOT repeat "
    "the question. For yes/no questions answer exactly 'Yes' or 'No'. If the "
    "context does not contain the answer, answer exactly 'Unanswerable'."
)


class ConciseOllamaQA(BaseQAModel):
    def __init__(self, model: str = "qwen2.5:14b-instruct", max_tokens: int = 256):
        self.model = model
        self.max_tokens = max_tokens

    def answer_question(self, context: str, question: str) -> str:
        return ollama_ask(context, question, model=self.model,
                          system_prompt=SHORT_ANSWER_PROMPT, max_tokens=self.max_tokens)


class ConciseGeminiQA(BaseQAModel):
    def __init__(self, api_key: str = "", model: str = "gemini-2.5-flash", max_tokens: int = 256):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self.model = model
        self.max_tokens = max_tokens

    def answer_question(self, context: str, question: str) -> str:
        return gemini_ask(context, question, api_key=self.api_key, model=self.model,
                         system_prompt=SHORT_ANSWER_PROMPT, max_tokens=self.max_tokens)


class ConciseGroqQA(BaseQAModel):
    def __init__(self, api_key: str = "", model: str = "llama-3.3-70b-versatile", max_tokens: int = 256):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY", "")
        self.model = model
        self.max_tokens = max_tokens

    def answer_question(self, context: str, question: str) -> str:
        return groq_ask(context, question, api_key=self.api_key, model=self.model,
                       system_prompt=SHORT_ANSWER_PROMPT, max_tokens=self.max_tokens)


def make_concise_qa(provider: str, model: str, groq_api_key: str = "", gemini_api_key: str = ""):
    """Factory: a short-answer QA model for the given provider."""
    if provider == "gemini":
        return ConciseGeminiQA(api_key=gemini_api_key, model=model)
    if provider == "groq":
        return ConciseGroqQA(api_key=groq_api_key, model=model)
    return ConciseOllamaQA(model=model)
