"""
QuALITY baseline runner — NO RAPTOR tree.

Pipeline:
  1. Split article into fixed-size overlapping chunks (~200 tokens each)
  2. Embed every chunk with Ollama (nomic-embed-text)
  3. Embed the question, pick top-k chunks by cosine similarity
  4. Feed concatenated chunks + question to Ollama LLM
  5. Extract the A/B/C/D letter and score

This is the "SBERT without RAPTOR" condition from Table 2 of the paper
(54.9% accuracy with GPT-4).  Comparing this against quality_runner.py
(RAPTOR collapsed) shows the gain from the tree structure.

Run:
    python quality_baseline.py --llm_model llama3.1:8b --max_samples 100
"""

import json
import time
import argparse
import re
import logging
import requests
import numpy as np
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
from groq_models import groq_ask

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

OPTION_LETTERS = ["A", "B", "C", "D"]
OLLAMA_BASE_URL = "http://localhost:11434"


# ─────────────────────────────────────────────────────────────────────────────
# Text chunking
# ─────────────────────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50) -> list[str]:
    """
    Split text into overlapping word-based chunks.
    chunk_size / overlap are in words (rough token proxy; 1 token ≈ 0.75 words).
    """
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += chunk_size - overlap
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Ollama helpers
# ─────────────────────────────────────────────────────────────────────────────

def embed(text: str, model: str) -> list[float]:
    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/embed",
        json={"model": model, "input": text},
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    embeddings = data.get("embeddings") or data.get("embedding")
    if not embeddings:
        raise ValueError(f"No embeddings returned for model '{model}'")
    return embeddings[0] if isinstance(embeddings[0], list) else embeddings


def cosine_similarity(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a), np.array(b)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / denom) if denom > 0 else 0.0


SYSTEM_PROMPT = (
    "You are an expert reading comprehension assistant. "
    "Read the provided context carefully and answer the multiple-choice question. "
    "Think step-by-step, then give your final answer as a single letter: A, B, C, or D."
)


def ask_llm(context: str, mc_question: str, model: str, max_tokens: int = 300) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\n{mc_question}"},
        ],
        "options": {"temperature": 0.1, "num_predict": max_tokens},
        "stream": False,
    }
    resp = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["message"]["content"].strip()


# ─────────────────────────────────────────────────────────────────────────────
# Answer extraction + gold label helpers (same as quality_runner.py)
# ─────────────────────────────────────────────────────────────────────────────

def extract_letter(text: str) -> str:
    text = text.strip()
    if text and text[0].upper() in OPTION_LETTERS:
        return text[0].upper()
    m = re.search(r"\b(?:answer(?:\s+is)?|option|choice)[:\s]+([A-D])\b", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    matches = re.findall(r"\b([A-D])\b", text)
    return matches[-1].upper() if matches else ""


def gold_to_letter(raw) -> str:
    if isinstance(raw, int) and 1 <= raw <= 4:
        return OPTION_LETTERS[raw - 1]
    if isinstance(raw, str) and raw.strip().upper() in OPTION_LETTERS:
        return raw.strip().upper()
    return str(raw)


def format_mc_question(question: str, options: list) -> str:
    opts = "\n".join(f"{OPTION_LETTERS[i]}. {opt}" for i, opt in enumerate(options))
    return (
        f"{question}\n\nOptions:\n{opts}\n\n"
        "Read the context carefully, reason step-by-step, then give your final answer "
        "as a single letter: A, B, C, or D."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def build_output_filename(args) -> str:
    return f"quality_baseline_top{args.top_k}_results.jsonl"


def _flush(results, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def run(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / build_output_filename(args)
    log.info("Results -> %s", results_path)

    # Resume support
    already_done = 0
    existing_results = []
    if args.resume and results_path.exists():
        with open(results_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    existing_results.append(json.loads(line))
        already_done = len(existing_results)
        log.info("Resume: skipping %d completed rows", already_done)

    log.info("Loading emozilla/quality (split=%s) ...", args.split)
    ds = load_dataset("emozilla/quality", split=args.split)
    n_total = len(ds)
    if args.max_samples > 0:
        n_total = min(args.max_samples, n_total)
        ds = ds.select(range(n_total))
    log.info("Dataset: %d samples | already done: %d | remaining: %d",
             n_total, already_done, n_total - already_done)

    cols = ds.column_names
    log.info("Columns: %s", cols)
    article_field  = next((c for c in cols if c.lower() in ("article", "document", "context", "text")), cols[0])
    question_field = next((c for c in cols if "question" in c.lower()), None)
    options_field  = next((c for c in cols if c.lower() in ("options", "choices")), None)
    gold_field     = next(
        (c for c in cols if c.lower() in ("gold_label", "writer_label", "turker_label", "gold", "label", "answer")),
        None,
    )
    log.info("Fields -> article:'%s' question:'%s' options:'%s' gold:'%s'",
             article_field, question_field, options_field, gold_field)

    results = list(existing_results)
    correct = sum(1 for r in results if r.get("correct"))

    for idx, row in enumerate(tqdm(ds, desc="QuALITY baseline", total=n_total)):
        if idx < already_done:
            continue

        article      = row.get(article_field, "") or ""
        question     = row.get(question_field, "") if question_field else ""
        options      = row.get(options_field, [])  if options_field  else []
        gold_raw     = row.get(gold_field)          if gold_field     else None
        if gold_raw is None:
            for fb in ("writer_label", "gold_label", "turker_label"):
                if fb in row and row[fb] is not None:
                    gold_raw = row[fb]
                    break
        gold_letter  = gold_to_letter(gold_raw)

        if not article.strip():
            results.append({"idx": idx, "question": question, "gold_letter": gold_letter,
                            "options": options, "predicted_letter": "", "raw_answer": "",
                            "correct": False, "error": "empty_article", "method": "baseline"})
            _flush(results, results_path)
            continue

        # ── Chunk + embed ────────────────────────────────────────────────────
        try:
            chunks = chunk_text(article, chunk_size=args.chunk_size, overlap=args.overlap)
            log.info("[%d/%d] %d chunks from %d chars | embedding ...",
                     idx + 1, n_total, len(chunks), len(article))

            chunk_embeddings = [embed(c, args.embed_model) for c in chunks]
            q_embedding = embed(question, args.embed_model)

            scores = [cosine_similarity(q_embedding, ce) for ce in chunk_embeddings]
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:args.top_k]
            top_chunks = [chunks[i] for i in sorted(top_indices)]
            context = "\n\n---\n\n".join(top_chunks)

            mc_question = format_mc_question(question, options)
            if args.llm_provider == "groq":
                api_key = args.groq_api_key or __import__("os").environ.get("GROQ_API_KEY", "")
                raw_answer = groq_ask(context, mc_question, api_key=api_key, model=args.llm_model)
            else:
                raw_answer = ask_llm(context, mc_question, args.llm_model)
            predicted_letter = extract_letter(raw_answer)
            is_correct = (predicted_letter == gold_letter)

        except Exception as e:
            log.warning("Error row %d: %s", idx, e, exc_info=True)
            results.append({"idx": idx, "question": question, "gold_letter": gold_letter,
                            "options": options, "predicted_letter": "", "raw_answer": "",
                            "correct": False, "error": str(e), "method": "baseline"})
            _flush(results, results_path)
            continue

        if is_correct:
            correct += 1

        results.append({
            "idx":              idx,
            "question":         question,
            "gold_letter":      gold_letter,
            "options":          options,
            "predicted_letter": predicted_letter,
            "raw_answer":       raw_answer,
            "correct":          is_correct,
            "num_chunks":       len(chunks),
            "top_k":            args.top_k,
            "method":           "baseline_cosine",
        })

        acc = correct / len(results)
        log.info("[%d/%d] Gold=%s Pred=%s %s | Running acc: %.1f%%",
                 len(results), n_total, gold_letter, predicted_letter,
                 "CORRECT" if is_correct else "WRONG", acc * 100)

        _flush(results, results_path)
        time.sleep(0.1)

    total = len(results)
    accuracy = correct / total if total else 0.0
    log.info("Done. %d total | %d correct | accuracy=%.1f%% | saved -> %s",
             total, correct, accuracy * 100, results_path)
    print(f"\n{'='*60}")
    print(f"Baseline (no RAPTOR) Accuracy: {correct}/{total} = {accuracy:.1%}")
    print(f"Paper SBERT w/o RAPTOR:  54.9%  |  SBERT + RAPTOR:  56.6%")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="QuALITY baseline — plain chunking + cosine retrieval, no RAPTOR tree"
    )
    parser.add_argument("--llm_model",    default="llama-3.3-70b-versatile")
    parser.add_argument("--embed_model",  default="nomic-embed-text")
    parser.add_argument("--llm_provider", default="groq", choices=["ollama", "groq"])
    parser.add_argument("--groq_api_key", default="")
    parser.add_argument("--max_samples",  type=int, default=100, help="0 = full dataset")
    parser.add_argument("--split",        default="train",
                        choices=["train", "validation", "test"])
    parser.add_argument("--output_dir",   default="results/quality")
    parser.add_argument("--chunk_size",   type=int, default=200,
                        help="Chunk size in words (~150 tokens)")
    parser.add_argument("--overlap",      type=int, default=50,
                        help="Word overlap between adjacent chunks")
    parser.add_argument("--top_k",        type=int, default=5,
                        help="Number of top chunks to retrieve per question")
    parser.add_argument("--resume",       action="store_true",
                        help="Skip rows already saved in the output file")

    run(parser.parse_args())
