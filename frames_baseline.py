"""
FRAMES baseline runner — NO RAPTOR tree.

Pipeline:
  1. Fetch Wikipedia articles for each question (same as frames_runner.py)
  2. Split all articles into fixed-size overlapping chunks (~200 words)
  3. Embed every chunk with Ollama (nomic-embed-text)
  4. Embed the question, pick top-k chunks by cosine similarity
  5. Feed concatenated chunks + question to Ollama LLM
  6. Save results

Compare this against frames_runner.py (with RAPTOR tree) to show
whether RAPTOR's tree structure helps on multi-hop FRAMES questions.

Run:
    python frames_baseline.py --llm_model qwen2.5:7b-instruct --max_samples 100
"""

import ast
import json
import re
import time
import argparse
import logging
import requests
import numpy as np
from pathlib import Path
from urllib.parse import unquote

from datasets import load_dataset
from tqdm import tqdm
from groq_models import groq_ask

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

OLLAMA_BASE_URL = "http://localhost:11434"
WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKI_HEADERS = {"User-Agent": "RAPTOR-FRAMES-Baseline/1.0"}
TEXT_CACHE = {}


# ─────────────────────────────────────────────────────────────────────────────
# Wikipedia helpers (identical to frames_runner.py)
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_by_title(title):
    params = {
        "action": "query", "prop": "extracts", "exintro": False,
        "explaintext": True, "redirects": True,
        "titles": title, "format": "json",
    }
    resp = requests.get(WIKI_API, params=params, timeout=30, headers=WIKI_HEADERS)
    resp.raise_for_status()
    pages = resp.json()["query"]["pages"]
    page = next(iter(pages.values()))
    if str(page.get("pageid", -1)) == "-1":
        return ""
    return page.get("extract", "")


def _search_title(query):
    params = {
        "action": "opensearch", "search": query, "limit": 1,
        "redirects": "resolve", "format": "json",
    }
    resp = requests.get(WIKI_API, params=params, timeout=15, headers=WIKI_HEADERS)
    resp.raise_for_status()
    data = resp.json()
    titles = data[1] if len(data) > 1 else []
    return titles[0] if titles else None


def fetch_wikipedia_text(title):
    cache_key = title.lower().strip()
    if cache_key in TEXT_CACHE:
        return TEXT_CACHE[cache_key]
    text = _fetch_by_title(title)
    if not text or not text.strip():
        resolved = _search_title(title)
        if resolved and resolved.lower() != title.lower():
            text = _fetch_by_title(resolved)
    TEXT_CACHE[cache_key] = text
    return text


def title_from_url(url):
    match = re.search(r"wikipedia\.org/wiki/(.+)$", url)
    if match:
        return unquote(match.group(1)).replace("_", " ")
    return unquote(url.split("/")[-1]).replace("_", " ")


def parse_wiki_links(raw):
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        candidates = [str(x).strip() for x in raw]
    elif isinstance(raw, str):
        s = raw.strip()
        if s.startswith("["):
            try:
                parsed = ast.literal_eval(s)
                candidates = [str(x).strip() for x in parsed]
            except Exception:
                candidates = []
        elif "\n" in s:
            candidates = [line.strip() for line in s.splitlines()]
        else:
            candidates = [s]
    else:
        candidates = []
    return [c for c in candidates if c and "wikipedia.org/wiki/" in c]


def fetch_all_articles(wiki_links):
    texts = []
    for url in wiki_links:
        title = title_from_url(url)
        if len(re.sub(r"[^\w]", "", title)) < 3:
            continue
        text = fetch_wikipedia_text(title)
        if text and text.strip():
            texts.append("=== " + title + " ===\n\n" + text.strip())
    return "\n\n\n".join(texts)


# ─────────────────────────────────────────────────────────────────────────────
# Chunking + retrieval
# ─────────────────────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50) -> list:
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


def embed(text: str, model: str) -> list:
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


def cosine_similarity(a, b):
    va, vb = np.array(a), np.array(b)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / denom) if denom > 0 else 0.0


def retrieve_top_k(question: str, chunks: list, embed_model: str, top_k: int) -> list:
    chunk_embs = [embed(c, embed_model) for c in chunks]
    q_emb = embed(question, embed_model)
    scores = [cosine_similarity(q_emb, ce) for ce in chunk_embs]
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [chunks[i] for i in sorted(top_idx)]


# ─────────────────────────────────────────────────────────────────────────────
# LLM
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an expert research assistant specialising in multi-step reasoning. "
    "You will be given context (one or more Wikipedia passages) and a question "
    "that may require chaining several facts together.\n\n"
    "Instructions:\n"
    "1. Read the context carefully and identify all relevant facts.\n"
    "2. Think step-by-step to connect those facts and answer the question.\n"
    "3. Give a concise, direct final answer.\n"
    "4. If information is genuinely absent, state what is missing and give the best partial answer."
)


def ask_llm(context: str, question: str, model: str, max_tokens: int = 512) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"Context:\n{context}\n\n"
                f"Question: {question}\n\n"
                f"Think step-by-step, then give your final answer:"
            )},
        ],
        "options": {"temperature": 0.1, "num_predict": max_tokens},
        "stream": False,
    }
    resp = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["message"]["content"].strip()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def _flush(results, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def run(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / f"frames_baseline_top{args.top_k}_results.jsonl"
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

    log.info("Loading google/frames-benchmark ...")
    ds = load_dataset("google/frames-benchmark", split="test")
    n_total = len(ds)
    if args.max_samples > 0:
        n_total = min(args.max_samples, n_total)
        ds = ds.select(range(n_total))
    log.info("Running on %d samples | already done: %d", n_total, already_done)

    cols = ds.column_names
    question_field = next((c for c in cols if c.lower() in ("prompt", "question")), cols[0])
    answer_field   = next((c for c in cols if c.lower() in ("answer", "gold_answer")), cols[1])
    links_field    = next(
        (c for c in cols if any(k in c.lower() for k in ("wiki", "link", "url", "source"))),
        None,
    )
    log.info("Fields -> question:'%s'  answer:'%s'  links:'%s'",
             question_field, answer_field, links_field)

    results = list(existing_results)

    for idx, row in enumerate(tqdm(ds, desc="FRAMES baseline", total=n_total)):
        if idx < already_done:
            continue

        question    = row[question_field]
        gold_answer = row[answer_field]
        raw_links   = row[links_field] if links_field else None
        wiki_links  = parse_wiki_links(raw_links)

        if not wiki_links:
            results.append({
                "idx": idx, "question": question, "gold_answer": gold_answer,
                "baseline_answer": "", "wiki_links": [], "method": "baseline_cosine",
                "error": "no_wiki_links",
            })
            _flush(results, results_path)
            continue

        try:
            combined = fetch_all_articles(wiki_links)
            if not combined.strip():
                results.append({
                    "idx": idx, "question": question, "gold_answer": gold_answer,
                    "baseline_answer": "", "wiki_links": wiki_links,
                    "method": "baseline_cosine", "error": "all_articles_empty",
                })
                _flush(results, results_path)
                continue

            chunks = chunk_text(combined, chunk_size=args.chunk_size, overlap=args.overlap)
            log.info("[%d/%d] %d chunks from %d chars | retrieving top-%d ...",
                     idx + 1, n_total, len(chunks), len(combined), args.top_k)

            top_chunks = retrieve_top_k(question, chunks, args.embed_model, args.top_k)
            context = "\n\n---\n\n".join(top_chunks)
            if args.llm_provider == "groq":
                api_key = args.groq_api_key or __import__("os").environ.get("GROQ_API_KEY", "")
                answer = groq_ask(context, question, api_key=api_key, model=args.llm_model)
            else:
                answer = ask_llm(context, question, args.llm_model)

        except Exception as e:
            log.warning("Error row %d '%s': %s", idx, question[:60], e, exc_info=True)
            results.append({
                "idx": idx, "question": question, "gold_answer": gold_answer,
                "baseline_answer": "", "wiki_links": wiki_links,
                "method": "baseline_cosine", "error": str(e),
            })
            _flush(results, results_path)
            continue

        gold_norm   = re.sub(r"\s+", " ", gold_answer.strip().lower())
        answer_norm = re.sub(r"\s+", " ", answer.strip().lower())
        gold_found  = gold_norm in answer_norm

        results.append({
            "idx":              idx,
            "question":         question,
            "gold_answer":      gold_answer,
            "baseline_answer":  answer,
            "wiki_links":       wiki_links,
            "gold_found":       gold_found,
            "method":           "baseline_cosine",
            "top_k":            args.top_k,
        })

        log.info("[%d/%d] gold_found=%s | %s", len(results), n_total,
                 gold_found, question[:60])

        _flush(results, results_path)
        time.sleep(0.3)

    answered = sum(1 for r in results if r.get("baseline_answer"))
    gold_found = sum(1 for r in results if r.get("gold_found"))
    log.info("Done. %d total | %d answered | %d gold substring match | saved -> %s",
             len(results), answered, gold_found, results_path)
    print(f"\n{'='*60}")
    print(f"FRAMES Baseline (no RAPTOR): {len(results)} questions")
    print(f"  Answered:          {answered}/{len(results)}")
    print(f"  Gold string match: {gold_found}/{len(results)} = {gold_found/len(results):.1%}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FRAMES baseline — flat cosine chunking, no RAPTOR tree"
    )
    parser.add_argument("--llm_model",   default="llama-3.3-70b-versatile")
    parser.add_argument("--llm_provider", default="groq", choices=["ollama", "groq"])
    parser.add_argument("--groq_api_key", default="")
    parser.add_argument("--embed_model", default="nomic-embed-text")
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--output_dir",  default="results/frames_baseline")
    parser.add_argument("--chunk_size",  type=int, default=200)
    parser.add_argument("--overlap",     type=int, default=50)
    parser.add_argument("--top_k",       type=int, default=5)
    parser.add_argument("--resume",      action="store_true")

    run(parser.parse_args())
