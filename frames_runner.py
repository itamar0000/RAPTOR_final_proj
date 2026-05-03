"""
FRAMES benchmark runner for RAPTOR with local Ollama models.

Run:
    python frames_runner.py --llm_model qwen2.5:7b-instruct --embed_model nomic-embed-text --max_samples 50
"""

import ast
import json
import time
import argparse
import re
import logging
import shutil
from pathlib import Path

import requests
from datasets import load_dataset
from tqdm import tqdm

from raptor import RetrievalAugmentation, RetrievalAugmentationConfig
from ollama_models import OllamaSummarizer, OllamaQA, OllamaEmbedding


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# Wikipedia fetcher
# ------------------------------------------------------------------------------

def fetch_wikipedia_text(title: str) -> str:
    params = {
        "action": "query",
        "prop": "extracts",
        "exintro": False,
        "explaintext": True,
        "redirects": True,
        "titles": title,
        "format": "json",
    }
    resp = requests.get(
        "https://en.wikipedia.org/w/api.php",
        params=params,
        timeout=30,
        headers={"User-Agent": "RAPTOR-FRAMES-Eval/1.0"},
    )
    resp.raise_for_status()
    pages = resp.json()["query"]["pages"]
    page = next(iter(pages.values()))
    return page.get("extract", "")


def title_from_url(url: str) -> str:
    match = re.search(r"wikipedia\.org/wiki/(.+)$", url)
    if match:
        return match.group(1).replace("_", " ")
    return url.split("/")[-1].replace("_", " ")


def parse_wiki_links(raw) -> list:
    """
    Parse wiki_links regardless of how FRAMES stores it.
    Handles: real list, Python list-literal string, newline-separated string.
    Only returns valid wikipedia.org/wiki/ URLs.
    """
    if raw is None:
        return []

    # Already a real list or tuple
    if isinstance(raw, (list, tuple)):
        candidates = [str(x).strip() for x in raw]

    elif isinstance(raw, str):
        s = raw.strip()
        if s.startswith("["):
            # Python literal: "['https://en.wikipedia.org/wiki/Foo', ...]"
            try:
                parsed = ast.literal_eval(s)
                candidates = [str(x).strip() for x in parsed]
            except Exception:
                candidates = []
        elif "\n" in s:
            candidates = [l.strip() for l in s.splitlines()]
        else:
            candidates = [s]
    else:
        candidates = []

    return [c for c in candidates if c and "wikipedia.org/wiki/" in c]


# ------------------------------------------------------------------------------
# Tree builder  (Memory Cached to bypass RAPTOR disk load bugs)
# ------------------------------------------------------------------------------

TREE_CACHE = {}

def build_or_load_tree(article_title: str, tree_dir: Path, config):
    safe_name = re.sub(r"[^\w\-]", "_", article_title)[:80].strip("_")
    if len(safe_name) < 3:
        log.warning(f"Skipping short/invalid title: '{article_title}'")
        return None

    # Check RAM cache first
    if safe_name in TREE_CACHE:
        log.info(f"Using memory-cached tree for: {safe_name}")
        return TREE_CACHE[safe_name]

    log.info(f"Building tree for: {article_title}")
    text = fetch_wikipedia_text(article_title)
    if not text or not text.strip():
        log.warning(f"Empty Wikipedia article for: {article_title}")
        return None

    # Build fresh tree
    ra = RetrievalAugmentation(config=config)
    ra.add_documents(text)
    
    # Store in memory cache
    TREE_CACHE[safe_name] = ra
    return ra


# ------------------------------------------------------------------------------
# Main runner
# ------------------------------------------------------------------------------

def run(args):
    output_dir  = Path(args.output_dir)
    tree_dir    = output_dir / "trees"
    results_path = output_dir / "raptor_frames_results.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)

    config = RetrievalAugmentationConfig(
        summarization_model=OllamaSummarizer(model=args.llm_model),
        qa_model=OllamaQA(model=args.llm_model),
        embedding_model=OllamaEmbedding(model=args.embed_model),
    )

    # Load dataset
    log.info("Loading google/frames-benchmark ...")
    ds = load_dataset("google/frames-benchmark", split="test")
    log.info(f"Columns: {ds.column_names}")
    log.info(f"First row sample: { {k: str(v)[:120] for k, v in ds[0].items()} }")

    if args.max_samples > 0:
        ds = ds.select(range(min(args.max_samples, len(ds))))
    log.info(f"Running on {len(ds)} samples")

    # Auto-detect field names
    cols = ds.column_names
    question_field = next((c for c in cols if c.lower() in ("prompt", "question")), cols[0])
    answer_field   = next((c for c in cols if c.lower() in ("answer", "gold_answer")), cols[1])
    links_field    = next(
        (c for c in cols if any(k in c.lower() for k in ("wiki", "link", "url", "source"))),
        None
    )
    log.info(f"Using -> question:'{question_field}'  answer:'{answer_field}'  links:'{links_field}'")

    qa_model = OllamaQA(model=args.llm_model)
    results  = []

    for row in tqdm(ds, desc="FRAMES questions"):
        question    = row[question_field]
        gold_answer = row[answer_field]
        raw_links   = row[links_field] if links_field else None
        wiki_links  = parse_wiki_links(raw_links)

        if not wiki_links:
            log.warning(f"No valid wiki links for: {question[:80]}")
            log.warning(f"  raw links value: {repr(raw_links)[:300]}")
            results.append({
                "question":      question,
                "gold_answer":   gold_answer,
                "raptor_answer": "",
                "error":         "no_wiki_links",
            })
            continue

        per_article_answers = []

        for url in wiki_links:
            title = title_from_url(url)
            try:
                ra = build_or_load_tree(title, tree_dir, config)
                if ra is None:
                    continue
                # answer_question is the correct public API on RetrievalAugmentation
                answer = ra.answer_question(question=question)
                per_article_answers.append(answer)
                time.sleep(0.3)
            except Exception as e:
                log.warning(f"Error on '{title}': {e}")
                continue

        if not per_article_answers:
            results.append({
                "question":      question,
                "gold_answer":   gold_answer,
                "raptor_answer": "",
                "error":         "all_trees_failed",
            })
            continue

        # Single article — use its answer directly
        # Multiple articles — do one synthesis pass
        if len(per_article_answers) == 1:
            final_answer = per_article_answers[0]
        else:
            try:
                final_answer = qa_model.answer_question(
                    context="\n\n---\n\n".join(per_article_answers),
                    question=question,
                )
            except Exception as e:
                log.warning(f"Synthesis error: {e}")
                final_answer = per_article_answers[0]

        results.append({
            "question":             question,
            "gold_answer":          gold_answer,
            "raptor_answer":        final_answer,
            "per_article_answers":  per_article_answers,
        })

    # Write results — UTF-8 explicit to handle non-ASCII characters on Windows
    with open(results_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    answered = sum(1 for r in results if r.get("raptor_answer"))
    log.info(f"Done. {len(results)} total | {answered} answered | saved -> {results_path}")

    if args.score:
        score_with_ragas(results, output_dir, args)


# ------------------------------------------------------------------------------
# Ragas scoring
# ------------------------------------------------------------------------------

def score_with_ragas(results: list, output_dir: Path, args):
    try:
        from ragas import evaluate
        from ragas.metrics import answer_correctness, faithfulness, context_precision
        from datasets import Dataset
        from langchain_ollama import OllamaLLM, OllamaEmbeddings
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
    except ImportError as e:
        log.warning(f"Missing dependency for scoring: {e}")
        return

    valid = [r for r in results if r.get("raptor_answer") and r.get("per_article_answers")]
    if not valid:
        log.warning("No valid results to score.")
        return

    llm = LangchainLLMWrapper(OllamaLLM(model=args.llm_model))
    emb = LangchainEmbeddingsWrapper(OllamaEmbeddings(model=args.embed_model))

    result = evaluate(
        Dataset.from_dict({
            "question":     [r["question"]             for r in valid],
            "answer":       [r["raptor_answer"]        for r in valid],
            "contexts":     [r["per_article_answers"]  for r in valid],
            "ground_truth": [r["gold_answer"]          for r in valid],
        }),
        metrics=[answer_correctness, faithfulness, context_precision],
        llm=llm,
        embeddings=emb,
    )

    scores_path = output_dir / "ragas_scores.json"
    with open(scores_path, "w", encoding="utf-8") as f:
        json.dump(dict(result), f, indent=2)
    log.info(f"Ragas scores saved -> {scores_path}")
    for k, v in result.items():
        print(f"  {k}: {v:.4f}")


# ------------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAPTOR on FRAMES with local Ollama")
    parser.add_argument("--llm_model",   default="qwen2.5:7b-instruct")
    parser.add_argument("--embed_model", default="nomic-embed-text")
    parser.add_argument("--max_samples", type=int, default=50)
    parser.add_argument("--output_dir",  default="results/raptor")
    parser.add_argument("--score",       action="store_true")
    run(parser.parse_args())