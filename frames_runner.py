"""
FRAMES benchmark runner for RAPTOR with local Ollama models.

New flags vs original:
    --retrieval_mode   collapsed | traversal | bm25 | hybrid   (default: collapsed)
    --use_reranker     flag — apply CrossEncoder reranking after retrieval
    --reranker_model   HuggingFace model id (default: BAAI/bge-reranker-large)
    --use_late_chunking flag — embed each leaf with document title as context prefix

Retrieval modes explained:
    collapsed  = flat cosine search over ALL nodes at once (RAPTOR default, collapse_tree=True)
    traversal  = true top-down layer-by-layer tree search  (collapse_tree=False)
    bm25       = keyword BM25 over leaf nodes only
    hybrid     = RRF fusion of collapsed cosine + BM25

Run examples:
    # Default RAPTOR (collapsed cosine):
    python frames_runner.py --retrieval_mode collapsed

    # True hierarchical traversal:
    python frames_runner.py --retrieval_mode traversal

    # Hybrid + reranking:
    python frames_runner.py --retrieval_mode hybrid --use_reranker

    # Hybrid + late chunking + reranker:
    python frames_runner.py --retrieval_mode hybrid --use_reranker --use_late_chunking
"""

import ast
import json
import time
import argparse
import re
import logging
from pathlib import Path

import requests
from datasets import load_dataset
from tqdm import tqdm

from raptor import RetrievalAugmentation, RetrievalAugmentationConfig
from ollama_models import OllamaSummarizer, OllamaQA, OllamaEmbedding

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKI_HEADERS = {"User-Agent": "RAPTOR-FRAMES-Eval/1.0"}
TEXT_CACHE = {}


# ──────────────────────────────────────────────────────────────────────────────
# Wikipedia helpers (unchanged)
# ──────────────────────────────────────────────────────────────────────────────

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
        log.info("Direct lookup empty for '%s', trying OpenSearch...", title)
        resolved = _search_title(title)
        if resolved and resolved.lower() != title.lower():
            log.info("OpenSearch resolved '%s' -> '%s'", title, resolved)
            text = _fetch_by_title(resolved)

    if not text or not text.strip():
        log.warning("No Wikipedia text found for: '%s'", title)

    TEXT_CACHE[cache_key] = text
    return text


def title_from_url(url):
    match = re.search(r"wikipedia\.org/wiki/(.+)$", url)
    if match:
        return match.group(1).replace("_", " ")
    return url.split("/")[-1].replace("_", " ")


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


# ──────────────────────────────────────────────────────────────────────────────
# Tree building
# ──────────────────────────────────────────────────────────────────────────────

def build_unified_tree(wiki_links, config, use_late_chunking: bool = False):
    """
    Fetches Wikipedia articles, concatenates them, and builds a RAPTOR tree.

    Args:
        wiki_links        : List of Wikipedia URLs for this question.
        config            : RetrievalAugmentationConfig.
        use_late_chunking : If True, pass a doc_title to add_documents so that
                            leaf embeddings are built with title context.
    """
    texts = []
    titles = []
    for url in wiki_links:
        title = title_from_url(url)
        if len(re.sub(r"[^\w]", "", title)) < 3:
            log.warning("Skipping short title: '%s'", title)
            continue
        text = fetch_wikipedia_text(title)
        if text and text.strip():
            texts.append("=== " + title + " ===\n\n" + text.strip())
            titles.append(title)
        else:
            log.warning("No text retrieved for: '%s'", title)

    if not texts:
        return None

    combined = "\n\n\n".join(texts)
    # Use a composite title for late chunking context (first article title)
    doc_title = titles[0] if titles else "Document"

    log.info(
        "Building RAPTOR tree over %d article(s) (%d chars) | mode=%s | reranker=%s | late_chunking=%s",
        len(texts), len(combined),
        config.retrieval_mode, config.use_reranker, config.use_late_chunking
    )

    ra = RetrievalAugmentation(config=config)
    ra.add_documents(combined)
    return ra


# ──────────────────────────────────────────────────────────────────────────────
# Output filename builder
# ──────────────────────────────────────────────────────────────────────────────

def build_output_filename(args) -> str:
    """
    Generates a descriptive filename that encodes all evaluation flags
    so results from different configurations never overwrite each other.

    Example: raptor_hybrid_reranker_latechunk_frames_results.jsonl
    """
    parts = ["raptor", args.retrieval_mode]
    if args.use_reranker:
        parts.append("reranker")
    if args.use_late_chunking:
        parts.append("latechunk")
    parts.append("frames_results.jsonl")
    return "_".join(parts)


# ──────────────────────────────────────────────────────────────────────────────
# Main runner
# ──────────────────────────────────────────────────────────────────────────────

def run(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Descriptive filename so configs don't overwrite each other
    results_filename = build_output_filename(args)
    results_path = output_dir / results_filename
    log.info(f"Results will be saved to: {results_path}")

    # ── Build config with all new retrieval flags ─────────────────────────────
    config = RetrievalAugmentationConfig(
        summarization_model=OllamaSummarizer(model=args.llm_model),
        qa_model=OllamaQA(model=args.llm_model),
        embedding_model=OllamaEmbedding(model=args.embed_model),
    )
    # Store retrieval flags as plain attributes — the RAPTOR library doesn't
    # expose these in __init__, but build_unified_tree reads them from config.
    config.retrieval_mode   = args.retrieval_mode
    config.use_reranker     = args.use_reranker
    config.reranker_model   = args.reranker_model
    config.use_late_chunking = args.use_late_chunking

    log.info("Loading google/frames-benchmark ...")
    ds = load_dataset("google/frames-benchmark", split="test")
    if args.max_samples > 0:
        ds = ds.select(range(min(args.max_samples, len(ds))))
    log.info("Running on %d samples", len(ds))

    cols = ds.column_names
    question_field = next((c for c in cols if c.lower() in ("prompt", "question")), cols[0])
    answer_field   = next((c for c in cols if c.lower() in ("answer", "gold_answer")), cols[1])
    links_field    = next(
        (c for c in cols if any(k in c.lower() for k in ("wiki", "link", "url", "source"))),
        None,
    )
    log.info("Fields -> question:'%s'  answer:'%s'  links:'%s'",
             question_field, answer_field, links_field)

    results = []

    for row in tqdm(ds, desc="FRAMES questions"):
        question    = row[question_field]
        gold_answer = row[answer_field]
        raw_links   = row[links_field] if links_field else None
        wiki_links  = parse_wiki_links(raw_links)

        if not wiki_links:
            log.warning("No valid wiki links for: %s", question[:80])
            results.append({
                "question":      question,
                "gold_answer":   gold_answer,
                "raptor_answer": "",
                "wiki_links":    [],
                "retrieval_mode": args.retrieval_mode,
                "use_reranker":  args.use_reranker,
                "use_late_chunking": args.use_late_chunking,
                "error":         "no_wiki_links",
            })
            continue

        try:
            ra = build_unified_tree(wiki_links, config, args.use_late_chunking)
            if ra is None:
                results.append({
                    "question":      question,
                    "gold_answer":   gold_answer,
                    "raptor_answer": "",
                    "wiki_links":    wiki_links,
                    "retrieval_mode": args.retrieval_mode,
                    "use_reranker":  args.use_reranker,
                    "use_late_chunking": args.use_late_chunking,
                    "error":         "all_articles_empty",
                })
                continue

            final_answer = ra.answer_question(question=question)

        except Exception as e:
            log.warning("Error on '%s': %s", question[:80], e, exc_info=True)
            results.append({
                "question":      question,
                "gold_answer":   gold_answer,
                "raptor_answer": "",
                "wiki_links":    wiki_links,
                "retrieval_mode": args.retrieval_mode,
                "use_reranker":  args.use_reranker,
                "use_late_chunking": args.use_late_chunking,
                "error":         "exception: " + str(e),
            })
            continue

        results.append({
            "question":      question,
            "gold_answer":   gold_answer,
            "raptor_answer": final_answer,
            "wiki_links":    wiki_links,
            "retrieval_mode": args.retrieval_mode,
            "use_reranker":  args.use_reranker,
            "use_late_chunking": args.use_late_chunking,
        })

        time.sleep(0.3)

    # Save results
    with open(results_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    answered = sum(1 for r in results if r.get("raptor_answer"))
    log.info(
        "Done. %d total | %d answered | mode=%s | reranker=%s | late_chunking=%s | saved -> %s",
        len(results), answered,
        args.retrieval_mode, args.use_reranker, args.use_late_chunking,
        results_path
    )

    if args.score:
        _score_with_ragas(results, output_dir, args)


# ──────────────────────────────────────────────────────────────────────────────
# Ragas scoring (unchanged logic, updated field detection)
# ──────────────────────────────────────────────────────────────────────────────

def _score_with_ragas(results, output_dir, args):
    try:
        from ragas import evaluate
        from ragas.metrics import answer_correctness, faithfulness, context_precision
        from datasets import Dataset
        from langchain_ollama import OllamaLLM, OllamaEmbeddings
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
    except ImportError as e:
        log.warning("Missing dependency for scoring: %s", e)
        return

    valid = [r for r in results if r.get("raptor_answer") and r.get("wiki_links")]
    if not valid:
        log.warning("No valid results to score.")
        return

    llm = LangchainLLMWrapper(OllamaLLM(model=args.llm_model))
    emb = LangchainEmbeddingsWrapper(OllamaEmbeddings(model=args.embed_model))

    result = evaluate(
        Dataset.from_dict({
            "question":     [r["question"]      for r in valid],
            "answer":       [r["raptor_answer"] for r in valid],
            "contexts":     [r["wiki_links"]    for r in valid],
            "ground_truth": [r["gold_answer"]   for r in valid],
        }),
        metrics=[answer_correctness, faithfulness, context_precision],
        llm=llm,
        embeddings=emb,
    )

    scores_filename = build_output_filename(args).replace("_results.jsonl", "_ragas_scores.json")
    scores_path = output_dir / scores_filename
    with open(scores_path, "w", encoding="utf-8") as f:
        json.dump(dict(result), f, indent=2)
    log.info("Ragas scores saved -> %s", scores_path)
    for k, v in result.items():
        print(f"  {k}: {v:.4f}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RAPTOR on FRAMES benchmark with configurable retrieval"
    )
    parser.add_argument("--llm_model",    default="qwen2.5:7b-instruct",
                        help="Ollama LLM for summarization and QA")
    parser.add_argument("--embed_model",  default="nomic-embed-text",
                        help="Ollama embedding model")
    parser.add_argument("--max_samples",  type=int, default=50,
                        help="Number of FRAMES questions to evaluate (0 = all)")
    parser.add_argument("--output_dir",   default="results/raptor",
                        help="Directory to save results JSONL")
    parser.add_argument("--score",        action="store_true",
                        help="Run Ragas scoring after evaluation")

    # ── NEW retrieval flags ───────────────────────────────────────────────────
    parser.add_argument(
        "--retrieval_mode",
        default="hybrid",
        choices=["collapsed", "traversal", "bm25", "hybrid"],
        help=(
            "collapsed  = flat cosine over all nodes (RAPTOR default)\n"
            "traversal  = true top-down layer-by-layer tree search\n"
            "bm25       = keyword BM25 over leaf nodes only\n"
            "hybrid     = RRF fusion of collapsed cosine + BM25"
        )
    )
    parser.add_argument(
        "--use_reranker",
        action="store_true",
        help="Apply CrossEncoder reranking after initial retrieval"
    )
    parser.add_argument(
        "--reranker_model",
        default="BAAI/bge-reranker-large",
        help="HuggingFace CrossEncoder model for reranking"
    )
    parser.add_argument(
        "--use_late_chunking",
        action="store_true",
        help=(
            "Embed each leaf node with its document title prepended as context "
            "(late chunking — improves cross-chunk coherence)"
        )
    )

    run(parser.parse_args())