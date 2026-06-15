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
from groq_models import GroqSummarizer, GroqQA
from gemini_models import GeminiSummarizer, GeminiQA
from summary_logger import SummaryLogger

# Default LLM model per provider (used when --llm_model is left blank)
DEFAULT_MODELS = {
    "gemini": "gemini-2.5-flash",
    "groq":   "llama-3.3-70b-versatile",
    "ollama": "qwen2.5:14b-instruct",
}

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
    from urllib.parse import unquote
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
    # Summarization AND QA use the same provider. Gemini's generous free tier
    # (1M TPM, ~1500 req/day) handles the many summary calls per article that
    # exhaust Groq's 131K-tokens/day cap. Local Ollama 8B summaries are poor.
    import os
    model = args.llm_model or DEFAULT_MODELS[args.llm_provider]
    if args.llm_provider == "gemini":
        api_key = args.gemini_api_key or os.environ.get("GEMINI_API_KEY", "")
        log.info("Summarizer + QA: Gemini (%s)", model)
        summarizer = GeminiSummarizer(api_key=api_key, model=model)
        qa_model   = GeminiQA(api_key=api_key, model=model)
    elif args.llm_provider == "groq":
        api_key = args.groq_api_key or os.environ.get("GROQ_API_KEY", "")
        log.info("Summarizer + QA: Groq (%s)", model)
        summarizer = GroqSummarizer(api_key=api_key, model=model)
        qa_model   = GroqQA(api_key=api_key, model=model)
    else:
        log.info("Summarizer + QA: Ollama (%s)", model)
        summarizer = OllamaSummarizer(model=model)
        qa_model   = OllamaQA(model=model)

    # Optionally wrap the summarizer to log every (context -> summary) pair,
    # so you can inspect whether bad answers stem from bad summaries.
    if args.save_summaries:
        summaries_path = args.summaries_file or (
            output_dir / build_output_filename(args).replace("results.jsonl", "summaries.jsonl")
        )
        summarizer = SummaryLogger(summarizer, summaries_path, reset=not args.resume)
        log.info("Logging summaries to: %s", summaries_path)

    config = RetrievalAugmentationConfig(
        summarization_model=summarizer,
        qa_model=qa_model,
        embedding_model=OllamaEmbedding(model=args.embed_model),
        tb_max_tokens=args.tb_max_tokens,
    )
    log.info("RAPTOR leaf chunk size: tb_max_tokens=%d", args.tb_max_tokens)
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

    # ── Resume: load already-completed rows, skip them below ──────────────────
    already_done = 0
    results = []
    if args.resume and results_path.exists():
        with open(results_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    results.append(json.loads(line))
        already_done = len(results)
        log.info("Resume mode: skipping first %d already-completed rows", already_done)

    for idx, row in enumerate(tqdm(ds, desc="FRAMES questions")):
        if idx < already_done:
            continue  # already processed in a previous run

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
            _flush(results, results_path)
            continue

        if args.save_summaries:
            summarizer.tag = f"frames{idx}"
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
                _flush(results, results_path)
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
            _flush(results, results_path)
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
        _flush(results, results_path)

        time.sleep(0.3)

    # Final results are already on disk via _flush after every sample.
    answered = sum(1 for r in results if r.get("raptor_answer"))
    log.info(
        "Done. %d total | %d answered | mode=%s | reranker=%s | late_chunking=%s | saved -> %s",
        len(results), answered,
        args.retrieval_mode, args.use_reranker, args.use_late_chunking,
        results_path
    )

    if args.score:
        _score_with_ragas(results, output_dir, args)


def _flush(results, path):
    """Write all results to disk after every sample — safe against stop/crash."""
    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


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
    parser.add_argument("--llm_model",    default="",
                        help="LLM model name (blank = provider default: "
                             "gemini-2.0-flash / llama-3.3-70b-versatile / qwen2.5:14b-instruct)")
    parser.add_argument("--embed_model",  default="nomic-embed-text",
                        help="Ollama embedding model (always local)")
    parser.add_argument("--llm_provider", default="gemini", choices=["ollama", "groq", "gemini"],
                        help="LLM backend for summarization AND QA: gemini (default), groq, or ollama")
    parser.add_argument("--groq_api_key", default="",
                        help="Groq API key (or set GROQ_API_KEY env var)")
    parser.add_argument("--gemini_api_key", default="",
                        help="Gemini API key (or set GEMINI_API_KEY env var)")
    parser.add_argument("--tb_max_tokens", type=int, default=500,
                        help="RAPTOR leaf chunk size in tokens. Larger = fewer leaf "
                             "nodes = fewer summary API calls (500 cuts calls ~5x vs 100)")
    parser.add_argument("--max_samples",  type=int, default=50,
                        help="Number of FRAMES questions to evaluate (0 = all)")
    parser.add_argument("--output_dir",   default="results/raptor",
                        help="Directory to save results JSONL")
    parser.add_argument("--score",        action="store_true",
                        help="Run Ragas scoring after evaluation")
    parser.add_argument("--resume",       action="store_true",
                        help="Skip rows already saved in the output file (continue a stopped run)")
    parser.add_argument("--save_summaries", action="store_true",
                        help="Log every RAPTOR (context -> summary) pair to console "
                             "and a JSONL file, to inspect summary quality")
    parser.add_argument("--summaries_file", default="",
                        help="Where to write summaries (default: alongside results, "
                             "<name>_summaries.jsonl)")

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