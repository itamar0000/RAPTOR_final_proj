"""
QuALITY benchmark runner for RAPTOR with local Ollama models.

Dataset  : emozilla/quality (HuggingFace)
Task     : Multiple-choice reading comprehension on long stories
Metric   : Accuracy — fraction of questions where the model picks the right letter (A/B/C/D)
Paper    : RAPTOR (Sarthi et al., ICLR 2024) — GPT-4 baseline: 76.2%

Why QuALITY is better than FRAMES for RAPTOR:
    RAPTOR builds a hierarchical summary tree over a SINGLE long document.
    QuALITY provides one article per question (~5 000 tokens average) — exactly
    the use case RAPTOR was designed for.  FRAMES requires multi-hop across
    multiple Wikipedia articles, which is much harder.

Run examples:
    # 100 samples, collapsed cosine (RAPTOR default):
    python quality_runner.py

    # Full validation set with 14B model:
    python quality_runner.py --llm_model qwen2.5:14b-instruct --max_samples 0

    # Hybrid retrieval, 200 samples:
    python quality_runner.py --retrieval_mode hybrid --max_samples 200

    # Resume after a crash (skips already-saved rows):
    python quality_runner.py --resume
"""

import json
import time
import argparse
import re
import logging
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

from raptor import RetrievalAugmentation, RetrievalAugmentationConfig
from ollama_models import OllamaSummarizer, OllamaQA, OllamaEmbedding

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

OPTION_LETTERS = ["A", "B", "C", "D"]


# ─────────────────────────────────────────────────────────────────────────────
# Multiple-choice helpers
# ─────────────────────────────────────────────────────────────────────────────

def format_mc_question(question: str, options: list) -> str:
    """Build the prompt string fed into RAPTOR's QA model."""
    opts = "\n".join(f"{OPTION_LETTERS[i]}. {opt}" for i, opt in enumerate(options))
    return (
        f"{question}\n\n"
        f"Options:\n{opts}\n\n"
        "Read the context carefully, reason step-by-step, then give your final answer "
        "as a single letter: A, B, C, or D."
    )


def extract_letter(text: str) -> str:
    """
    Pull the predicted A/B/C/D out of a free-form model response.
    Falls back through three strategies in order of confidence.
    """
    text = text.strip()
    # Strategy 1: response starts with the letter (most reliable)
    if text and text[0].upper() in OPTION_LETTERS:
        return text[0].upper()
    # Strategy 2: explicit "Answer: X" / "The answer is X" pattern
    m = re.search(r"\b(?:answer(?:\s+is)?|option|choice)[:\s]+([A-D])\b", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # Strategy 3: any standalone letter near end of response
    matches = re.findall(r"\b([A-D])\b", text)
    if matches:
        return matches[-1].upper()
    return ""


def gold_to_letter(raw) -> str:
    """Convert QuALITY gold labels (1-indexed int or letter string) to A/B/C/D."""
    if isinstance(raw, int) and 1 <= raw <= 4:
        return OPTION_LETTERS[raw - 1]
    if isinstance(raw, str) and raw.strip().upper() in OPTION_LETTERS:
        return raw.strip().upper()
    return str(raw)


# ─────────────────────────────────────────────────────────────────────────────
# Filename builder
# ─────────────────────────────────────────────────────────────────────────────

def build_output_filename(args) -> str:
    parts = ["raptor_quality", args.retrieval_mode]
    if args.use_reranker:
        parts.append("reranker")
    parts.append("results.jsonl")
    return "_".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────────────────────────────────────

def run(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / build_output_filename(args)
    log.info("Results will be saved to: %s", results_path)

    # ── Resume: load already-completed rows ──────────────────────────────────
    already_done = 0
    existing_results = []
    if args.resume and results_path.exists():
        with open(results_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    existing_results.append(json.loads(line))
        already_done = len(existing_results)
        log.info("Resume mode: skipping first %d already-completed rows", already_done)

    # ── Build RAPTOR config ──────────────────────────────────────────────────
    config = RetrievalAugmentationConfig(
        summarization_model=OllamaSummarizer(model=args.llm_model),
        qa_model=OllamaQA(model=args.llm_model),
        embedding_model=OllamaEmbedding(model=args.embed_model),
    )
    config.retrieval_mode    = args.retrieval_mode
    config.use_reranker      = args.use_reranker
    config.reranker_model    = args.reranker_model
    config.use_late_chunking = False

    # ── Load dataset ─────────────────────────────────────────────────────────
    log.info("Loading emozilla/quality (split=%s) ...", args.split)
    ds = load_dataset("emozilla/quality", split=args.split)

    n_total = len(ds)
    if args.max_samples > 0:
        n_total = min(args.max_samples, n_total)
        ds = ds.select(range(n_total))
    log.info("Dataset size: %d  |  Already done: %d  |  Remaining: %d",
             n_total, already_done, n_total - already_done)

    # ── Auto-detect column names ─────────────────────────────────────────────
    cols = ds.column_names
    log.info("Columns: %s", cols)
    article_field  = next((c for c in cols if c.lower() in ("article", "document", "context", "text")), cols[0])
    question_field = next((c for c in cols if "question" in c.lower()), None)
    options_field  = next((c for c in cols if c.lower() in ("options", "choices")), None)
    gold_field     = next((c for c in cols if "gold" in c.lower() or "label" in c.lower()), None)
    log.info("Using fields -> article:'%s'  question:'%s'  options:'%s'  gold:'%s'",
             article_field, question_field, options_field, gold_field)

    # ── Main loop ─────────────────────────────────────────────────────────────
    results = list(existing_results)  # start from resumed rows
    correct = sum(1 for r in results if r.get("correct"))

    for idx, row in enumerate(tqdm(ds, desc="QuALITY questions", total=n_total)):
        if idx < already_done:
            continue  # skip completed rows in resume mode

        article      = row.get(article_field, "") or ""
        question     = row.get(question_field, "") if question_field else ""
        options      = row.get(options_field, [])  if options_field  else []
        gold_raw     = row.get(gold_field)          if gold_field     else None
        gold_letter  = gold_to_letter(gold_raw)

        # ── Empty article ────────────────────────────────────────────────────
        if not article.strip():
            log.warning("Empty article at row %d, skipping.", idx)
            results.append({
                "idx": idx, "question": question, "gold_letter": gold_letter,
                "options": options, "predicted_letter": "", "raw_answer": "",
                "correct": False, "error": "empty_article",
                "retrieval_mode": args.retrieval_mode,
            })
            _flush(results, results_path)
            continue

        # ── Build RAPTOR tree + answer ────────────────────────────────────────
        mc_question = format_mc_question(question, options)
        log.info("[%d/%d] Article: %d chars | Building RAPTOR tree ...",
                 idx + 1, n_total, len(article))
        try:
            ra = RetrievalAugmentation(config=config)
            ra.add_documents(article)
            raw_answer = ra.answer_question(question=mc_question)
            predicted_letter = extract_letter(raw_answer)
            is_correct = (predicted_letter == gold_letter)

        except Exception as e:
            log.warning("Error on row %d '%s': %s", idx, question[:60], e, exc_info=True)
            results.append({
                "idx": idx, "question": question, "gold_letter": gold_letter,
                "options": options, "predicted_letter": "", "raw_answer": "",
                "correct": False, "error": str(e),
                "retrieval_mode": args.retrieval_mode,
            })
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
            "retrieval_mode":   args.retrieval_mode,
            "use_reranker":     args.use_reranker,
        })

        running_acc = correct / len(results)
        log.info("[%d/%d] Gold=%s Pred=%s %s | Running accuracy: %.1f%%",
                 len(results), n_total, gold_letter, predicted_letter,
                 "CORRECT" if is_correct else "WRONG", running_acc * 100)

        _flush(results, results_path)
        time.sleep(0.3)

    # ── Final score ───────────────────────────────────────────────────────────
    total_answered = len(results)
    accuracy = correct / total_answered if total_answered else 0.0
    log.info(
        "Done. %d total | %d correct | accuracy=%.1f%% | mode=%s | saved -> %s",
        total_answered, correct, accuracy * 100, args.retrieval_mode, results_path,
    )
    print(f"\n{'='*60}")
    print(f"QuALITY Accuracy: {correct}/{total_answered} = {accuracy:.1%}")
    print(f"Paper (GPT-4 + RAPTOR collapsed):  76.2%")
    print(f"{'='*60}")


def _flush(results, path):
    """Write all results to disk after every sample — safe against crashes."""
    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RAPTOR on QuALITY multiple-choice benchmark"
    )
    parser.add_argument("--llm_model",      default="qwen2.5:7b-instruct",
                        help="Ollama LLM for summarization and QA")
    parser.add_argument("--embed_model",    default="nomic-embed-text",
                        help="Ollama embedding model")
    parser.add_argument("--max_samples",    type=int, default=100,
                        help="Number of questions to run (0 = full dataset)")
    parser.add_argument("--split",          default="validation",
                        choices=["train", "validation", "test"],
                        help="Dataset split to use")
    parser.add_argument("--output_dir",     default="results/quality",
                        help="Directory to save results JSONL")
    parser.add_argument("--retrieval_mode", default="collapsed",
                        choices=["collapsed", "traversal", "bm25", "hybrid"],
                        help="collapsed = RAPTOR default flat cosine search")
    parser.add_argument("--use_reranker",   action="store_true",
                        help="Apply CrossEncoder reranking after retrieval")
    parser.add_argument("--reranker_model", default="BAAI/bge-reranker-large")
    parser.add_argument("--resume",         action="store_true",
                        help="Skip rows already saved in the output file")

    run(parser.parse_args())
