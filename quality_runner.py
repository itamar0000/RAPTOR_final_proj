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
from groq_models import GroqSummarizer, GroqQA
from gemini_models import GeminiSummarizer, GeminiQA
from summary_logger import SummaryLogger

# Default LLM model per provider (used when --llm_model is left blank)
DEFAULT_MODELS = {
    "gemini": "gemini-2.5-flash-lite",
    "groq":   "llama-3.3-70b-versatile",
    "ollama": "qwen2.5:14b-instruct",
}

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

    The model states its FINAL choice last (often as \\boxed{X} or "Final
    answer: X"), but its reasoning mentions other options along the way
    ("ruling out option C"). So we look for explicit final-answer markers
    first and take the LAST match; only then fall back to looser heuristics.
    """
    text = text.strip()
    if not text:
        return ""

    # 1. LaTeX \boxed{X} — the model's most common final-answer format
    boxed = re.findall(r"\\boxed\{\s*([A-D])\b", text)
    if boxed:
        return boxed[-1].upper()

    # 2. "final answer ... X" / "the answer is X" / "answer: X" — last occurrence.
    #    Allow a few wrapper chars (": ", "is ", "$\boxed{") between the cue and letter.
    ans = re.findall(
        r"(?:final\s+answer|the\s+answer\s+is|answer)\b[^A-Za-z0-9\n]{0,12}([A-D])\b",
        text, re.IGNORECASE,
    )
    if ans:
        return ans[-1].upper()

    # 3. Response *starts* with a standalone letter (e.g. "B" or "B.")
    if text[0].upper() in OPTION_LETTERS and (len(text) == 1 or not text[1].isalpha()):
        return text[0].upper()

    # 4. Last standalone A-D anywhere (the final choice is usually stated last)
    matches = re.findall(r"\b([A-D])\b", text)
    if matches:
        return matches[-1].upper()
    return ""


def gold_to_letter(raw) -> str:
    """Convert QuALITY gold labels to A/B/C/D.

    emozilla/quality stores `answer` as a 0-indexed int (0=A, 1=B, 2=C, 3=D),
    which matches our prompt's option order (A = first option). A previous
    1-indexed assumption made every gold label off-by-one and turned 0 into "0".
    """
    if isinstance(raw, bool):           # guard: bool is a subclass of int
        return str(raw)
    if isinstance(raw, int) and 0 <= raw <= 3:
        return OPTION_LETTERS[raw]
    if isinstance(raw, str):
        s = raw.strip()
        if s.upper() in OPTION_LETTERS:
            return s.upper()
        if s.isdigit() and 0 <= int(s) <= 3:
            return OPTION_LETTERS[int(s)]
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
    # Separate outputs per provider so gemini/ollama/groq runs never overwrite
    # each other: results/quality/<provider>/...
    output_dir = Path(args.output_dir) / args.llm_provider
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / build_output_filename(args)
    log.info("Results will be saved to: %s", results_path)

    # ── Resume: keep SUCCESSFUL rows, retry rows that errored (e.g. 503) ──────
    done_idx = set()
    existing_results = []
    if args.resume and results_path.exists():
        loaded = []
        with open(results_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    loaded.append(json.loads(line))
        # Only treat error-free rows as done; drop error rows so they re-run.
        done_idx = {r["idx"] for r in loaded if not r.get("error")}
        existing_results = [r for r in loaded if not r.get("error")]
        n_err = len(loaded) - len(existing_results)
        log.info("Resume mode: %d completed rows kept, %d error rows will be retried",
                 len(done_idx), n_err)

    # ── Build RAPTOR config ──────────────────────────────────────────────────
    import os
    model = args.llm_model or DEFAULT_MODELS[args.llm_provider]
    if args.llm_provider == "gemini":
        api_key = args.gemini_api_key or os.environ.get("GEMINI_API_KEY", "")
        log.info("Using Gemini API (model=%s)", model)
        summarizer = GeminiSummarizer(api_key=api_key, model=model)
        qa_model   = GeminiQA(api_key=api_key, model=model)
    elif args.llm_provider == "groq":
        api_key = args.groq_api_key or os.environ.get("GROQ_API_KEY", "")
        log.info("Using Groq API (model=%s)", model)
        summarizer = GroqSummarizer(api_key=api_key, model=model)
        qa_model   = GroqQA(api_key=api_key, model=model)
    else:
        log.info("Using Ollama (model=%s)", model)
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
             n_total, len(done_idx), n_total - len(done_idx))

    # ── Auto-detect column names ─────────────────────────────────────────────
    cols = ds.column_names
    log.info("Columns: %s", cols)
    article_field  = next((c for c in cols if c.lower() in ("article", "document", "context", "text")), cols[0])
    question_field = next((c for c in cols if "question" in c.lower()), None)
    options_field  = next((c for c in cols if c.lower() in ("options", "choices")), None)
    # QuALITY uses writer_label (author's answer) or gold_label — try both
    gold_field = next(
        (c for c in cols if c.lower() in ("gold_label", "writer_label", "turker_label", "gold", "label", "answer", "correct_answer")),
        None,
    )
    log.info("All columns: %s", cols)
    log.info("Using fields -> article:'%s'  question:'%s'  options:'%s'  gold:'%s'",
             article_field, question_field, options_field, gold_field)

    # ── Main loop ─────────────────────────────────────────────────────────────
    results = list(existing_results)  # start from resumed rows
    correct = sum(1 for r in results if r.get("correct"))

    for idx, row in enumerate(tqdm(ds, desc="QuALITY questions", total=n_total)):
        if idx in done_idx:
            continue  # skip rows already completed successfully (resume mode)

        article      = row.get(article_field, "") or ""
        question     = row.get(question_field, "") if question_field else ""
        options      = row.get(options_field, [])  if options_field  else []
        gold_raw = row.get(gold_field) if gold_field else None
        # Some QuALITY rows have None in gold_label but a value in writer_label
        if gold_raw is None:
            for fallback in ("writer_label", "gold_label", "turker_label"):
                if fallback in row and row[fallback] is not None:
                    gold_raw = row[fallback]
                    break
        gold_letter = gold_to_letter(gold_raw)

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
        if args.save_summaries:
            summarizer.tag = f"q{idx}"
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
    parser.add_argument("--llm_model",      default="",
                        help="LLM model name (blank = provider default: "
                             "gemini-2.0-flash / llama-3.3-70b-versatile / qwen2.5:14b-instruct)")
    parser.add_argument("--embed_model",    default="nomic-embed-text",
                        help="Ollama embedding model (always local)")
    parser.add_argument("--llm_provider",   default="gemini", choices=["ollama", "groq", "gemini"],
                        help="LLM backend: gemini (default), groq, or ollama")
    parser.add_argument("--groq_api_key",   default="",
                        help="Groq API key (or set GROQ_API_KEY env var)")
    parser.add_argument("--gemini_api_key", default="",
                        help="Gemini API key (or set GEMINI_API_KEY env var)")
    parser.add_argument("--tb_max_tokens",  type=int, default=500,
                        help="RAPTOR leaf chunk size in tokens. Larger = fewer leaf "
                             "nodes = fewer summary API calls (500 cuts calls ~5x vs 100)")
    parser.add_argument("--max_samples",    type=int, default=100,
                        help="Number of questions to run (0 = full dataset)")
    parser.add_argument("--split",          default="train",
                        choices=["train", "validation", "test"],
                        help="Dataset split to use (validation/test withhold gold labels)")
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
    parser.add_argument("--save_summaries", action="store_true",
                        help="Log every RAPTOR (context -> summary) pair to console "
                             "and a JSONL file, to inspect summary quality")
    parser.add_argument("--summaries_file", default="",
                        help="Where to write summaries (default: alongside results, "
                             "<name>_summaries.jsonl)")

    run(parser.parse_args())
