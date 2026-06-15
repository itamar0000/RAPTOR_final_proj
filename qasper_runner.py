"""
QASPER benchmark runner for RAPTOR (Sarthi et al., ICLR 2024).

Dataset : allenai/qasper — question answering over full-text NLP papers.
Task    : Answer questions about a paper given its full text.
Metric  : token-level Answer F1, max over the reference (annotator) answers —
          the standard QASPER metric (see metrics.max_token_f1).

RAPTOR builds one summary tree per PAPER, then answers all of that paper's
questions from the tree. --max_samples counts PAPERS (each has several questions).

Run examples:
    python qasper_runner.py --retrieval_mode collapsed --max_samples 50 --llm_provider ollama
    python qasper_runner.py --retrieval_mode collapsed --resume --save_summaries
"""

import json
import time
import argparse
import logging
from pathlib import Path

from tqdm import tqdm

from raptor import RetrievalAugmentation, RetrievalAugmentationConfig
from ollama_models import OllamaSummarizer, OllamaQA, OllamaEmbedding
from groq_models import GroqSummarizer, GroqQA
from gemini_models import GeminiSummarizer, GeminiQA
from summary_logger import SummaryLogger
from metrics import max_token_f1
from hf_utils import robust_load, to_list_of_dicts
from concise_qa import make_concise_qa

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DEFAULT_MODELS = {
    "gemini": "gemini-2.5-flash",
    "groq":   "llama-3.3-70b-versatile",
    "ollama": "qwen2.5:14b-instruct",
}


# ─────────────────────────────────────────────────────────────────────────────
# QASPER parsing helpers
# ─────────────────────────────────────────────────────────────────────────────

def qasper_document_text(row) -> str:
    """Concatenate title + abstract + all section paragraphs into one document."""
    parts = []
    if row.get("title"):
        parts.append(str(row["title"]))
    if row.get("abstract"):
        parts.append(str(row["abstract"]))
    ft = row.get("full_text") or {}
    names = ft.get("section_name") or []
    paras = ft.get("paragraphs") or []
    for i, plist in enumerate(paras):
        name = names[i] if i < len(names) else ""
        if name:
            parts.append(str(name))
        for p in (plist or []):
            if p and str(p).strip():
                parts.append(str(p))
    return "\n\n".join(parts)


def qasper_gold_answers(answer_block) -> list:
    """
    Turn one question's annotator answers into a list of gold strings.
    answer_block = qas['answers'][i] == {'answer': [ {answer dict}, ... ]}.
    Unanswerable -> "" (max_token_f1 credits an empty prediction).
    """
    golds = []
    for ann in to_list_of_dicts((answer_block or {}).get("answer")):
        if ann.get("unanswerable"):
            golds.append("")
            continue
        yn = ann.get("yes_no")
        if yn is not None:
            golds.append("Yes" if yn else "No")
            continue
        spans = ann.get("extractive_spans") or []
        if spans:
            golds.append(" ".join(str(s) for s in spans))
            continue
        ffa = ann.get("free_form_answer")
        if ffa:
            golds.append(str(ffa))
            continue
        golds.append("")
    return golds


def iter_questions(row):
    """Yield (question_id, question_text, gold_answers) for a paper row."""
    qas = row.get("qas") or {}
    questions = qas.get("question") or []
    qids = qas.get("question_id") or []
    answers = qas.get("answers") or []
    for i, q in enumerate(questions):
        qid = qids[i] if i < len(qids) else f"q{i}"
        gold = qasper_gold_answers(answers[i]) if i < len(answers) else []
        yield qid, q, gold


# ─────────────────────────────────────────────────────────────────────────────
# Config / output
# ─────────────────────────────────────────────────────────────────────────────

def build_config(args, output_dir):
    model = args.llm_model or DEFAULT_MODELS[args.llm_provider]
    import os
    if args.llm_provider == "gemini":
        api_key = args.gemini_api_key or os.environ.get("GEMINI_API_KEY", "")
        log.info("Summarizer: Gemini (%s) | QA: concise", model)
        summarizer = GeminiSummarizer(api_key=api_key, model=model)
    elif args.llm_provider == "groq":
        api_key = args.groq_api_key or os.environ.get("GROQ_API_KEY", "")
        log.info("Summarizer: Groq (%s) | QA: concise", model)
        summarizer = GroqSummarizer(api_key=api_key, model=model)
    else:
        log.info("Summarizer: Ollama (%s) | QA: concise", model)
        summarizer = OllamaSummarizer(model=model)

    # Free-form QA must be SHORT for token-F1 to be meaningful.
    qa_model = make_concise_qa(args.llm_provider, model, args.groq_api_key, args.gemini_api_key)

    if args.save_summaries:
        summaries_path = args.summaries_file or (
            output_dir / f"raptor_qasper_{args.retrieval_mode}_summaries.jsonl")
        summarizer = SummaryLogger(summarizer, summaries_path, reset=not args.resume)
        log.info("Logging summaries to: %s", summaries_path)

    config = RetrievalAugmentationConfig(
        summarization_model=summarizer,
        qa_model=qa_model,
        embedding_model=OllamaEmbedding(model=args.embed_model),
        tb_max_tokens=args.tb_max_tokens,
    )
    config.retrieval_mode    = args.retrieval_mode
    config.use_reranker      = args.use_reranker
    config.reranker_model    = args.reranker_model
    config.use_late_chunking = False
    return config, summarizer


def _flush(results, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────────────────────────────────────

def run(args):
    output_dir = Path(args.output_dir) / args.llm_provider
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / f"raptor_qasper_{args.retrieval_mode}_results.jsonl"
    log.info("Results -> %s", results_path)

    # Resume: keep successful rows, retry errored ones, skip done question_ids.
    done_qids = set()
    results = []
    if args.resume and results_path.exists():
        loaded = [json.loads(l) for l in results_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        done_qids = {r["question_id"] for r in loaded if "question_id" in r and not r.get("error")}
        results = [r for r in loaded if not r.get("error")]
        log.info("Resume: %d answered questions kept, retrying errored ones", len(done_qids))

    config, summarizer = build_config(args, output_dir)

    log.info("Loading allenai/qasper (split=%s) ...", args.split)
    ds = robust_load("allenai/qasper", args.split, "qasper")
    if args.max_samples > 0:
        ds = ds.select(range(min(args.max_samples, len(ds))))
    log.info("Papers: %d", len(ds))

    f1_sum = sum(r.get("f1", 0.0) for r in results)
    n_scored = len(results)

    for p_idx, row in enumerate(tqdm(ds, desc="QASPER papers")):
        qlist = list(iter_questions(row))
        if qlist and all(qid in done_qids for qid, _, _ in qlist):
            continue  # whole paper already done

        doc_text = qasper_document_text(row)
        if not doc_text.strip():
            continue

        if args.save_summaries:
            summarizer.tag = f"paper{p_idx}"

        # Build the tree ONCE per paper, then answer all its questions.
        try:
            ra = RetrievalAugmentation(config=config)
            ra.add_documents(doc_text)
        except Exception as e:
            log.warning("Paper %d tree build failed: %s", p_idx, e)
            for qid, q, gold in qlist:
                if qid in done_qids:
                    continue
                results.append({"paper_idx": p_idx, "question_id": qid, "question": q,
                                "gold_answers": gold, "raptor_answer": "", "f1": 0.0,
                                "error": "tree_build: " + str(e),
                                "retrieval_mode": args.retrieval_mode})
            _flush(results, results_path)
            continue

        for qid, question, gold in qlist:
            if qid in done_qids:
                continue
            try:
                answer = ra.answer_question(question=question)
                f1 = max_token_f1(answer, gold)
            except Exception as e:
                log.warning("Q %s failed: %s", qid, e)
                results.append({"paper_idx": p_idx, "question_id": qid, "question": question,
                                "gold_answers": gold, "raptor_answer": "", "f1": 0.0,
                                "error": str(e), "retrieval_mode": args.retrieval_mode})
                _flush(results, results_path)
                continue

            f1_sum += f1
            n_scored += 1
            results.append({"paper_idx": p_idx, "question_id": qid, "question": question,
                            "gold_answers": gold, "raptor_answer": answer, "f1": f1,
                            "retrieval_mode": args.retrieval_mode,
                            "use_reranker": args.use_reranker})
            log.info("[paper %d | %s] F1=%.3f | running mean F1=%.3f",
                     p_idx, qid, f1, f1_sum / max(n_scored, 1))
            _flush(results, results_path)
            time.sleep(0.1)

    answered = [r for r in results if not r.get("error")]
    mean_f1 = sum(r.get("f1", 0.0) for r in answered) / max(len(answered), 1)
    log.info("Done. %d questions | mean Answer F1 = %.4f | saved -> %s",
             len(answered), mean_f1, results_path)
    print(f"\n{'='*60}")
    print(f"QASPER Answer F1: {mean_f1:.4f}  over {len(answered)} questions")
    print(f"Paper (RAPTOR + GPT-4) reference Answer F1: ~0.46")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAPTOR on QASPER (Answer F1)")
    parser.add_argument("--llm_model",     default="",
                        help="blank = provider default (gemini-2.5-flash / llama-3.3-70b-versatile / qwen2.5:14b-instruct)")
    parser.add_argument("--embed_model",   default="nomic-embed-text")
    parser.add_argument("--llm_provider",  default="ollama", choices=["ollama", "groq", "gemini"])
    parser.add_argument("--groq_api_key",  default="")
    parser.add_argument("--gemini_api_key", default="")
    parser.add_argument("--tb_max_tokens", type=int, default=500)
    parser.add_argument("--max_samples",   type=int, default=50, help="number of PAPERS (0 = all)")
    parser.add_argument("--split",         default="validation", choices=["train", "validation", "test"])
    parser.add_argument("--output_dir",    default="results/qasper")
    parser.add_argument("--retrieval_mode", default="collapsed",
                        choices=["collapsed", "traversal", "bm25", "hybrid"])
    parser.add_argument("--use_reranker",  action="store_true")
    parser.add_argument("--reranker_model", default="BAAI/bge-reranker-large")
    parser.add_argument("--resume",        action="store_true")
    parser.add_argument("--save_summaries", action="store_true")
    parser.add_argument("--summaries_file", default="")
    run(parser.parse_args())
