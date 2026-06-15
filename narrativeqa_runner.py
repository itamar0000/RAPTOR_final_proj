"""
NarrativeQA benchmark runner for RAPTOR (Sarthi et al., ICLR 2024).

Dataset : deepmind/narrativeqa — QA over books / movie scripts.
Task    : Free-form answer generation; each question has 2 reference answers.
Metrics : BLEU-1, BLEU-4, ROUGE-L, METEOR (max over references) — the paper's
          NarrativeQA metrics (see metrics.narrativeqa_metrics).

Document source:
    By default RAPTOR builds its tree over the dataset's human-written SUMMARY
    (clean, ~1 page, fast). Pass --use_full_text to build over the FULL story
    (faithful to the paper, but each doc is a whole book -> very slow + messy
    HTML). One tree is built per document and reused across its questions.

--max_samples counts QUESTIONS (the dataset is one row per question).

Run examples:
    python narrativeqa_runner.py --retrieval_mode collapsed --max_samples 100 --llm_provider ollama
    python narrativeqa_runner.py --retrieval_mode collapsed --resume --save_summaries
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
from metrics import narrativeqa_metrics
from hf_utils import robust_load, to_list_of_dicts

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DEFAULT_MODELS = {
    "gemini": "gemini-2.5-flash",
    "groq":   "llama-3.3-70b-versatile",
    "ollama": "qwen2.5:14b-instruct",
}


def doc_id_of(row) -> str:
    return (row.get("document") or {}).get("id", "")


def doc_text_of(row, use_full_text: bool) -> str:
    doc = row.get("document") or {}
    if use_full_text:
        return doc.get("text", "") or ""
    return (doc.get("summary") or {}).get("text", "") or ""


def refs_of(row) -> list:
    return [d.get("text", "") for d in to_list_of_dicts(row.get("answers")) if d.get("text")]


def build_config(args, output_dir):
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

    if args.save_summaries:
        summaries_path = args.summaries_file or (
            output_dir / f"raptor_narrativeqa_{args.retrieval_mode}_summaries.jsonl")
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


def _means(rows):
    rows = [r for r in rows if not r.get("error")]
    if not rows:
        return {"bleu1": 0, "bleu4": 0, "rouge_l": 0, "meteor": 0}
    keys = ("bleu1", "bleu4", "rouge_l", "meteor")
    return {k: sum(r.get(k, 0.0) for r in rows) / len(rows) for k in keys}


def run(args):
    output_dir = Path(args.output_dir) / args.llm_provider
    output_dir.mkdir(parents=True, exist_ok=True)
    src = "fulltext" if args.use_full_text else "summary"
    results_path = output_dir / f"raptor_narrativeqa_{args.retrieval_mode}_{src}_results.jsonl"
    log.info("Results -> %s", results_path)

    done_idx = set()
    results = []
    if args.resume and results_path.exists():
        loaded = [json.loads(l) for l in results_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        done_idx = {r["idx"] for r in loaded if "idx" in r and not r.get("error")}
        results = [r for r in loaded if not r.get("error")]
        log.info("Resume: %d answered questions kept", len(done_idx))

    config, summarizer = build_config(args, output_dir)

    log.info("Loading deepmind/narrativeqa (split=%s) ...", args.split)
    ds = robust_load("deepmind/narrativeqa", args.split, "default")
    if args.max_samples > 0:
        ds = ds.select(range(min(args.max_samples, len(ds))))
    log.info("Questions: %d | document source: %s", len(ds), src)

    prev_doc, ra = None, None
    for idx, row in enumerate(tqdm(ds, desc="NarrativeQA")):
        if idx in done_idx:
            continue
        question = (row.get("question") or {}).get("text", "")
        refs = refs_of(row)
        did = doc_id_of(row)

        # (Re)build the tree only when the document changes.
        if did != prev_doc:
            text = doc_text_of(row, args.use_full_text)
            if not text.strip():
                results.append({"idx": idx, "doc_id": did, "question": question,
                                "references": refs, "raptor_answer": "",
                                "bleu1": 0, "bleu4": 0, "rouge_l": 0, "meteor": 0,
                                "error": "empty_document", "retrieval_mode": args.retrieval_mode})
                _flush(results, results_path)
                continue
            if args.save_summaries:
                summarizer.tag = f"doc:{did[:8]}"
            try:
                ra = RetrievalAugmentation(config=config)
                ra.add_documents(text)
                prev_doc = did
            except Exception as e:
                log.warning("Doc %s tree build failed: %s", did, e)
                ra, prev_doc = None, None
                results.append({"idx": idx, "doc_id": did, "question": question,
                                "references": refs, "raptor_answer": "",
                                "bleu1": 0, "bleu4": 0, "rouge_l": 0, "meteor": 0,
                                "error": "tree_build: " + str(e), "retrieval_mode": args.retrieval_mode})
                _flush(results, results_path)
                continue

        try:
            answer = ra.answer_question(question=question)
            m = narrativeqa_metrics(answer, refs)
        except Exception as e:
            log.warning("Q %d failed: %s", idx, e)
            results.append({"idx": idx, "doc_id": did, "question": question,
                            "references": refs, "raptor_answer": "",
                            "bleu1": 0, "bleu4": 0, "rouge_l": 0, "meteor": 0,
                            "error": str(e), "retrieval_mode": args.retrieval_mode})
            _flush(results, results_path)
            continue

        results.append({"idx": idx, "doc_id": did, "question": question,
                        "references": refs, "raptor_answer": answer,
                        **m, "retrieval_mode": args.retrieval_mode,
                        "use_reranker": args.use_reranker})
        mean = _means(results)
        log.info("[%d] BLEU1=%.3f BLEU4=%.3f ROUGE-L=%.3f METEOR=%.3f | mean R-L=%.3f",
                 idx, m["bleu1"], m["bleu4"], m["rouge_l"], m["meteor"], mean["rouge_l"])
        _flush(results, results_path)
        time.sleep(0.1)

    mean = _means(results)
    answered = len([r for r in results if not r.get("error")])
    log.info("Done. %d questions | BLEU1=%.4f BLEU4=%.4f ROUGE-L=%.4f METEOR=%.4f",
             answered, mean["bleu1"], mean["bleu4"], mean["rouge_l"], mean["meteor"])
    print(f"\n{'='*60}")
    print(f"NarrativeQA ({src})  over {answered} questions")
    print(f"  BLEU-1 : {mean['bleu1']:.4f}")
    print(f"  BLEU-4 : {mean['bleu4']:.4f}")
    print(f"  ROUGE-L: {mean['rouge_l']:.4f}")
    print(f"  METEOR : {mean['meteor']:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAPTOR on NarrativeQA (BLEU/ROUGE-L/METEOR)")
    parser.add_argument("--llm_model",     default="",
                        help="blank = provider default (gemini-2.5-flash / llama-3.3-70b-versatile / qwen2.5:14b-instruct)")
    parser.add_argument("--embed_model",   default="nomic-embed-text")
    parser.add_argument("--llm_provider",  default="ollama", choices=["ollama", "groq", "gemini"])
    parser.add_argument("--groq_api_key",  default="")
    parser.add_argument("--gemini_api_key", default="")
    parser.add_argument("--tb_max_tokens", type=int, default=500)
    parser.add_argument("--max_samples",   type=int, default=100, help="number of QUESTIONS (0 = all)")
    parser.add_argument("--split",         default="validation", choices=["train", "validation", "test"])
    parser.add_argument("--output_dir",    default="results/narrativeqa")
    parser.add_argument("--retrieval_mode", default="collapsed",
                        choices=["collapsed", "traversal", "bm25", "hybrid"])
    parser.add_argument("--use_reranker",  action="store_true")
    parser.add_argument("--reranker_model", default="BAAI/bge-reranker-large")
    parser.add_argument("--use_full_text", action="store_true",
                        help="build the tree over the FULL story instead of the summary (slow)")
    parser.add_argument("--resume",        action="store_true")
    parser.add_argument("--save_summaries", action="store_true")
    parser.add_argument("--summaries_file", default="")
    run(parser.parse_args())
