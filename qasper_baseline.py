"""
QASPER baseline — plain chunking + cosine retrieval, NO RAPTOR tree.

Same reader + embeddings as qasper_runner, so the only difference is the absence
of the summary tree. Metric: token-level Answer F1 (metrics.max_token_f1).

Run:
    python qasper_baseline.py --max_samples 50 --llm_provider ollama
"""

import json
import time
import argparse
import logging
from pathlib import Path

from tqdm import tqdm

from metrics import max_token_f1
from hf_utils import robust_load
from concise_qa import make_concise_qa
# reuse the chunking / embedding helpers from the QuALITY baseline
from quality_baseline import chunk_text, embed, cosine_similarity
from qasper_runner import qasper_document_text, iter_questions

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DEFAULT_MODELS = {
    "gemini": "gemini-2.5-flash",
    "groq":   "llama-3.3-70b-versatile",
    "ollama": "qwen2.5:14b-instruct",
}


def make_qa(args):
    model = args.llm_model or DEFAULT_MODELS[args.llm_provider]
    qa = make_concise_qa(args.llm_provider, model, args.groq_api_key, args.gemini_api_key)
    return qa, model


def _flush(results, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def run(args):
    output_dir = Path(args.output_dir) / args.llm_provider
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / f"qasper_baseline_top{args.top_k}_results.jsonl"
    log.info("Results -> %s", results_path)

    done_qids = set()
    results = []
    if args.resume and results_path.exists():
        loaded = [json.loads(l) for l in results_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        done_qids = {r["question_id"] for r in loaded if "question_id" in r and not r.get("error")}
        results = [r for r in loaded if not r.get("error")]
        log.info("Resume: %d answered questions kept", len(done_qids))

    qa_model, model = make_qa(args)
    log.info("Baseline reader: provider=%s model=%s", args.llm_provider, model)

    log.info("Loading allenai/qasper (split=%s) ...", args.split)
    ds = robust_load("allenai/qasper", args.split, "qasper")
    if args.max_samples > 0:
        ds = ds.select(range(min(args.max_samples, len(ds))))
    log.info("Papers: %d", len(ds))

    for p_idx, row in enumerate(tqdm(ds, desc="QASPER baseline")):
        qlist = list(iter_questions(row))
        if qlist and all(qid in done_qids for qid, _, _ in qlist):
            continue
        doc_text = qasper_document_text(row)
        if not doc_text.strip():
            continue

        # Chunk + embed the whole paper once, reuse for its questions.
        chunks = chunk_text(doc_text, chunk_size=args.chunk_size, overlap=args.overlap)
        chunk_emb = [embed(c, args.embed_model) for c in chunks]

        for qid, question, gold in qlist:
            if qid in done_qids:
                continue
            try:
                q_emb = embed(question, args.embed_model)
                scores = [cosine_similarity(q_emb, ce) for ce in chunk_emb]
                top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:args.top_k]
                context = "\n\n---\n\n".join(chunks[i] for i in sorted(top))
                answer = qa_model.answer_question(context, question)
                f1 = max_token_f1(answer, gold)
            except Exception as e:
                log.warning("Q %s failed: %s", qid, e)
                results.append({"paper_idx": p_idx, "question_id": qid, "question": question,
                                "gold_answers": gold, "baseline_answer": "", "f1": 0.0,
                                "error": str(e), "method": "baseline_cosine"})
                _flush(results, results_path)
                continue

            results.append({"paper_idx": p_idx, "question_id": qid, "question": question,
                            "gold_answers": gold, "baseline_answer": answer, "f1": f1,
                            "top_k": args.top_k, "method": "baseline_cosine"})
            mean = sum(r.get("f1", 0.0) for r in results if not r.get("error")) / \
                   max(len([r for r in results if not r.get("error")]), 1)
            log.info("[paper %d | %s] F1=%.3f | running mean F1=%.3f", p_idx, qid, f1, mean)
            _flush(results, results_path)
            time.sleep(0.1)

    answered = [r for r in results if not r.get("error")]
    mean_f1 = sum(r.get("f1", 0.0) for r in answered) / max(len(answered), 1)
    log.info("Done. %d questions | mean Answer F1 = %.4f", len(answered), mean_f1)
    print(f"\n{'='*60}")
    print(f"QASPER baseline (no RAPTOR) Answer F1: {mean_f1:.4f} over {len(answered)} questions")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QASPER baseline (no RAPTOR), Answer F1")
    parser.add_argument("--llm_model",     default="",
                        help="blank = provider default")
    parser.add_argument("--embed_model",   default="nomic-embed-text")
    parser.add_argument("--llm_provider",  default="ollama", choices=["ollama", "groq", "gemini"])
    parser.add_argument("--groq_api_key",  default="")
    parser.add_argument("--gemini_api_key", default="")
    parser.add_argument("--max_samples",   type=int, default=50, help="number of PAPERS (0 = all)")
    parser.add_argument("--split",         default="validation", choices=["train", "validation", "test"])
    parser.add_argument("--output_dir",    default="results/qasper")
    parser.add_argument("--chunk_size",    type=int, default=200, help="chunk size in words")
    parser.add_argument("--overlap",       type=int, default=50)
    parser.add_argument("--top_k",         type=int, default=5)
    parser.add_argument("--resume",        action="store_true")
    run(parser.parse_args())
