"""
NarrativeQA baseline — plain chunking + cosine retrieval, NO RAPTOR tree.

Same reader + embeddings + document source as narrativeqa_runner, so the only
difference is the absence of the summary tree. Metrics: BLEU-1/4, ROUGE-L, METEOR.

Run:
    python narrativeqa_baseline.py --max_samples 100 --llm_provider ollama
"""

import json
import time
import argparse
import logging
from pathlib import Path

from tqdm import tqdm

from ollama_models import OllamaQA, OllamaEmbedding
from groq_models import GroqQA
from gemini_models import GeminiQA
from metrics import narrativeqa_metrics
from hf_utils import robust_load
from quality_baseline import chunk_text, embed, cosine_similarity
from narrativeqa_runner import doc_id_of, doc_text_of, refs_of, _means

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DEFAULT_MODELS = {
    "gemini": "gemini-2.5-flash",
    "groq":   "llama-3.3-70b-versatile",
    "ollama": "qwen2.5:14b-instruct",
}


def make_qa(args):
    import os
    model = args.llm_model or DEFAULT_MODELS[args.llm_provider]
    if args.llm_provider == "gemini":
        return GeminiQA(api_key=args.gemini_api_key or os.environ.get("GEMINI_API_KEY", ""), model=model), model
    if args.llm_provider == "groq":
        return GroqQA(api_key=args.groq_api_key or os.environ.get("GROQ_API_KEY", ""), model=model), model
    return OllamaQA(model=model), model


def _flush(results, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def run(args):
    output_dir = Path(args.output_dir) / args.llm_provider
    output_dir.mkdir(parents=True, exist_ok=True)
    src = "fulltext" if args.use_full_text else "summary"
    results_path = output_dir / f"narrativeqa_baseline_top{args.top_k}_{src}_results.jsonl"
    log.info("Results -> %s", results_path)

    done_idx = set()
    results = []
    if args.resume and results_path.exists():
        loaded = [json.loads(l) for l in results_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        done_idx = {r["idx"] for r in loaded if "idx" in r and not r.get("error")}
        results = [r for r in loaded if not r.get("error")]
        log.info("Resume: %d answered questions kept", len(done_idx))

    qa_model, model = make_qa(args)
    log.info("Baseline reader: provider=%s model=%s | source=%s", args.llm_provider, model, src)

    log.info("Loading deepmind/narrativeqa (split=%s) ...", args.split)
    ds = robust_load("deepmind/narrativeqa", args.split, "default")
    if args.max_samples > 0:
        ds = ds.select(range(min(args.max_samples, len(ds))))
    log.info("Questions: %d", len(ds))

    prev_doc, chunks, chunk_emb = None, [], []
    for idx, row in enumerate(tqdm(ds, desc="NarrativeQA baseline")):
        if idx in done_idx:
            continue
        question = (row.get("question") or {}).get("text", "")
        refs = refs_of(row)
        did = doc_id_of(row)

        if did != prev_doc:
            text = doc_text_of(row, args.use_full_text)
            if not text.strip():
                results.append({"idx": idx, "doc_id": did, "question": question,
                                "references": refs, "baseline_answer": "",
                                "bleu1": 0, "bleu4": 0, "rouge_l": 0, "meteor": 0,
                                "error": "empty_document", "method": "baseline_cosine"})
                _flush(results, results_path)
                continue
            chunks = chunk_text(text, chunk_size=args.chunk_size, overlap=args.overlap)
            chunk_emb = [embed(c, args.embed_model) for c in chunks]
            prev_doc = did

        try:
            q_emb = embed(question, args.embed_model)
            scores = [cosine_similarity(q_emb, ce) for ce in chunk_emb]
            top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:args.top_k]
            context = "\n\n---\n\n".join(chunks[i] for i in sorted(top))
            answer = qa_model.answer_question(context, question)
            m = narrativeqa_metrics(answer, refs)
        except Exception as e:
            log.warning("Q %d failed: %s", idx, e)
            results.append({"idx": idx, "doc_id": did, "question": question,
                            "references": refs, "baseline_answer": "",
                            "bleu1": 0, "bleu4": 0, "rouge_l": 0, "meteor": 0,
                            "error": str(e), "method": "baseline_cosine"})
            _flush(results, results_path)
            continue

        results.append({"idx": idx, "doc_id": did, "question": question,
                        "references": refs, "baseline_answer": answer,
                        **m, "top_k": args.top_k, "method": "baseline_cosine"})
        mean = _means(results)
        log.info("[%d] BLEU1=%.3f ROUGE-L=%.3f | mean R-L=%.3f",
                 idx, m["bleu1"], m["rouge_l"], mean["rouge_l"])
        _flush(results, results_path)
        time.sleep(0.1)

    mean = _means(results)
    answered = len([r for r in results if not r.get("error")])
    print(f"\n{'='*60}")
    print(f"NarrativeQA baseline (no RAPTOR, {src})  over {answered} questions")
    print(f"  BLEU-1 : {mean['bleu1']:.4f}")
    print(f"  BLEU-4 : {mean['bleu4']:.4f}")
    print(f"  ROUGE-L: {mean['rouge_l']:.4f}")
    print(f"  METEOR : {mean['meteor']:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NarrativeQA baseline (no RAPTOR)")
    parser.add_argument("--llm_model",     default="", help="blank = provider default")
    parser.add_argument("--embed_model",   default="nomic-embed-text")
    parser.add_argument("--llm_provider",  default="ollama", choices=["ollama", "groq", "gemini"])
    parser.add_argument("--groq_api_key",  default="")
    parser.add_argument("--gemini_api_key", default="")
    parser.add_argument("--max_samples",   type=int, default=100, help="number of QUESTIONS (0 = all)")
    parser.add_argument("--split",         default="validation", choices=["train", "validation", "test"])
    parser.add_argument("--output_dir",    default="results/narrativeqa")
    parser.add_argument("--chunk_size",    type=int, default=200)
    parser.add_argument("--overlap",       type=int, default=50)
    parser.add_argument("--top_k",         type=int, default=5)
    parser.add_argument("--use_full_text", action="store_true",
                        help="retrieve from the FULL story instead of the summary (slow)")
    parser.add_argument("--resume",        action="store_true")
    run(parser.parse_args())
