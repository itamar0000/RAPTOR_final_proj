"""
Standalone Ragas scorer — reads any results JSONL and scores with local Ollama.

Works for RAPTOR results and any other chunking strategy that outputs the same format:
    {"question": "...", "gold_answer": "...", "raptor_answer": "...", "per_article_answers": [...]}

Usage:
    python score_results.py --results results/raptor/raptor_frames_results.jsonl
    python score_results.py --results results/gmm/gmm_frames_results.jsonl --llm_model qwen2.5:7b-instruct
"""

import json
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def load_results(path: str) -> list:
    results = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    log.info(f"Loaded {len(results)} results from {path}")
    return results


def score(args):
    # ── Imports ───────────────────────────────────────────────────────────────
    try:
        from ragas import evaluate
        from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness, ResponseRelevancy
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.run_config import RunConfig
        from datasets import Dataset
    except ImportError as e:
        log.error(f"Missing ragas: pip install ragas   ({e})")
        return

    try:
        from langchain_ollama import ChatOllama, OllamaEmbeddings
    except ImportError:
        try:
            from langchain_community.chat_models import ChatOllama
            from langchain_community.embeddings import OllamaEmbeddings
        except ImportError as e:
            log.error(f"Missing langchain-ollama: pip install langchain-ollama   ({e})")
            return

    # ── Load results ──────────────────────────────────────────────────────────
    results = load_results(args.results)

    # Detect field names flexibly — works for RAPTOR, GMM, or any strategy
    def get_answer(r):
        for key in ("raptor_answer", "gmm_answer", "answer", "generated_answer"):
            if r.get(key):
                return r[key]
        return ""

    def get_contexts(r):
        for key in ("per_article_answers", "contexts", "retrieved_chunks", "context"):
            if r.get(key):
                val = r[key]
                return val if isinstance(val, list) else [val]
        # Fallback: wrap the answer itself as context
        return [get_answer(r)]

    valid = [r for r in results if get_answer(r) and r.get("gold_answer")]
    if not valid:
        log.error("No valid results found — check that answers and gold_answers are populated.")
        return

    log.info(f"Scoring {len(valid)} valid results (skipped {len(results) - len(valid)} empty)")

    dataset = Dataset.from_dict({
        "question":     [r["question"]       for r in valid],
        "answer":       [get_answer(r)        for r in valid],
        "contexts":     [get_contexts(r)      for r in valid],
        "reference":    [r["gold_answer"]     for r in valid],  # new Ragas uses 'reference' not 'ground_truth'
    })

    # ── Build Ollama-backed evaluator ─────────────────────────────────────────
    # Use temperature=0 for deterministic evaluation scores
    llm = LangchainLLMWrapper(
        ChatOllama(model=args.llm_model, temperature=0)
    )
    emb = LangchainEmbeddingsWrapper(
        OllamaEmbeddings(model=args.embed_model)
    )

    # RunConfig: increase timeout since local Ollama is slower than API
    run_config = RunConfig(
        max_workers=1,      # serial to avoid overwhelming local Ollama
        timeout=120,        # 2 min per metric call
        max_retries=3,
    )

    metrics = [
        Faithfulness(),         # is the answer grounded in the retrieved context?
        FactualCorrectness(),   # does it match the gold answer factually?
        LLMContextRecall(),     # did retrieval capture what was needed?
        ResponseRelevancy(),    # is the answer relevant to the question?
    ]

    log.info(f"Running Ragas evaluation with {args.llm_model} ...")

    try:
        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=llm,
            embeddings=emb,
            run_config=run_config,
            raise_exceptions=False,   # don't crash on individual failures
        )
    except Exception as e:
        log.error(f"Ragas evaluation failed: {e}")
        return

    # ── Save + print ──────────────────────────────────────────────────────────
    results_dir = Path(args.results).parent
    scores_path = results_dir / "ragas_scores.json"

    # EvaluationResult API differs across Ragas versions
    if hasattr(result, "items"):
        scores_dict = {k: float(v) for k, v in result.items()}
    elif hasattr(result, "to_pandas"):
        df = result.to_pandas()
        scores_dict = {col: float(df[col].mean()) for col in df.columns if str(df[col].dtype).startswith("float")}
    elif hasattr(result, "scores"):
        import pandas as pd
        df = pd.DataFrame(result.scores)
        scores_dict = {col: float(df[col].mean()) for col in df.columns if str(df[col].dtype).startswith("float")}
    else:
        scores_dict = dict(result)
    with open(scores_path, "w", encoding="utf-8") as f:
        json.dump(scores_dict, f, indent=2)

    print("\n" + "="*50)
    print(f"Ragas scores for: {args.results}")
    print("="*50)
    for k, v in scores_dict.items():
        bar = "█" * int(v * 20)
        print(f"  {k:<30} {v:.4f}  {bar}")
    print(f"\nSaved to: {scores_path}")


# ------------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score RAG results with Ragas + local Ollama")
    parser.add_argument("--results",     required=True,
                        help="Path to the results JSONL file")
    parser.add_argument("--llm_model",   default="qwen2.5:7b-instruct",
                        help="Ollama model for evaluation (default: qwen2.5:7b-instruct)")
    parser.add_argument("--embed_model", default="nomic-embed-text",
                        help="Ollama embedding model (default: nomic-embed-text)")
    score(parser.parse_args())                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              