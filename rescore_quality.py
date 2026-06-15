"""
Re-score an existing QuALITY results JSONL with the CORRECT 0-indexed gold.

emozilla/quality stores `answer` as a 0-indexed int (0=A, 1=B, 2=C, 3=D). An
earlier 1-indexed assumption mislabeled every gold answer, so accuracy numbers
from older runs are wrong. This script recomputes them WITHOUT calling any LLM:
it reuses the predicted_letter already saved in the results file and only fixes
the gold side.

Usage:
    python rescore_quality.py results/quality/raptor_quality_collapsed_results.jsonl
    python rescore_quality.py <file> --split train --write   # also rewrite file with fixed gold/correct
"""

import json
import argparse
from pathlib import Path

from datasets import load_dataset

OPTION_LETTERS = ["A", "B", "C", "D"]


def gold_to_letter(raw) -> str:
    if isinstance(raw, bool):
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


def main():
    ap = argparse.ArgumentParser(description="Re-score QuALITY results with 0-indexed gold")
    ap.add_argument("results_file", help="path to *_results.jsonl")
    ap.add_argument("--split", default="train", choices=["train", "validation", "test"])
    ap.add_argument("--write", action="store_true",
                    help="overwrite the file in place with corrected gold_letter/correct fields")
    args = ap.parse_args()

    path = Path(args.results_file)
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]

    ds = load_dataset("emozilla/quality", split=args.split)
    # answer field is the 0-indexed gold; map idx -> letter
    gold_by_idx = {i: gold_to_letter(ds[i]["answer"]) for i in range(len(ds))}

    correct = answered = errors = 0
    for r in rows:
        idx = r.get("idx")
        true_gold = gold_by_idx.get(idx, r.get("gold_letter", ""))
        pred = r.get("predicted_letter", "")
        if r.get("error"):
            errors += 1
        if pred:
            answered += 1
            is_correct = (pred == true_gold)
            if is_correct:
                correct += 1
        else:
            is_correct = False
        # update the row in memory (for optional rewrite)
        r["gold_letter"] = true_gold
        r["correct"] = is_correct

    total = len(rows)
    print(f"\n{'='*60}")
    print(f"File: {path}")
    print(f"Rows: {total} | answered: {answered} | errors (no answer): {errors}")
    if answered:
        print(f"Accuracy over ANSWERED rows : {correct}/{answered} = {correct/answered:.1%}")
    print(f"Accuracy over ALL rows      : {correct}/{total} = {correct/total:.1%}")
    print(f"Paper (GPT-4 + RAPTOR collapsed): 76.2%")
    print(f"{'='*60}")

    if args.write:
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Rewrote {path} with corrected gold_letter/correct fields.")


if __name__ == "__main__":
    main()
