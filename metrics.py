"""
Evaluation metrics for the RAPTOR reproduction, matching the paper:

  QASPER       -> token-level Answer F1 (max over reference answers)
  NarrativeQA  -> BLEU-1, BLEU-4, ROUGE-L, METEOR (max over the 2 references)

All implementations are pure-Python (no nltk / rouge_score dependency) so they
run anywhere the runners do. METEOR is the standard exact-match formulation
(unigram alignment + fragmentation penalty); it omits WordNet synonym/stem
matching, so treat it as a close approximation of the official METEOR.
"""

import re
import math
from collections import Counter

# ─────────────────────────────────────────────────────────────────────────────
# Tokenization / normalization
# ─────────────────────────────────────────────────────────────────────────────

def normalize(s: str) -> str:
    """SQuAD/QASPER-style normalization: lowercase, drop articles & punctuation."""
    s = (s or "").lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def norm_tokens(s: str):
    return normalize(s).split()


def word_tokens(s: str):
    """Lighter tokenization for BLEU/ROUGE/METEOR (keeps no articles removal)."""
    return re.findall(r"[a-z0-9]+", (s or "").lower())


# ─────────────────────────────────────────────────────────────────────────────
# QASPER: token-level F1
# ─────────────────────────────────────────────────────────────────────────────

def token_f1(pred: str, gold: str) -> float:
    p, g = norm_tokens(pred), norm_tokens(gold)
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision = same / len(p)
    recall = same / len(g)
    return 2 * precision * recall / (precision + recall)


def max_token_f1(pred: str, golds) -> float:
    golds = [g for g in golds if g is not None and str(g).strip() != ""]
    if not golds:
        # No gold text usually means "unanswerable": credit an empty prediction.
        return 1.0 if not norm_tokens(pred) else 0.0
    return max(token_f1(pred, g) for g in golds)


# ─────────────────────────────────────────────────────────────────────────────
# NarrativeQA: BLEU
# ─────────────────────────────────────────────────────────────────────────────

def _ngram_counts(tokens, n):
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def sentence_bleu(pred: str, refs, max_n: int = 4) -> float:
    """
    Sentence-level BLEU-max_n in [0,1] with add-1 (Laplace) smoothing on the
    n-gram precisions and the standard brevity penalty. refs is a list of
    reference strings; n-grams are clipped against the best-matching reference.
    """
    c = word_tokens(pred)
    references = [word_tokens(r) for r in refs if r]
    if not c or not references:
        return 0.0

    log_prec_sum = 0.0
    for n in range(1, max_n + 1):
        cand_ng = _ngram_counts(c, n)
        cand_total = sum(cand_ng.values())
        if cand_total == 0:
            # candidate shorter than n: smoothed precision = 1/(0+1)
            log_prec_sum += (1.0 / max_n) * math.log(1.0 / 1.0e9)
            continue
        max_ref = Counter()
        for ref in references:
            for g, ct in _ngram_counts(ref, n).items():
                if ct > max_ref[g]:
                    max_ref[g] = ct
        clip = sum(min(ct, max_ref[g]) for g, ct in cand_ng.items())
        # add-1 smoothing
        precision = (clip + 1.0) / (cand_total + 1.0)
        log_prec_sum += (1.0 / max_n) * math.log(precision)

    # brevity penalty against the closest reference length
    closest = min((len(r) for r in references),
                  key=lambda rl: (abs(rl - len(c)), rl))
    bp = 1.0 if len(c) > closest else math.exp(1.0 - closest / len(c))
    return bp * math.exp(log_prec_sum)


def max_bleu(pred: str, refs, max_n: int = 4) -> float:
    """BLEU computed against all refs jointly (standard multi-reference BLEU)."""
    return sentence_bleu(pred, refs, max_n=max_n)


# ─────────────────────────────────────────────────────────────────────────────
# NarrativeQA: ROUGE-L (LCS-based F-measure)
# ─────────────────────────────────────────────────────────────────────────────

def _lcs_len(a, b):
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        ai = a[i - 1]
        for j in range(1, len(b) + 1):
            if ai == b[j - 1]:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = prev[j] if prev[j] >= cur[j - 1] else cur[j - 1]
        prev = cur
    return prev[len(b)]


def rouge_l(pred: str, refs) -> float:
    """ROUGE-L F1 (beta=1), max over references."""
    c = word_tokens(pred)
    best = 0.0
    for r in refs:
        g = word_tokens(r)
        if not c or not g:
            continue
        lcs = _lcs_len(c, g)
        if lcs == 0:
            continue
        prec = lcs / len(c)
        rec = lcs / len(g)
        f = 2 * prec * rec / (prec + rec)
        best = max(best, f)
    return best


# ─────────────────────────────────────────────────────────────────────────────
# NarrativeQA: METEOR (exact-match approximation, no WordNet)
# ─────────────────────────────────────────────────────────────────────────────

def _meteor_single(c, g) -> float:
    if not c or not g:
        return 0.0
    # greedy unigram alignment by exact match
    g_pool = Counter(g)
    matches = 0
    matched_positions = []
    for idx, tok in enumerate(c):
        if g_pool[tok] > 0:
            g_pool[tok] -= 1
            matches += 1
            matched_positions.append(idx)
    if matches == 0:
        return 0.0
    precision = matches / len(c)
    recall = matches / len(g)
    fmean = (10 * precision * recall) / (recall + 9 * precision)
    # fragmentation penalty: count chunks of consecutive matched positions
    chunks = 1
    for k in range(1, len(matched_positions)):
        if matched_positions[k] != matched_positions[k - 1] + 1:
            chunks += 1
    penalty = 0.5 * (chunks / matches) ** 3
    return fmean * (1 - penalty)


def meteor(pred: str, refs) -> float:
    c = word_tokens(pred)
    return max((_meteor_single(c, word_tokens(r)) for r in refs), default=0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: all NarrativeQA metrics at once
# ─────────────────────────────────────────────────────────────────────────────

def narrativeqa_metrics(pred: str, refs) -> dict:
    return {
        "bleu1":   max_bleu(pred, refs, max_n=1),
        "bleu4":   max_bleu(pred, refs, max_n=4),
        "rouge_l": rouge_l(pred, refs),
        "meteor":  meteor(pred, refs),
    }


if __name__ == "__main__":
    # quick self-test
    print("F1 yes/yes:", token_f1("Yes", "yes"))
    print("F1 partial:", round(token_f1("the cat sat on the mat", "a cat sat"), 3))
    print("maxF1:", round(max_token_f1("BERT is a transformer", ["a transformer model", "BERT"]), 3))
    refs = ["the cat sat on the mat"]
    print("BLEU1:", round(max_bleu("the cat sat", refs, 1), 3))
    print("BLEU4:", round(max_bleu("the cat sat on the mat", refs, 4), 3))
    print("ROUGE-L:", round(rouge_l("the cat sat", refs), 3))
    print("METEOR:", round(meteor("the cat sat", refs), 3))
