"""
Generate analysis plots and statistics from RAPTOR FRAMES results.
Outputs figures to outputs/figures/ for use in the report.
"""
import ast
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS_JSONL = ROOT / "results" / "raptor" / "raptor_frames_results.jsonl"
TREES_DIR = ROOT / "results" / "raptor" / "trees"
FIGS = ROOT / "outputs" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

# ── colour palette ────────────────────────────────────────────────────────────
C_BLUE   = "#2C5F2D"   # answered (dark green)
C_AMBER  = "#E07B39"   # insufficient context
C_RED    = "#B85042"   # error / blank
C_GREY   = "#8D99AE"   # grid / neutral
C_LIGHT  = "#EDF2F4"   # background
PALETTE  = [C_BLUE, C_AMBER, C_RED, "#065A82", "#028090", "#6D2E46"]


# ── helpers ───────────────────────────────────────────────────────────────────
REFUSAL_MARKERS = [
    "does not contain",
    "not enough information",
    "cannot answer",
    "cannot determine",
    "cannot provide",
    "not possible",
    "based solely on",
    "provided context",
]

def classify(row):
    err = row.get("error", "")
    ans = str(row.get("raptor_answer", "") or "").strip()
    if err:
        return "Pipeline Error"
    if not ans:
        return "Blank Answer"
    low = ans.lower()
    if any(m in low for m in REFUSAL_MARKERS):
        return "Insufficient Context"
    return "Answered"

def load_results():
    rows = []
    with RESULTS_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    for r in rows:
        r["_status"] = classify(r)
        per = r.get("per_article_answers") or []
        if isinstance(per, str):
            try:
                per = ast.literal_eval(per)
            except Exception:
                per = [per] if per else []
        r["_n_articles"] = len(per) if isinstance(per, list) else 0
        r["_answer_len"] = len(str(r.get("raptor_answer","") or ""))
    return rows


# ── plot 1 : answer-status pie ────────────────────────────────────────────────
def plot_status_pie(rows):
    counts = Counter(r["_status"] for r in rows)
    labels = list(counts.keys())
    values = list(counts.values())
    colors = {
        "Answered":             C_BLUE,
        "Insufficient Context": C_AMBER,
        "Pipeline Error":       C_RED,
        "Blank Answer":         C_GREY,
    }
    clrs = [colors.get(l, C_GREY) for l in labels]

    fig, ax = plt.subplots(figsize=(6, 5), facecolor="white")
    wedges, texts, autotexts = ax.pie(
        values, labels=None, autopct="%1.0f%%",
        colors=clrs, startangle=140,
        pctdistance=0.75, wedgeprops=dict(linewidth=1.5, edgecolor="white")
    )
    for at in autotexts:
        at.set_fontsize(12)
        at.set_color("white")
        at.set_fontweight("bold")

    patches = [mpatches.Patch(color=colors.get(l, C_GREY), label=f"{l}  ({v})")
               for l, v in zip(labels, values)]
    ax.legend(handles=patches, loc="lower center", bbox_to_anchor=(0.5, -0.12),
              ncol=2, frameon=False, fontsize=10)
    ax.set_title(f"RAPTOR FRAMES Answer Status  (n={len(rows)})",
                 fontsize=13, fontweight="bold", pad=14)
    fig.tight_layout()
    out = FIGS / "fig1_answer_status_pie.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")
    return counts


# ── plot 2 : articles-per-question bar ───────────────────────────────────────
def plot_articles_bar(rows):
    n_articles = [r["_n_articles"] for r in rows]
    cnt = Counter(n_articles)
    xs = sorted(cnt.keys())
    ys = [cnt[x] for x in xs]

    fig, ax = plt.subplots(figsize=(7, 4), facecolor="white")
    bars = ax.bar([str(x) for x in xs], ys, color=C_BLUE, edgecolor="white", linewidth=1.2)
    for bar, y in zip(bars, ys):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                str(y), ha="center", va="bottom", fontsize=10)
    ax.set_xlabel("Number of Wikipedia Articles Retrieved", fontsize=11)
    ax.set_ylabel("Question Count", fontsize=11)
    ax.set_title("Wikipedia Articles per FRAMES Question", fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_facecolor(C_LIGHT)
    ax.yaxis.grid(True, color="white", linewidth=1.2)
    ax.set_axisbelow(True)
    fig.tight_layout()
    out = FIGS / "fig2_articles_per_question.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ── plot 3 : answer length distribution by status ────────────────────────────
def plot_answer_length(rows):
    statuses = ["Answered", "Insufficient Context"]
    data = {s: [r["_answer_len"] for r in rows if r["_status"] == s] for s in statuses}
    colors_map = {"Answered": C_BLUE, "Insufficient Context": C_AMBER}

    fig, ax = plt.subplots(figsize=(7, 4), facecolor="white")
    for i, s in enumerate(statuses):
        vals = data[s]
        ax.hist(vals, bins=15, alpha=0.75, color=colors_map[s],
                label=f"{s} (n={len(vals)})", edgecolor="white")
    ax.set_xlabel("Answer Length (characters)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("RAPTOR Answer Length Distribution by Status", fontsize=13, fontweight="bold")
    ax.legend(frameon=False, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_facecolor(C_LIGHT)
    fig.tight_layout()
    out = FIGS / "fig3_answer_length_dist.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ── plot 4 : RAPTOR vs Baseline conceptual comparison bar ─────────────────────
def plot_raptor_vs_baseline():
    """
    Illustrates expected RAPTOR advantage on long-doc QA benchmarks
    (values from the original RAPTOR paper, Table 1).
    """
    benchmarks = ["NarrativeQA\n(METEOR)", "QASPER\n(F-1)", "QuALITY\n(Accuracy)"]
    baseline   = [0.121, 0.422, 0.561]   # best non-RAPTOR baseline in paper
    raptor     = [0.175, 0.557, 0.762]   # RAPTOR (collapsed, GPT-4)

    x = np.arange(len(benchmarks))
    w = 0.34
    fig, ax = plt.subplots(figsize=(8, 5), facecolor="white")
    b1 = ax.bar(x - w/2, baseline, w, label="Best Baseline",
                color=C_GREY, edgecolor="white", linewidth=1.2)
    b2 = ax.bar(x + w/2, raptor, w, label="RAPTOR (paper)",
                color=C_BLUE, edgecolor="white", linewidth=1.2)

    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks, fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("RAPTOR vs. Best Baseline (Paper Results)", fontsize=13, fontweight="bold")
    ax.legend(frameon=False, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_facecolor(C_LIGHT)
    ax.yaxis.grid(True, color="white", linewidth=1.2)
    ax.set_axisbelow(True)
    fig.tight_layout()
    out = FIGS / "fig4_raptor_vs_baseline_paper.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ── plot 5 : FRAMES gold-string-found metric ─────────────────────────────────
def plot_gold_found(rows):
    def norm(t):
        return re.sub(r"\s+", " ", str(t or "")).strip().lower()

    found = sum(1 for r in rows
                if r.get("gold_answer")
                and norm(r["gold_answer"]) in norm(r.get("raptor_answer", "")))
    not_found = len(rows) - found

    fig, ax = plt.subplots(figsize=(5, 4), facecolor="white")
    ax.bar(["Gold Answer\nFound in Response", "Not Found"],
           [found, not_found],
           color=[C_BLUE, C_AMBER], edgecolor="white", linewidth=1.2)
    for i, v in enumerate([found, not_found]):
        ax.text(i, v + 0.2, str(v), ha="center", va="bottom", fontsize=13, fontweight="bold")
    ax.set_ylabel("Questions", fontsize=11)
    ax.set_title("Soft Accuracy: Gold Answer Substring Match\n(FRAMES, n=50)",
                 fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_facecolor(C_LIGHT)
    fig.tight_layout()
    out = FIGS / "fig5_gold_found.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")
    return found


# ── plot 6 : RAPTOR tree architecture diagram ─────────────────────────────────
def plot_tree_diagram():
    fig, ax = plt.subplots(figsize=(9, 6), facecolor="white")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    def node(cx, cy, text, color, r=0.48, fontsize=8):
        circle = plt.Circle((cx, cy), r, color=color, zorder=3)
        ax.add_patch(circle)
        ax.text(cx, cy, text, ha="center", va="center",
                fontsize=fontsize, color="white", fontweight="bold", zorder=4,
                wrap=True, multialignment="center")

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#444", lw=1.4), zorder=2)

    # Leaf nodes (layer 0)
    leaf_xs = [1.2, 2.5, 3.8, 5.1, 6.4, 7.7, 9.0]
    for x in leaf_xs:
        node(x, 0.8, "Chunk", "#5C8374", fontsize=7)

    # Layer 1 summaries
    l1_xs = [1.85, 4.45, 7.05]
    for x in l1_xs:
        node(x, 2.6, "Sum L1", "#028090", fontsize=7)

    # Layer 2 summaries
    l2_xs = [3.1, 6.3]
    for x in l2_xs:
        node(x, 4.2, "Sum L2", "#065A82", fontsize=7)

    # Root
    node(5.0, 5.6, "Root", "#1E2761", fontsize=8)

    # Arrows leaf->L1  (each pair of leaves -> one L1 node; last leaf alone -> last L1)
    leaf_to_l1 = [0, 0, 1, 1, 2, 2, 2]
    for xi, li in zip(leaf_xs, leaf_to_l1):
        lx = l1_xs[li]
        arrow(xi, 1.28, lx, 2.12)

    # Arrows L1->L2
    for i, lx in enumerate(l1_xs):
        rx = l2_xs[i//2]
        arrow(lx, 3.08, rx, 3.72)

    # Arrows L2->root
    for rx in l2_xs:
        arrow(rx, 4.68, 5.0, 5.12)

    # Labels on right
    for y, label, color in [
        (0.8, "Layer 0 — Leaf Chunks (~100 tokens)", "#5C8374"),
        (2.6, "Layer 1 — Cluster Summaries", "#028090"),
        (4.2, "Layer 2 — Higher-Level Summaries", "#065A82"),
        (5.6, "Root — Full Document Summary", "#1E2761"),
    ]:
        ax.text(9.7, y, label, va="center", fontsize=8.5, color=color, fontweight="bold")

    ax.set_title("RAPTOR Recursive Tree Structure", fontsize=13, fontweight="bold", pad=10)
    fig.tight_layout()
    out = FIGS / "fig6_tree_diagram.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ── plot 7 : retrieval strategy comparison (collapsed vs traversal) ───────────
def plot_retrieval_strategies():
    categories = ["QuALITY\n(Acc.)", "NarrativeQA\n(METEOR)", "QASPER\n(F-1)"]
    traversal = [0.714, 0.167, 0.542]
    collapsed  = [0.762, 0.175, 0.557]

    x = np.arange(len(categories))
    w = 0.34
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor="white")
    b1 = ax.bar(x - w/2, traversal, w, label="Tree Traversal",
                color=C_AMBER, edgecolor="white")
    b2 = ax.bar(x + w/2, collapsed, w, label="Collapsed Tree",
                color=C_BLUE, edgecolor="white")
    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Retrieval Strategy: Tree Traversal vs Collapsed Tree\n(Paper Results, GPT-4)",
                 fontsize=12, fontweight="bold")
    ax.legend(frameon=False, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_facecolor(C_LIGHT)
    ax.yaxis.grid(True, color="white", linewidth=1.2)
    ax.set_axisbelow(True)
    fig.tight_layout()
    out = FIGS / "fig7_retrieval_strategies.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ── print summary stats ───────────────────────────────────────────────────────
def print_stats(rows, counts, gold_found):
    print("\n" + "="*55)
    print("  RAPTOR FRAMES Reproduction — Summary Statistics")
    print("="*55)
    total = len(rows)
    print(f"  Total questions evaluated : {total}")
    for status, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {status:<30}: {cnt:3d}  ({cnt/total*100:.0f}%)")
    print(f"\n  Gold-answer substring match : {gold_found}/{total}  ({gold_found/total*100:.0f}%)")
    n_arts = [r["_n_articles"] for r in rows]
    print(f"  Avg Wikipedia articles/Q   : {np.mean(n_arts):.2f}")
    print(f"  Max Wikipedia articles/Q   : {max(n_arts)}")
    ans_lens = [r["_answer_len"] for r in rows if r["_answer_len"] > 0]
    print(f"  Avg answer length (chars)  : {np.mean(ans_lens):.0f}")
    print("="*55 + "\n")


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    rows = load_results()
    counts = plot_status_pie(rows)
    plot_articles_bar(rows)
    plot_answer_length(rows)
    plot_raptor_vs_baseline()
    gold_found = plot_gold_found(rows)
    plot_tree_diagram()
    plot_retrieval_strategies()
    print_stats(rows, counts, gold_found)
    print(f"All figures saved to: {FIGS}")
