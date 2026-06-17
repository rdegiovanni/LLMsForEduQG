"""
Retrieval Similarity Analysis
==============================
Analyzes retrieval quality from a CSV with columns:
    gt_question_id, gt_question, gt_support,
    retrieved_rank, retrieved_question, retrieved_support,
    support_distance, support_similarity, question_similarity

Usage:
    python analyze_retrieval_similarity.py --input <path_to_csv>

Outputs:
    - Console report (7 sections)
    - plots/01_similarity_by_rank.png
    - plots/02_per_question_heatmap.png
    - plots/03_scatter_with_false_positives.png
    - plots/04_joint_threshold_survival.png
    - plots/05_elbow_support.png
    - plots/06_elbow_question.png
    - false_positives.csv
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Retrieval similarity analysis")
    parser.add_argument("--input", required=True, help="Path to the retrieval CSV file")
    parser.add_argument("--support-threshold", type=float, default=0.65,
                        help="Override data-driven support threshold (default: auto)")
    parser.add_argument("--question-threshold", type=float, default=0.55,
                        help="Question similarity threshold (default: 0.55, task-informed)")
    return parser.parse_args()


# ── ELBOW DETECTION ────────────────────────────────────────────────────────────

def find_elbow(values: np.ndarray) -> int:
    """
    Find the elbow index in a sorted (descending) array using
    maximum perpendicular distance from the line connecting
    first and last point.
    """
    n = len(values)
    if n < 3:
        return 0
    x = np.arange(n, dtype=float)
    y = values.astype(float)

    # Line from first to last point
    x0, y0 = x[0], y[0]
    x1, y1 = x[-1], y[-1]
    dx, dy = x1 - x0, y1 - y0
    line_len = np.sqrt(dx**2 + dy**2)

    # Perpendicular distances
    distances = np.abs(dy * x - dx * y + x1 * y0 - y1 * x0) / line_len
    return int(np.argmax(distances))


def data_driven_threshold(series: pd.Series, label: str, out_path: str) -> float:
    """
    Sort values descending, find elbow, plot, return threshold value.
    """
    sorted_vals = np.sort(series.values)[::-1]
    elbow_idx = find_elbow(sorted_vals)
    threshold = float(sorted_vals[elbow_idx])

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(sorted_vals, color="#4C72B0", linewidth=1.2, label="Sorted similarity")
    ax.axvline(elbow_idx, color="#C44E52", linestyle="--", linewidth=1,
               label=f"Elbow at index {elbow_idx}")
    ax.axhline(threshold, color="#C44E52", linestyle=":", linewidth=1,
               label=f"Threshold = {threshold:.3f}")
    ax.scatter([elbow_idx], [threshold], color="#C44E52", zorder=5, s=60)
    ax.set_xlabel("Rank (sorted by similarity)")
    ax.set_ylabel("Similarity")
    ax.set_title(f"Elbow detection — {label}\n(auto threshold = {threshold:.3f})")
    ax.legend(fontsize=8)
    ax.grid(linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out_path}")

    return threshold


# ── ANALYSIS FUNCTIONS ─────────────────────────────────────────────────────────

def similarity_by_rank(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("retrieved_rank")[["support_similarity", "question_similarity"]]
        .mean().round(4)
    )


def rank_of_best_retrieval(df: pd.DataFrame) -> pd.Series:
    best_idx = df.groupby("gt_question_id")["support_similarity"].idxmax()
    return df.loc[best_idx]["retrieved_rank"].value_counts().sort_index()


def false_positive_cases(df: pd.DataFrame, q_thresh: float, sup_thresh: float) -> pd.DataFrame:
    """
    High question similarity but low support similarity.
    The retriever is confident but wrong — dangerous for generation.
    """
    mask = (df["question_similarity"] >= q_thresh) & (df["support_similarity"] < sup_thresh)
    cols = ["gt_question_id", "gt_question", "retrieved_rank",
            "retrieved_question", "retrieved_support",
            "support_similarity", "question_similarity"]
    return (df[mask][cols]
            .sort_values("question_similarity", ascending=False)
            .reset_index(drop=True))


def joint_survival(df: pd.DataFrame, thresholds: np.ndarray) -> np.ndarray:
    """
    For each threshold t, count pairs where BOTH similarities >= t.
    Returns array of survival counts.
    """
    counts = []
    for t in thresholds:
        n = ((df["support_similarity"] >= t) & (df["question_similarity"] >= t)).sum()
        counts.append(n)
    return np.array(counts)


def per_question_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["gt_question_id", "gt_question"])
        .agg(
            support_sim_mean=("support_similarity", "mean"),
            support_sim_max=("support_similarity", "max"),
            question_sim_mean=("question_similarity", "mean"),
            question_sim_max=("question_similarity", "max"),
        )
        .round(4).reset_index().sort_values("support_sim_mean")
    )


# ── PRINTING HELPERS ───────────────────────────────────────────────────────────

def section(title: str):
    print(f"\n{'─' * 60}\n  {title}\n{'─' * 60}")


def print_dict(d: dict):
    for k, v in d.items():
        print(f"  {k}: {v}")


# ── PLOT FUNCTIONS ─────────────────────────────────────────────────────────────

COLORS = {"support": "#4C72B0", "question": "#DD8452", "red": "#C44E52"}


def plot_similarity_by_rank(rank_stats: pd.DataFrame, out_path: str):
    ranks = rank_stats.index.tolist()
    x = np.arange(len(ranks))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width / 2, rank_stats["support_similarity"], width,
           label="Support similarity", color=COLORS["support"])
    ax.bar(x + width / 2, rank_stats["question_similarity"], width,
           label="Question similarity", color=COLORS["question"])
    ax.set_xlabel("Retrieved rank")
    ax.set_ylabel("Mean cosine similarity")
    ax.set_title("Mean similarity by retrieved rank")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Rank {r}" for r in ranks])
    ax.set_ylim(0, 1)
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.05))
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out_path}")


def plot_per_question_heatmap(pq: pd.DataFrame, out_path: str):
    metrics = ["support_sim_mean", "support_sim_max",
               "question_sim_mean", "question_sim_max"]
    labels = ["Support\nmean", "Support\nmax", "Question\nmean", "Question\nmax"]
    data = pq[metrics].values
    y_labels = [(q[:55] + "…") if len(q) > 55 else q for q in pq["gt_question"].tolist()]
    fig_height = max(4, len(pq) * 0.45 + 1.5)
    fig, ax = plt.subplots(figsize=(8, fig_height))
    im = ax.imshow(data, aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=7)
    ax.set_title("Per-question similarity summary\n(sorted by worst support mean → top)", fontsize=10)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color="black" if data[i, j] < 0.75 else "white")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Similarity")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


def plot_scatter(df: pd.DataFrame, out_path: str, q_thresh: float, sup_thresh: float):
    """
    Scatter of all pairs.
    Quadrants:
      ✅ top-right  = both high  → good for generation
      ❌ bottom-right = high Q-sim, low sup-sim → false positives
      ⚠️  top-left   = low Q-sim, high sup-sim → different question, right context
      ✗  bottom-left = both low → bad retrieval
    """
    good  = (df["question_similarity"] >= q_thresh) & (df["support_similarity"] >= sup_thresh)
    fp    = (df["question_similarity"] >= q_thresh) & (df["support_similarity"] <  sup_thresh)
    other = ~good & ~fp

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.scatter(df.loc[other, "question_similarity"], df.loc[other, "support_similarity"],
               alpha=0.35, s=25, color="gray", label="Other")
    ax.scatter(df.loc[good,  "question_similarity"], df.loc[good,  "support_similarity"],
               alpha=0.5,  s=30, color=COLORS["support"], label="✅ Both high (keep)")
    if fp.any():
        ax.scatter(df.loc[fp, "question_similarity"], df.loc[fp, "support_similarity"],
                   alpha=0.85, s=55, color=COLORS["red"], marker="X",
                   label="❌ False positive (Q high, sup low)")

    ax.axvline(q_thresh,   color=COLORS["question"], linestyle="--", linewidth=0.9,
               label=f"Q-sim threshold ({q_thresh:.3f})")
    ax.axhline(sup_thresh, color=COLORS["support"],  linestyle=":",  linewidth=0.9,
               label=f"Sup-sim threshold ({sup_thresh:.3f})")

    ax.set_xlabel("Question similarity")
    ax.set_ylabel("Support similarity")
    ax.set_title("Scatter: question vs support similarity\n(data-driven thresholds)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out_path}")


def plot_joint_survival(df: pd.DataFrame, out_path: str, q_thresh: float, sup_thresh: float):
    """
    Line plot: as joint threshold increases, how many pairs survive?
    Marks the auto-detected thresholds.
    """
    thresholds = np.linspace(0, 1, 200)
    counts = joint_survival(df, thresholds)
    total = len(df)

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax2 = ax1.twinx()

    ax1.plot(thresholds, counts, color=COLORS["support"], linewidth=1.5)
    ax2.plot(thresholds, counts / total * 100, color=COLORS["support"],
             linewidth=1.5, alpha=0)  # invisible — just for right axis scale

    # Mark auto thresholds
    joint_thresh = min(q_thresh, sup_thresh)
    n_at_joint = int(((df["support_similarity"] >= sup_thresh) &
                      (df["question_similarity"] >= q_thresh)).sum())

    ax1.axvline(q_thresh,   color=COLORS["question"], linestyle="--", linewidth=1,
                label=f"Q threshold ({q_thresh:.3f})")
    ax1.axvline(sup_thresh, color=COLORS["support"],  linestyle=":",  linewidth=1,
                label=f"Sup threshold ({sup_thresh:.3f})")
    ax1.scatter([joint_thresh], [n_at_joint], color=COLORS["red"], zorder=5, s=60,
                label=f"Pairs kept at both thresholds: {n_at_joint}/{total}")

    ax1.set_xlabel("Joint similarity threshold (both must be ≥ t)")
    ax1.set_ylabel("Pairs surviving (count)")
    ax2.set_ylabel("Pairs surviving (%)")
    ax2.set_ylim(0, ax1.get_ylim()[1] / total * 100)
    ax1.set_title("Joint threshold survival\n(how many pairs pass BOTH thresholds?)")
    ax1.legend(fontsize=8)
    ax1.grid(linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ── MAIN ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    df = pd.read_csv(args.input)
    print(f"\nLoaded {len(df)} rows, {df['gt_question_id'].nunique()} unique GT questions.")

    plots_dir = os.path.join(os.path.dirname(os.path.abspath(args.input)), "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # ── Auto thresholds (elbow detection) ──────────────────────────────────────
    section("0. Data-driven threshold detection (elbow method)")

    if args.support_threshold is not None:
        sup_thresh = args.support_threshold
        print(f"  Support threshold: {sup_thresh} (manual override)")
    else:
        sup_thresh = data_driven_threshold(
            df["support_similarity"], "Support similarity",
            os.path.join(plots_dir, "05_elbow_support.png")
        )
        print(f"  Support threshold (auto): {sup_thresh:.4f}")

    # Question similarity: elbow is unreliable here because the curve is smooth
    # (no natural cliff). 0.55 is task-informed: below this, questions drift
    # to same-domain but different-concept, which hurts few-shot quality.
    # Same-topic / different-angle pairs (0.55–0.75) are desirable — that's
    # the variation teachers need across exam years.
    q_thresh = args.question_threshold
    print(f"  Question threshold: {q_thresh:.3f} (task-informed, not elbow)")

    # ── 1. Similarity by rank ──────────────────────────────────────────────────
    section("1. Mean similarity by retrieved rank")
    rank_stats = similarity_by_rank(df)
    print(rank_stats.to_string())
    plot_similarity_by_rank(rank_stats, os.path.join(plots_dir, "01_similarity_by_rank.png"))

    # ── 2. Best rank per GT question ───────────────────────────────────────────
    section("2. Which rank holds the best support match?")
    print(rank_of_best_retrieval(df).to_string())

    # ── 3. Joint survival ──────────────────────────────────────────────────────
    section("3. Joint threshold survival")
    n_good = int(((df["support_similarity"] >= sup_thresh) &
                  (df["question_similarity"] >= q_thresh)).sum())
    print(f"  Pairs where BOTH >= threshold: {n_good} / {len(df)} "
          f"({n_good / len(df) * 100:.1f}%)")
    plot_joint_survival(df, os.path.join(plots_dir, "04_joint_threshold_survival.png"),
                        q_thresh, sup_thresh)

    # ── 4. False positives ────────────────────────────────────────────────────
    section(f"4. False positives (Q-sim >= {q_thresh:.3f}, sup-sim < {sup_thresh:.3f})")
    fp = false_positive_cases(df, q_thresh, sup_thresh)
    if fp.empty:
        print("  None found.")
    else:
        print(f"  {len(fp)} false positives flagged.\n")
        pd.set_option("display.max_colwidth", 80)
        print(fp[["gt_question_id", "gt_question", "retrieved_rank",
                   "retrieved_question", "support_similarity",
                   "question_similarity"]].to_string(index=False))
        fp_path = os.path.join(os.path.dirname(os.path.abspath(args.input)),
                               "false_positives.csv")
        fp.to_csv(fp_path, index=False)
        print(f"\n  Full rows (incl. retrieved_support) saved → {fp_path}")

    plot_scatter(df, os.path.join(plots_dir, "03_scatter_with_false_positives.png"),
                 q_thresh, sup_thresh)

    # ── 5. Per-question heatmap ────────────────────────────────────────────────
    section("5. Per-GT-question summary (sorted by worst support_sim_mean)")
    pq = per_question_summary(df)
    print(pq.to_string(index=False))
    plot_per_question_heatmap(pq, os.path.join(plots_dir, "02_per_question_heatmap.png"))

    section("Done")
    print(f"  All plots saved to: {plots_dir}/")


if __name__ == "__main__":
    main()