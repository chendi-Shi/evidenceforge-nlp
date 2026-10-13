"""Plot audited, complete MiniLM adaptation results without loading any model.

Optional plotting dependency: pip install matplotlib
Run from the repository root after both dataset evaluations and the independent
audit have completed: python scripts/plot_adaptation_results.py

The figure uses the experiment report's paired-query bootstrap intervals. The
independent audit verifies point estimates, not the bootstrap endpoints.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ARMS = ("base", "random", "retrieved", "filtered")
DATASETS = {"scifact": ("SciFact", 5183, 300), "nfcorpus": ("NFCorpus", 3633, 323)}
COLORS = ("#7B858C", "#5C9291", "#34699A", "#C78A38")
METRIC = "ndcg@10"
PRIMARY = "filtered_minus_retrieved"


def finite_number(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def load_results(results_dir: Path) -> tuple[dict, dict]:
    """Reject partial evaluations, failed audits, and mismatched point estimates."""
    report = json.loads((results_dir / "report.json").read_text(encoding="utf-8"))
    audit = json.loads((results_dir / "independent_audit.json").read_text(encoding="utf-8"))
    if audit.get("ok") is not True or audit.get("errors"):
        raise ValueError("A successful independent audit is required before plotting")
    if set(report["datasets"]) != set(DATASETS) or set(report["comparisons"]) != set(DATASETS):
        raise ValueError("Both complete SciFact and NFCorpus evaluations are required")

    values, comparisons = {}, {}
    for dataset, (_, corpus_size, query_count) in DATASETS.items():
        methods = report["datasets"][dataset]["methods"]
        if set(methods) != set(ARMS):
            raise ValueError(f"{dataset}: all four predeclared encoder variants are required")
        values[dataset] = []
        for arm in ARMS:
            method = methods[arm]
            if method["corpus_size"] != corpus_size or method["query_count"] != query_count:
                raise ValueError(f"{dataset}/{arm}: incomplete corpus or query coverage")
            score = finite_number(method["metrics"][METRIC], f"{dataset}/{arm}/{METRIC}")
            verified = finite_number(audit["verified_metrics"][dataset][arm][METRIC], "audited nDCG")
            if not 0 <= score <= 1 or not math.isclose(score, verified, rel_tol=0, abs_tol=1e-10):
                raise ValueError(f"{dataset}/{arm}: invalid score or stale independent audit")
            values[dataset].append(score)
        comparison = report["comparisons"][dataset][PRIMARY]
        delta, low, high = (
            finite_number(comparison[key], f"{dataset}/{PRIMARY}/{key}")
            for key in ("delta", "ci95_low", "ci95_high")
        )
        if low > high or min(delta, low, high) < -1 or max(delta, low, high) > 1:
            raise ValueError(f"{dataset}: invalid paired confidence interval")
        if not math.isclose(delta, values[dataset][3] - values[dataset][2], rel_tol=0, abs_tol=1e-10):
            raise ValueError(f"{dataset}: paired difference does not match audited aggregate metrics")
        if comparison["samples"] != 2000 or comparison["seed"] != 42:
            raise ValueError(f"{dataset}: unexpected bootstrap configuration")
        comparisons[dataset] = (delta, low, high)
    return values, comparisons


def plot_results(results_dir: Path) -> tuple[Path, Path]:
    values, comparisons = load_results(results_dir)
    try:
        import matplotlib
    except ImportError as exc:
        raise RuntimeError("Optional plotting dependency missing; install with: pip install matplotlib") from exc
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, MultipleLocator

    style = {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "axes.edgecolor": "#B7BEC4",
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": "#3C454D",
        "ytick.color": "#3C454D",
        "text.color": "#23313C",
        "axes.labelcolor": "#23313C",
        "svg.fonttype": "none",
        "svg.hashsalt": "evidenceforge-adaptation-v2",
        "savefig.facecolor": "white",
    }
    with plt.rc_context(style):
        figure, axes = plt.subplots(1, 3, figsize=(13.8, 5.4), gridspec_kw={"width_ratios": [1, 1, 1.25]})
        figure.subplots_adjust(left=0.065, right=0.975, bottom=0.28, top=0.76, wspace=0.37)
        figure.suptitle("MiniLM adaptation: full-corpus retrieval", x=0.065, y=0.955,
                       ha="left", fontsize=17, fontweight="bold")
        figure.text(0.065, 0.885, "Same float baseline and 128-token budget; final-epoch models fixed before evaluation",
                    ha="left", fontsize=10.5, color="#52616C")

        for axis, (dataset, (label, corpus_size, query_count)) in zip(axes[:2], DATASETS.items(), strict=True):
            bars = axis.bar(range(len(ARMS)), values[dataset], color=COLORS, width=0.64,
                            edgecolor="white", linewidth=0.7, zorder=3)
            axis.bar_label(bars, labels=[f"{value:.6f}" for value in values[dataset]], padding=5, fontsize=8.5)
            axis.set_title(f"{label}\n{corpus_size:,} documents | {query_count} queries", loc="left", pad=14)
            axis.set_xticks(range(len(ARMS)), [arm.capitalize() for arm in ARMS], rotation=25, ha="right")
            axis.set_ylim(0, 1)
            axis.yaxis.set_major_locator(MultipleLocator(0.2))
            axis.set_ylabel("nDCG@10 (linear relevance gain)")
            axis.grid(axis="y", color="#E5E9EC", linewidth=0.7, zorder=0)
            axis.tick_params(axis="both", length=3)

        axis = axes[2]
        axis.set_title("Primary comparison\nFiltered minus retrieved", loc="left", pad=14)
        axis.axvline(0, color="#7B858C", linestyle=(0, (3, 3)), linewidth=1, zorder=1)
        for position, dataset in enumerate(DATASETS):
            delta, low, high = comparisons[dataset]
            axis.hlines(position, low, high, color="#34699A", linewidth=2, zorder=3)
            axis.vlines([low, high], position - 0.055, position + 0.055,
                        color="#34699A", linewidth=1.2, zorder=3)
            axis.scatter(delta, position, color="#23313C", s=38, zorder=4)
            annotation = f"{delta:+.6f}  [{low:+.6f}, {high:+.6f}]"
            axis.text(0.5, position + 0.20, annotation, transform=axis.get_yaxis_transform(),
                      ha="center", va="center", fontsize=8.8)
        bound = max(abs(value) for comparison in comparisons.values() for value in comparison)
        # Symmetric limits keep positive and negative effects visually comparable.
        tick_step = max(0.01, math.ceil(bound * 1.25 / 2 / 0.01) * 0.01)
        axis.set_xlim(-2 * tick_step, 2 * tick_step)
        axis.xaxis.set_major_locator(MultipleLocator(tick_step))
        axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: "0" if abs(value) < 1e-12 else f"{value:+.2f}"))
        axis.set_yticks(range(len(DATASETS)), [item[0] for item in DATASETS.values()])
        axis.set_ylim(1.45, -0.40)
        axis.set_xlabel("Difference in nDCG@10 (absolute)", labelpad=10)
        axis.grid(axis="x", color="#E5E9EC", linewidth=0.7, zorder=0)
        axis.spines["left"].set_visible(False)
        axis.tick_params(axis="y", length=0)
        figure.text(0.065, 0.11,
                    "One optimization seed. Error bars: 95% paired-query bootstrap intervals (2,000 resamples); "
                    "not training-seed variance.", ha="left", fontsize=9, color="#52616C")
        figure.text(0.065, 0.067,
                    "All four encoders shown; no best-test selection. Audit verifies point estimates; intervals "
                    "come from the experiment report.", ha="left", fontsize=9, color="#52616C")
        destinations = tuple(results_dir / f"comparison.{extension}" for extension in ("svg", "png"))
        try:
            figure.savefig(destinations[0], metadata={"Date": None}, bbox_inches="tight")
            # Matplotlib emits harmless trailing blanks in multiline SVG paths.
            svg = "\n".join(line.rstrip() for line in destinations[0].read_text(encoding="utf-8").splitlines()) + "\n"
            destinations[0].write_text(svg, encoding="utf-8", newline="\n")
            figure.savefig(destinations[1], dpi=220, bbox_inches="tight")
        finally:
            plt.close(figure)
    return destinations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results/adaptation"))
    args = parser.parse_args()
    try:
        destinations = plot_results(args.results_dir)
    except (KeyError, OSError, TypeError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"Cannot plot adaptation results: {exc}\n")
    for destination in destinations:
        print(destination)


if __name__ == "__main__":
    main()
