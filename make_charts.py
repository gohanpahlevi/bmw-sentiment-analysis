"""
Charts for the README, drawn from the committed result tables.

This reads results/company_intervals.csv, which is in the repository, so the
figures can be regenerated without the raw reviews.

Usage:
    python make_charts.py
"""

import csv
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

RESULTS = "results"
FIGURES = "figures"

PERIODS = ["1_pre_covid", "2_covid", "3_post_covid"]
PERIOD_LABELS = ["Pre-COVID\n2015-2019", "COVID\n2020-2022", "Post-COVID\n2023-2024"]

BMW = "#0066B1"
MERCEDES = "#333333"
OTHER = "#BBBBBB"

# The two the README compares directly, then the rest for context.
FOCUS = ["BMW", "Mercedes-Benz"]
CONTEXT = ["Audi", "Porsche", "Volkswagen"]


def read_intervals():
    """company -> period -> (mean, ci_low, ci_high, n)."""
    rows = defaultdict(dict)
    with open(os.path.join(RESULTS, "company_intervals.csv")) as f:
        for r in csv.DictReader(f):
            rows[r["company"]][r["period"]] = (
                float(r["mean"]), float(r["ci_low"]), float(r["ci_high"]), int(r["n"])
            )
    return rows


def series(data, company):
    return [data[company][p][0] for p in PERIODS]


def errors(data, company):
    """Distance from the mean to each end, which is what errorbar wants."""
    low = [data[company][p][0] - data[company][p][1] for p in PERIODS]
    high = [data[company][p][2] - data[company][p][0] for p in PERIODS]
    return [low, high]


def spread(labels, gap):
    """Push labels apart so close ones stay readable. Expects them sorted by y."""
    out = []
    for y, name in labels:
        if out and y - out[-1][0] < gap:
            y = out[-1][0] + gap
        out.append((y, name))
    return out


def style(ax):
    ax.axhline(0, color="#999999", linewidth=0.8, zorder=1)
    ax.set_xticks(range(len(PERIODS)))
    ax.set_xticklabels(PERIOD_LABELS, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#EEEEEE", zorder=0)
    ax.set_axisbelow(True)


def main():
    data = read_intervals()
    os.makedirs(FIGURES, exist_ok=True)

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.4))
    x = range(len(PERIODS))

    # Left. The comparison the README rests on, with the intervals shown, because
    # they overlap and the chart should not imply a cleaner result than there is.
    for company, colour in zip(FOCUS, [BMW, MERCEDES]):
        left.errorbar(
            x, series(data, company), yerr=errors(data, company),
            color=colour, marker="o", markersize=6, linewidth=2,
            capsize=4, label=company, zorder=3,
        )
    left.set_title("BMW against Mercedes-Benz, with 95% intervals", fontsize=11)
    left.set_ylabel("Mean sentiment")
    left.legend(frameon=False, fontsize=9)
    style(left)

    # Right. Means only. Audi and Porsche were already negative before COVID,
    # so BMW's fall started from a different place than theirs.
    for company in CONTEXT:
        right.plot(x, series(data, company), color=OTHER, marker="o",
                   markersize=4, linewidth=1.4, zorder=2)
    for company, colour in zip(FOCUS, [BMW, MERCEDES]):
        right.plot(x, series(data, company), color=colour, marker="o",
                   markersize=6, linewidth=2.2, zorder=3)

    # BMW and Porsche end 0.006 apart, so the labels are spread before drawing.
    ends = sorted((data[c]["3_post_covid"][0], c) for c in CONTEXT + FOCUS)
    for y, company in spread(ends, gap=0.026):
        colour = {"BMW": BMW, "Mercedes-Benz": MERCEDES}.get(company, "#777777")
        right.annotate(
            company, (2.08, y), color=colour, fontsize=8, va="center",
            fontweight="bold" if company in FOCUS else "normal",
        )
    right.set_title("All five carmakers, means only", fontsize=11)
    right.set_xlim(-0.15, 2.75)
    style(right)

    fig.suptitle(
        "Employee sentiment on home office and flexible working",
        fontsize=13, y=0.99,
    )
    fig.text(
        0.5, 0.005,
        "Sentiment scored +1, 0 or -1. Bootstrap intervals, 2,000 resamples. "
        "n per company and period runs from 51 to 378.",
        ha="center", fontsize=8, color="#777777",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))

    out = os.path.join(FIGURES, "sentiment_by_period.png")
    fig.savefig(out, dpi=200)
    print(f"written to {out}")


if __name__ == "__main__":
    main()
