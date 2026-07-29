"""Matplotlib chart generators for training reports: reward, win-rate, Elo,
learning-rate and entropy curves, plus opponent-comparison bars. Used by
report.py to assemble the automatic milestone reports the brief asks for.

Palette/styling follows this project's dataviz conventions: one hue per series,
thin 2px lines, a light neutral surface, muted gridlines, and (for the fixed
sequence of opponent-comparison categories) the validated 8-slot categorical
order rather than matplotlib's default cycle.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # headless - never opens a window
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]


def _new_axes(figsize=(7, 4)):
    fig, ax = plt.subplots(figsize=figsize, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(BASELINE)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.xaxis.label.set_color(INK_SECONDARY)
    ax.yaxis.label.set_color(INK_SECONDARY)
    return fig, ax


def _line_chart(df: pd.DataFrame, x: str, y: str, title: str, ylabel: str, out_path: Path, color: str = CATEGORICAL[0]) -> Path:
    fig, ax = _new_axes()
    clean = df.dropna(subset=[y])
    ax.plot(clean[x], clean[y], color=color, linewidth=2, solid_capstyle="round")
    ax.set_title(title, color=INK_PRIMARY, fontsize=12, loc="left")
    ax.set_xlabel("Trainingsschritte")
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out_path


def reward_curve(metrics: pd.DataFrame, out_dir: Path) -> Path:
    return _line_chart(metrics, "step", "reward_mean", "Reward-Verlauf", "Durchschnittlicher Reward", out_dir / "reward_curve.png", CATEGORICAL[0])


def win_rate_curve(metrics: pd.DataFrame, out_dir: Path) -> Path:
    df = metrics.copy()
    df["win_rate_pct"] = df["win_rate"] * 100
    return _line_chart(df, "step", "win_rate_pct", "Siegquote", "Win Rate (%)", out_dir / "win_rate_curve.png", CATEGORICAL[2])


def learning_rate_curve(metrics: pd.DataFrame, out_dir: Path) -> Path:
    return _line_chart(metrics, "step", "learning_rate", "Learning Rate", "Learning Rate", out_dir / "learning_rate_curve.png", CATEGORICAL[3])


def entropy_curve(metrics: pd.DataFrame, out_dir: Path) -> Path:
    df = metrics.copy()
    df["entropy"] = -df["entropy_loss"]  # SB3 logs entropy *loss* (negative entropy); plot entropy itself
    return _line_chart(df, "step", "entropy", "Entropy (Exploration vs. Exploitation)", "Entropy", out_dir / "entropy_curve.png", CATEGORICAL[6])


def elo_curve(league, out_dir: Path) -> Path | None:
    if league is None or not league.entries:
        return None
    entries = sorted(league.entries.values(), key=lambda e: e.step)
    steps = [e.step for e in entries]
    elos = [league.elo.get(e.name) for e in entries]
    df = pd.DataFrame({"step": steps, "elo": elos})

    fig, ax = _new_axes()
    ax.plot(df["step"], df["elo"], color=CATEGORICAL[7], linewidth=2, marker="o", markersize=4)
    best = league.best()
    if best is not None:
        ax.scatter([best.step], [league.elo.get(best.name)], color=CATEGORICAL[5], s=60, zorder=5, label=f"beste Version: {best.name}")
        ax.legend(frameon=False, labelcolor=INK_SECONDARY, fontsize=9)
    ax.set_title("Elo-Rating", color=INK_PRIMARY, fontsize=12, loc="left")
    ax.set_xlabel("Trainingsschritte")
    ax.set_ylabel("Elo")
    fig.tight_layout()
    out_path = out_dir / "elo_curve.png"
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out_path


def opponent_comparison(summaries: dict, out_dir: Path, metric: str = "win_rate", title: str = "Gegnervergleich") -> Path | None:
    """`summaries`: {label: EvalSummary-like object with `.win_rate` etc.}"""
    if not summaries:
        return None
    labels = list(summaries.keys())
    values = [getattr(summaries[label], metric) * (100 if metric == "win_rate" else 1) for label in labels]
    colors = [CATEGORICAL[i % len(CATEGORICAL)] for i in range(len(labels))]

    fig, ax = _new_axes(figsize=(max(6, len(labels) * 1.1), 4))
    bars = ax.bar(labels, values, color=colors, width=0.6, zorder=3)
    for bar, value in zip(bars, values):
        ax.annotate(f"{value:.0f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()), ha="center", va="bottom", fontsize=9, color=INK_SECONDARY)
    ax.set_title(title, color=INK_PRIMARY, fontsize=12, loc="left")
    ax.set_ylabel("Win Rate (%)" if metric == "win_rate" else metric)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    out_path = out_dir / f"opponent_comparison_{metric}.png"
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out_path
