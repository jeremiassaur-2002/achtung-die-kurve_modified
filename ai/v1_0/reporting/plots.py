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


def survival_time_curve(metrics: pd.DataFrame, out_dir: Path) -> Path:
    """Overlebenszeit in Sekunden (ep_len_mean / 60 bei 60 Hz). ep_len_mean ist
    bereits ein gleitender Mittelwert über die letzten 100 Episoden (siehe
    MetricsLoggingCallback) - also die "zeitgemittelte" Überlebenszeit. In
    Solo-Phasen (n_opponents=0) ist DAS die Fortschrittsmetrik, nicht win_rate."""
    df = metrics.copy()
    df["survival_s"] = df["ep_len_mean"] / 60.0
    return _line_chart(
        df, "step", "survival_s", "Überlebenszeit (zeitgemittelt, 100 Episoden)", "Ø Überlebenszeit (s)", out_dir / "survival_time_curve.png", CATEGORICAL[4]
    )


# Rolling-mean windows for the mean_*.png chart variants, counted in metrics.csv
# ROWS (one row = one logged sample, every metrics_log_every_steps training steps -
# the x-axis stays "Trainingsschritte"). While fewer rows exist than the window,
# rolling(min_periods=1) degrades gracefully into the running average of
# everything so far and converges to a true moving window as the run grows.
MEAN_WINDOWS = (1_000, 10_000)


def _mean_chart(df: pd.DataFrame, y: str, title: str, ylabel: str, out_path: Path, color: str) -> Path | None:
    if y not in df or not df[y].notna().any():
        return None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    clean = df.dropna(subset=[y]).reset_index(drop=True)
    fig, ax = _new_axes()
    ax.plot(clean["step"], clean[y], color=INK_MUTED, linewidth=1, alpha=0.35, label="roh")
    for window, series_color in zip(MEAN_WINDOWS, (color, INK_PRIMARY)):
        rolled = clean[y].rolling(window, min_periods=1).mean()
        ax.plot(clean["step"], rolled, color=series_color, linewidth=2, solid_capstyle="round", label=f"Ø {window:,} Messpunkte".replace(",", "."))
    ax.legend(frameon=False, labelcolor=INK_SECONDARY, fontsize=9)
    ax.set_title(title, color=INK_PRIMARY, fontsize=12, loc="left")
    ax.set_xlabel("Trainingsschritte")
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out_path


def mean_variants(metrics: pd.DataFrame, out_dir: Path) -> list[tuple[str, Path]]:
    """One mean_<name>.png per Kurve des Reports: Rohserie blass im Hintergrund,
    darüber gleitende Mittel über 1.000 und 10.000 Messpunkte."""
    df = metrics.copy()
    if "entropy_loss" in df:
        df["entropy"] = -df["entropy_loss"]
    if "win_rate" in df:
        df["win_rate_pct"] = df["win_rate"] * 100
    if "ep_len_mean" in df:
        df["survival_s"] = df["ep_len_mean"] / 60.0

    jobs = [
        ("mean_reward_curve.png", "reward_mean", "Reward-Verlauf (gemittelt)", "Durchschnittlicher Reward", CATEGORICAL[0]),
        ("mean_win_rate_curve.png", "win_rate_pct", "Siegquote (gemittelt)", "Win Rate (%)", CATEGORICAL[2]),
        ("mean_learning_rate_curve.png", "learning_rate", "Learning Rate (gemittelt)", "Learning Rate", CATEGORICAL[3]),
        ("mean_entropy_curve.png", "entropy", "Entropy (gemittelt)", "Entropy", CATEGORICAL[6]),
        ("mean_survival_time_curve.png", "survival_s", "Überlebenszeit (gemittelt)", "Ø Überlebenszeit (s)", CATEGORICAL[4]),
    ]
    written: list[tuple[str, Path]] = []
    for filename, column, title, ylabel, color in jobs:
        path = _mean_chart(df, column, title, ylabel, out_dir / filename, color)
        if path is not None:
            written.append((title, path))
    return written


def death_causes_curve(dc: pd.DataFrame, out_dir: Path) -> Path | None:
    """Anteile der Episoden-Endursachen (gleitend über die letzten 100 Episoden):
    wall/self/doomed/... - beantwortet 'WORAN stirbt der Agent gerade?' direkt."""
    cause_cols = [c for c in dc.columns if c != "step" and dc[c].notna().any()]
    if dc.empty or not cause_cols:
        return None
    fig, ax = _new_axes()
    for i, col in enumerate(cause_cols):
        ax.plot(dc["step"], dc[col] * 100, color=CATEGORICAL[i % len(CATEGORICAL)], linewidth=2, solid_capstyle="round", label=col)
    ax.legend(frameon=False, labelcolor=INK_SECONDARY, fontsize=9, ncol=2)
    ax.set_title("Episoden-Ende: Ursachenanteile", color=INK_PRIMARY, fontsize=12, loc="left")
    ax.set_xlabel("Trainingsschritte")
    ax.set_ylabel("Anteil (%)")
    fig.tight_layout()
    out_path = out_dir / "death_causes_curve.png"
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out_path


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
