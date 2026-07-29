"""Assembles one Markdown training report per crossed step milestone (10k, 50k,
100k, 500k, 1M, 5M, 10M - see training/callbacks.py's MilestoneReportCallback):
current Elo, best version, reward/winrate/LR/entropy curves, training params, and
(if a League is attached) an Elo curve plus opponent-comparison bars.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from ai.reporting import plots

METRIC_COLUMNS = ["step", "reward_mean", "win_rate", "ep_len_mean", "policy_loss", "value_loss", "entropy_loss", "learning_rate"]


def _load_metrics(metrics_csv: str | Path) -> pd.DataFrame:
    path = Path(metrics_csv)
    if not path.exists():
        return pd.DataFrame(columns=METRIC_COLUMNS)
    return pd.read_csv(path)


def _last_valid(df: pd.DataFrame, col: str):
    if df.empty or col not in df or not df[col].notna().any():
        return None
    return df[col].dropna().iloc[-1]


def generate_report(metrics_csv: str | Path, report_dir: str | Path, milestone: int, run_config: dict, league=None) -> Path:
    report_dir = Path(report_dir)
    milestone_dir = report_dir / f"milestone_{milestone}"
    milestone_dir.mkdir(parents=True, exist_ok=True)

    metrics = _load_metrics(metrics_csv)

    image_lines = []
    if not metrics.empty:
        image_lines.append(("Reward-Verlauf", plots.reward_curve(metrics, milestone_dir)))
        image_lines.append(("Siegquote", plots.win_rate_curve(metrics, milestone_dir)))
        image_lines.append(("Learning Rate", plots.learning_rate_curve(metrics, milestone_dir)))
        image_lines.append(("Entropy", plots.entropy_curve(metrics, milestone_dir)))

    best = league.best() if league is not None else None
    if league is not None:
        elo_path = plots.elo_curve(league, milestone_dir)
        if elo_path is not None:
            image_lines.append(("Elo-Rating", elo_path))

    best_elo = league.elo.get(best.name) if (league is not None and best is not None) else None
    last_reward = _last_valid(metrics, "reward_mean")
    last_winrate = _last_valid(metrics, "win_rate")
    last_lr = _last_valid(metrics, "learning_rate")

    lines = [
        f"# Trainingsreport - Meilenstein {milestone:,} Schritte".replace(",", "."),
        "",
        f"- **Aktuelles Elo (beste Version)**: {best_elo:.0f}" if best_elo is not None else "- **Aktuelles Elo**: n/a",
        f"- **Beste Version**: {best.name}" if best is not None else "- **Beste Version**: n/a",
        f"- **Reward (Mittelwert, letzte Episoden)**: {last_reward:.3f}" if last_reward is not None else "- **Reward**: n/a",
        f"- **Win Rate (letzte Episoden)**: {last_winrate * 100:.1f}%" if last_winrate is not None else "- **Win Rate**: n/a",
        f"- **Learning Rate (aktuell)**: {last_lr}" if last_lr is not None else "- **Learning Rate**: n/a",
        "",
        "## Trainingsparameter",
        "```yaml",
        yaml.safe_dump(run_config, sort_keys=False).strip(),
        "```",
        "",
        "## Diagramme",
        "",
    ]
    for caption, path in image_lines:
        lines.append(f"**{caption}**")
        lines.append("")
        lines.append(f"![{caption}]({path.name})")
        lines.append("")

    if league is not None and len(league.entries) > 1:
        lines.append("## Vergleich zu früheren Versionen (Elo)")
        lines.append("")
        lines.append("| Version | Schritt | Elo |")
        lines.append("|---|---|---|")
        for entry in sorted(league.entries.values(), key=lambda e: e.step):
            marker = " **(beste)**" if best is not None and entry.name == best.name else ""
            lines.append(f"| {entry.name}{marker} | {entry.step:,} | {league.elo.get(entry.name):.0f} |".replace(",", "."))
        lines.append("")

    report_path = milestone_dir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path
