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

    def _has(col: str) -> bool:  # old metrics.csv files may lack newer columns - never crash a report over that
        return not metrics.empty and col in metrics and metrics[col].notna().any()

    image_lines = []
    if _has("reward_mean"):
        image_lines.append(("Reward-Verlauf", plots.reward_curve(metrics, milestone_dir)))
    if _has("ep_len_mean"):
        image_lines.append(("Überlebenszeit", plots.survival_time_curve(metrics, milestone_dir)))
    if _has("win_rate"):
        image_lines.append(("Siegquote", plots.win_rate_curve(metrics, milestone_dir)))
    if _has("learning_rate"):
        image_lines.append(("Learning Rate", plots.learning_rate_curve(metrics, milestone_dir)))
    if _has("entropy_loss"):
        image_lines.append(("Entropy", plots.entropy_curve(metrics, milestone_dir)))
    if not metrics.empty:
        image_lines.extend(plots.mean_variants(metrics, milestone_dir))

    death_csv = Path(metrics_csv).with_name("death_causes.csv")
    if death_csv.exists():
        dc = pd.read_csv(death_csv)
        dc_path = plots.death_causes_curve(dc, milestone_dir)
        if dc_path is not None:
            image_lines.append(("Episoden-Ende: Ursachen", dc_path))

    best = league.best() if league is not None else None
    if league is not None:
        elo_path = plots.elo_curve(league, milestone_dir)
        if elo_path is not None:
            image_lines.append(("Elo-Rating", elo_path))

    best_elo = league.elo.get(best.name) if (league is not None and best is not None) else None
    last_reward = _last_valid(metrics, "reward_mean")
    last_winrate = _last_valid(metrics, "win_rate")
    last_lr = _last_valid(metrics, "learning_rate")
    last_ep_len = _last_valid(metrics, "ep_len_mean")

    stages = run_config.get("curriculum", {}).get("stages", [])
    solo_only = bool(stages) and all(s.get("n_opponents", 0) == 0 for s in stages)

    lines = [
        f"# Trainingsreport - Meilenstein {milestone:,} Schritte".replace(",", "."),
        "",
        f"- **Aktuelles Elo (beste Version)**: {best_elo:.0f}" if best_elo is not None else "- **Aktuelles Elo**: n/a",
        f"- **Beste Version**: {best.name}" if best is not None else "- **Beste Version**: n/a",
        f"- **Reward (Mittelwert, letzte Episoden)**: {last_reward:.3f}" if last_reward is not None else "- **Reward**: n/a",
        f"- **Überlebenszeit (letzte Episoden)**: {last_ep_len / 60.0:.1f} s ({last_ep_len:.0f} Ticks)"
        if last_ep_len is not None
        else "- **Überlebenszeit**: n/a",
        f"- **Win Rate (letzte Episoden)**: {last_winrate * 100:.1f}%" if last_winrate is not None else "- **Win Rate**: n/a",
        f"- **Learning Rate (aktuell)**: {last_lr}" if last_lr is not None else "- **Learning Rate**: n/a",
    ]
    if solo_only:
        lines.append(
            "- _Hinweis: In dieser Phase sind keine Gegner konfiguriert (`n_opponents: 0`) - die Win Rate ist daher"
            " strukturell immer 0%. Maßgeblich ist hier die Überlebenszeit._"
        )
    lines += [
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


def main() -> None:
    """Milestone-Reports NACHTRÄGLICH aus einer bestehenden metrics.csv erzeugen -
    ohne Training. Damit lassen sich Meilensteine generieren, die ein bereits
    beendeter Lauf verpasst hat (z. B. weil sie damals nicht in MILESTONES standen):

        python -m ai.reporting.report --run-dir /content/drive/.../phase1 --milestones 2000000 3000000

    --run-dir erwartet die Standard-Struktur (metrics/metrics.csv, reports/,
    config_used.yaml); Einzelpfade lassen sich mit --metrics/--report-dir/--config
    übersteuern. Liegt --run-dir auf Drive, landen die Reports direkt auf Drive.
    """
    import argparse

    parser = argparse.ArgumentParser(description="(Re-)generate milestone training reports from an existing metrics.csv.")
    parser.add_argument("--run-dir", default=None, help="Run-Verzeichnis mit metrics/metrics.csv, reports/ und config_used.yaml")
    parser.add_argument("--metrics", default=None, help="expliziter Pfad zur metrics.csv (überstimmt --run-dir)")
    parser.add_argument("--report-dir", default=None, help="expliziter Report-Ordner (überstimmt --run-dir)")
    parser.add_argument("--config", default=None, help="YAML mit den Trainingsparametern (Default: <run-dir>/config_used.yaml)")
    parser.add_argument("--milestones", type=int, nargs="+", required=True, help="Schritt-Meilensteine, z. B. 2000000 3000000")
    parser.add_argument("--force", action="store_true", help="vorhandene milestone_<N>-Ordner überschreiben")
    args = parser.parse_args()

    if args.run_dir is None and (args.metrics is None or args.report_dir is None):
        parser.error("entweder --run-dir angeben oder --metrics UND --report-dir")
    run_dir = Path(args.run_dir) if args.run_dir else None
    metrics_csv = Path(args.metrics) if args.metrics else run_dir / "metrics" / "metrics.csv"
    report_dir = Path(args.report_dir) if args.report_dir else run_dir / "reports"
    config_path = Path(args.config) if args.config else (run_dir / "config_used.yaml" if run_dir else None)
    run_config = yaml.safe_load(config_path.read_text()) if (config_path is not None and config_path.exists()) else {}
    if not metrics_csv.exists():
        raise SystemExit(f"metrics.csv nicht gefunden: {metrics_csv}")

    for milestone in sorted(args.milestones):
        if (report_dir / f"milestone_{milestone}" / "report.md").exists() and not args.force:
            print(f"[report] milestone {milestone} existiert bereits - übersprungen (--force zum Überschreiben)")
            continue
        path = generate_report(metrics_csv, report_dir, milestone=milestone, run_config=run_config, league=None)
        print(f"[report] geschrieben: {path}")


if __name__ == "__main__":
    main()
