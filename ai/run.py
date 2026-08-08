"""Der EINE Einstiegspunkt. Alles, was ein Lauf braucht, geht hier durch.

    python -m ai.run train --version v1_0 --config ai/v1_0/config/phase1.yaml
    python -m ai.run train --version v1_1 --config ai/v1_1/config/dreamer.yaml
    python -m ai.run collect --version v1_1 --config ai/v1_1/config/dreamer.yaml --episodes 200
    python -m ai.run info

Warum ueberhaupt eine gemeinsame Schicht ueber `ai.v1_0.training.train` und
`ai.v1_1.training.*`: alles, was NICHT vom Algorithmus abhaengt, soll genau
einmal existieren und fuer beide Versionen identisch sein - Laufname und
Ausgabeverzeichnis (`ai/output/<version>/<run>/`), Hardware-Snapshot,
Phasen-Zeitlogging, das Wegschreiben der tatsaechlich benutzten Config und die
Fehlerbehandlung. Sonst driften zwei Trainingsskripte auseinander und ein
Vergleich "PPO gegen Dreamer" misst am Ende Buchhaltungsunterschiede statt
Algorithmen.

`--run-name auto` (Default) greift den juengsten Lauf der Version wieder auf,
statt einen zweiten danebenzulegen. Zusammen mit `--resume auto` ist derselbe
Befehl nach einem Absturz einfach nochmal aufrufbar - genau das, was ein
`while true`-Wrapper im tmux braucht (scripts/train.sh).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import yaml

from ai.core.utils import sysinfo
from ai.core.utils.paths import OUTPUT_ROOT, latest_run, run_paths
from ai.core.utils.timing import PhaseTimer

VERSIONS = ("v1_0", "v1_1")


def _resolve_run_name(version: str, requested: str | None, output_root: Path | None) -> tuple[str, bool]:
    """(Laufname, bereits_vorhanden). `auto` = juengsten Lauf fortsetzen, sonst neu."""
    if requested and requested != "auto":
        existing = run_paths(version, requested, output_root).root.exists()
        return requested, existing
    prev = latest_run(version, output_root)
    if prev is not None:
        return prev.run_name, True
    return f"{version}_{time.strftime('%Y%m%d_%H%M%S')}", False


def cmd_train(args: argparse.Namespace) -> int:
    version = args.version
    output_root = Path(args.output_root) if args.output_root else None
    run_name, existed = _resolve_run_name(version, args.run_name, output_root)
    paths = run_paths(version, run_name, output_root).mkdirs()

    print(f"[run] Version {version} | Lauf {run_name} | {'fortgesetzt' if existed else 'neu'}")
    print(f"[run] Ausgabe: {paths.root}")
    print(sysinfo.summary_line())

    with PhaseTimer(paths.timing, run_label=f"{version}/{run_name}") as timer:
        with timer.phase("setup"):
            cfg = yaml.safe_load(Path(args.config).read_text())
            if args.timesteps is not None:
                cfg["total_timesteps"] = args.timesteps
            if args.n_envs is not None:
                cfg["n_envs"] = args.n_envs
            paths.config_used.write_text(yaml.safe_dump(cfg, sort_keys=False))
            (paths.timing / "sysinfo.json").write_text(__import__("json").dumps(sysinfo.collect(), indent=2))
            # Import erst hier: v1_1 zieht Abhaengigkeiten, die ein reiner
            # v1_0-Lauf nicht braucht (und umgekehrt) - ein fehlendes Paket der
            # einen Version darf die andere nicht blockieren.
            if version == "v1_0":
                from ai.v1_0.entry import train as train_fn
            else:
                from ai.v1_1.entry import train as train_fn

        result = train_fn(cfg=cfg, paths=paths, timer=timer, resume=args.resume, init_from=args.init_from)

    print(f"[run] fertig -> {result}")
    print(f"[run] Timing: {paths.timing / 'timing.md'}")
    return 0


def cmd_collect(args: argparse.Namespace) -> int:
    if args.version != "v1_1":
        print("[run] 'collect' gibt es nur fuer v1_1 (Planer-Datensammlung).", file=sys.stderr)
        return 2
    from ai.v1_1.data.collect import collect

    out = Path(args.out) if args.out else (OUTPUT_ROOT / "v1_1" / "dataset")
    collect(args.config, out, episodes=args.episodes, seed=args.seed, worker_id=args.worker_id)
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    import json

    print(json.dumps(sysinfo.collect(), indent=2))
    print(f"\nAusgabewurzel: {OUTPUT_ROOT}")
    for v in VERSIONS:
        vdir = OUTPUT_ROOT / v
        runs = sorted([d.name for d in vdir.iterdir() if d.is_dir()]) if vdir.is_dir() else []
        print(f"  {v}: {len(runs)} Laeufe" + (f" (juengster: {latest_run(v).run_name})" if runs else ""))
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="ai.run", description="Trainingseinstieg fuer alle Versionen")
    sub = ap.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--version", choices=VERSIONS, required=True)
        p.add_argument("--config", required=True)
        p.add_argument("--output-root", default=None, help="ueberschreibt ai/output (z.B. ein persistentes Volume)")

    t = sub.add_parser("train", help="Training starten oder fortsetzen")
    add_common(t)
    t.add_argument("--run-name", default="auto", help="'auto' = juengsten Lauf der Version fortsetzen")
    t.add_argument("--resume", default="auto", help="'auto' | Pfad zu einem Checkpoint | 'none'")
    t.add_argument("--init-from", default=None, help="Gewichte warm starten (anderer Lauf/andere Phase)")
    t.add_argument("--timesteps", type=int, default=None)
    t.add_argument("--n-envs", type=int, default=None)
    t.set_defaults(func=cmd_train)

    c = sub.add_parser("collect", help="Expertentrajektorien sammeln (nur v1_1)")
    add_common(c)
    c.add_argument("--out", default=None)
    c.add_argument("--episodes", type=int, default=100)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--worker-id", type=int, default=0)
    c.set_defaults(func=cmd_collect)

    i = sub.add_parser("info", help="Hardware und vorhandene Laeufe anzeigen")
    i.set_defaults(func=cmd_info)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
