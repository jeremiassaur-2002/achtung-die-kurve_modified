"""Phasen-Zeitlogging: welcher Abschnitt eines Laufs hat wie lange gedauert.

Motivation: auf einer gemieteten GPU kostet jede Stunde Geld, und die Frage
"wofuer geht die Zeit eigentlich drauf" laesst sich aus TensorBoard nicht
beantworten - dort steht nur der Reward-Verlauf. TensorBoard zeigt das WAS,
diese Datei das WOFUER. Typische Antwort nach dem ersten Lauf: 70% Env-Steps
(CPU-gebunden), 20% Gradient-Updates (GPU), 10% Checkpoints/Video - woraus
direkt folgt, dass eine teurere GPU nichts bringt, mehr CPU-Kerne aber sehr wohl.

Zwei Ebenen:

- `PhaseTimer.phase("setup")` als Context-Manager fuer grobe, einmalige
  Abschnitte (setup, dataset, world_model, policy, export).
- `PhaseTimer.section("env_step")` fuer heisse, oft durchlaufene Abschnitte:
  akkumuliert Gesamtdauer + Aufrufzahl statt tausende Einzeleintraege.

Geschrieben wird nach JEDEM abgeschlossenen Abschnitt (`flush()`), nicht erst
am Ende - ein abgestuerzter Lauf soll seine bis dahin gemessenen Zeiten noch
auf der Platte haben. Ausgabe: timing/timing.json (maschinenlesbar) und
timing/timing.md (die Tabelle, die man tatsaechlich anschaut).
"""

from __future__ import annotations

import json
import platform
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


def _fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m {seconds % 60:04.1f}s"
    hours, rest = divmod(seconds, 3600)
    return f"{int(hours)}h {int(rest // 60):02d}m {rest % 60:04.1f}s"


@dataclass
class PhaseRecord:
    name: str
    seconds: float
    calls: int = 1
    started_at: float = 0.0
    meta: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "seconds": round(self.seconds, 3),
            "human": _fmt_duration(self.seconds),
            "calls": self.calls,
            "started_at": self.started_at,
            **({"meta": self.meta} if self.meta else {}),
        }


class PhaseTimer:
    """Sammelt Phasendauern eines Laufs und schreibt sie fortlaufend weg."""

    def __init__(self, out_dir: Path | str, run_label: str = "", write_on_exit: bool = True):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.run_label = run_label
        self.write_on_exit = write_on_exit
        self.t0 = time.perf_counter()
        self.wall_start = time.time()
        self.phases: list[PhaseRecord] = []
        self._sections: dict[str, PhaseRecord] = {}

    # ---------------- Messung ----------------

    @contextmanager
    def phase(self, name: str, **meta) -> Iterator[PhaseRecord]:
        """Ein einmaliger, benannter Abschnitt. Die Zeit wird auch dann
        festgehalten, wenn der Block mit einer Exception verlassen wird - sonst
        fehlt ausgerechnet fuer den abgestuerzten Abschnitt jede Angabe."""
        rec = PhaseRecord(name=name, seconds=0.0, started_at=time.time(), meta=dict(meta))
        start = time.perf_counter()
        self.phases.append(rec)
        try:
            yield rec
        finally:
            rec.seconds = time.perf_counter() - start
            self.flush()

    @contextmanager
    def section(self, name: str) -> Iterator[None]:
        """Heisser Abschnitt: Dauer und Aufrufzahl werden aufsummiert. Bewusst
        ohne flush() pro Aufruf - bei zehntausenden Aufrufen waere das I/O
        teurer als die gemessene Arbeit."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            rec = self._sections.get(name)
            if rec is None:
                self._sections[name] = PhaseRecord(name=name, seconds=elapsed, calls=1, started_at=time.time())
            else:
                rec.seconds += elapsed
                rec.calls += 1

    def mark(self, name: str, seconds: float, **meta) -> None:
        """Extern gemessene Dauer nachtragen (z.B. eine Zeit, die ein Callback
        selbst gestoppt hat)."""
        self.phases.append(PhaseRecord(name=name, seconds=seconds, started_at=time.time(), meta=dict(meta)))
        self.flush()

    # ---------------- Ausgabe ----------------

    @property
    def total_seconds(self) -> float:
        return time.perf_counter() - self.t0

    def as_dict(self) -> dict:
        accounted = sum(p.seconds for p in self.phases)
        return {
            "run": self.run_label,
            "host": platform.node(),
            "python": platform.python_version(),
            "started_at": self.wall_start,
            "total_seconds": round(self.total_seconds, 3),
            "total_human": _fmt_duration(self.total_seconds),
            "accounted_seconds": round(accounted, 3),
            # Differenz = alles, was in keiner Phase lag (Imports, Wartezeiten).
            # Wenn die gross wird, fehlt eine Messstelle.
            "unaccounted_seconds": round(max(0.0, self.total_seconds - accounted), 3),
            "phases": [p.as_dict() for p in self.phases],
            "sections": [s.as_dict() for s in sorted(self._sections.values(), key=lambda r: -r.seconds)],
        }

    def to_markdown(self) -> str:
        d = self.as_dict()
        lines = [
            f"# Timing - {d['run'] or 'run'}",
            "",
            f"Gesamt: **{d['total_human']}** (nicht zugeordnet: {_fmt_duration(d['unaccounted_seconds'])})",
            "",
            "## Phasen",
            "",
            "| Phase | Dauer | Anteil |",
            "| --- | ---: | ---: |",
        ]
        total = max(d["total_seconds"], 1e-9)
        for p in d["phases"]:
            lines.append(f"| {p['name']} | {p['human']} | {100.0 * p['seconds'] / total:.1f}% |")
        if d["sections"]:
            lines += [
                "",
                "## Abschnitte (kumuliert)",
                "",
                "| Abschnitt | Gesamt | Aufrufe | ø pro Aufruf |",
                "| --- | ---: | ---: | ---: |",
            ]
            for s in d["sections"]:
                avg = s["seconds"] / max(1, s["calls"])
                lines.append(f"| {s['name']} | {s['human']} | {s['calls']:,} | {_fmt_duration(avg)} |")
        return "\n".join(lines) + "\n"

    def flush(self) -> None:
        """Atomar schreiben (temp + rename): ein Crash mitten im Schreiben darf
        keine halbe JSON-Datei hinterlassen, die spaeter niemand mehr parsen kann."""
        for path, text in ((self.out_dir / "timing.json", json.dumps(self.as_dict(), indent=2)), (self.out_dir / "timing.md", self.to_markdown())):
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(text)
            tmp.replace(path)

    def __enter__(self) -> "PhaseTimer":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self.write_on_exit:
            self.flush()
        return False
