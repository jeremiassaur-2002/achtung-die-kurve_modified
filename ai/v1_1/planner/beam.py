"""Beam-Search-Pfadplaner: sucht die Aktionsfolge, die am laengsten ueberlebt.

Rolle in v1_1: das ist der "Pfadplanungsalgorithmus", der ohne jedes Training
schon spielen kann. Er dient hier drei Zwecken:

1. **Datenquelle fuer statisches (offline) Training.** Der Planer erzeugt
   Trajektorien, aus denen der Dreamer sein Weltmodell und - per Verhaltens-
   klonung - seinen Actor lernen kann, ohne dass das Netz je selbst spielen
   muss. Genau das war der Wunsch "das kann man vielleicht auch statisch
   trainieren".
2. **Obergrenze.** Er zeigt, was reine Suche mit perfekter Kenntnis des
   aktuellen Gitters erreicht. Ein gelerntes Modell, das darunter bleibt, hat
   noch Luft; eines das darueber kommt, hat etwas gelernt, das Suche nicht kann
   (Gegnerverhalten antizipieren).
3. **Gegner** in beiden Versionen - staerker und vor allem *anders* als der
   bestehende `rules_bot`.

**Warum Beam und nicht MCTS.** Der Aktionsraum ist 3 breit, die Belohnung fast
deterministisch, und der Zustand ist voll beobachtbar. Der teure Teil ist nicht
die Suchstrategie, sondern das Vorwaertssimulieren gegen das Gitter - und das
laesst sich fuer Hunderte Kandidaten in EINEM numpy-Gather erledigen
(`sensors.simulate_deltas`). MCTS mit seinen sequentiellen Einzel-Rollouts
verschenkt genau diese Vektorisierung. Ein breiter, flacher Beam ueber
Blockaktionen nutzt sie voll aus.

**Blockaktionen.** Jeden Tick frei zu waehlen waere 3^60 Sequenzen. Stattdessen
wird eine Aktion fuer `commit_ticks` Ticks festgehalten; ein Horizont von 60
Ticks bei commit_ticks=6 ergibt 10 Bloecke, und pro Erweiterungsschritt bleiben
nur die besten `beam_width` Praefixe stehen. Der Verlust ist gering, weil
Lenken in diesem Spiel ohnehin traege ist: eine Einzeltick-Korrektur aendert die
Bahn kaum, gefahren wird in Boegen.

**Receding Horizon.** Ausgefuehrt wird immer nur die erste Aktion, danach wird
neu geplant. Das ist noetig, weil `simulate_deltas` das Gitter einfriert -
Gegner zeichnen waehrend des Horizonts weiter. Ein einmal geplanter Pfad, blind
zu Ende gefahren, waere darum nicht sicher.

Bewertung eines Kandidaten (lexikographisch, als ein Skalar kodiert):
  1. ueberlebte Ticks (gedeckelt beim Horizont) - alles andere ist nachrangig
  2. Freiraum am Endpunkt (`sensors.clearance_at`) - trennt "ueberlebt knapp in
     einer Sackgasse" von "ueberlebt im offenen Feld"
  3. Geradeaus-Bonus - bricht Gleichstaende zugunsten ruhiger Bahnen, statt
     zwischen gleichwertigen Links/Rechts-Optionen zu zittern
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from ai.core.env import sensors
from ai.core.env.engine import CurveEngine, STRAIGHT, TURN_LEFT, TURN_RIGHT
from ai.core.env.opponents import Controller

_ACTIONS = (TURN_LEFT, STRAIGHT, TURN_RIGHT)
_DELTA = {TURN_LEFT: -1.0, STRAIGHT: 0.0, TURN_RIGHT: 1.0}


@dataclass
class PlannerConfig:
    horizon: int = 60
    commit_ticks: int = 6
    beam_width: int = 12
    # Freiraum-Bewertung am Endpunkt
    clearance_rays: int = 12
    clearance_range_px: float = 48.0
    clearance_weight: float = 0.5  # in "Tick"-Einheiten; < 1, damit Ueberleben dominiert
    straight_bonus: float = 0.05  # Gleichstandsbrecher
    # Epsilon-Rauschen: fuer die Datensammlung erwuenscht (ein rein
    # deterministischer Experte erzeugt einen Datensatz ohne jede Streuung, aus
    # dem ein Weltmodell die Konsequenzen von Fehlern nie lernt)
    epsilon: float = 0.0

    def __post_init__(self) -> None:
        if self.horizon < 1 or self.commit_ticks < 1:
            raise ValueError("horizon und commit_ticks muessen >= 1 sein")
        if self.beam_width < 1:
            raise ValueError("beam_width muss >= 1 sein")


@dataclass
class PlanResult:
    action: int
    survived: int  # Ticks, die die beste Sequenz ueberlebt (== horizon: kein Tod in Sicht)
    score: float
    n_evaluated: int  # ausgewertete Kandidaten - fuer Kostenabschaetzung im Timing


class BeamPlanner:
    """Zustandslos zwischen den Ticks; nur `cfg` und ein RNG fuer Epsilon."""

    def __init__(self, cfg: PlannerConfig | None = None, rng: random.Random | None = None):
        self.cfg = cfg or PlannerConfig()
        self.rng = rng or random.Random()

    # ---------------- Kern ----------------

    def plan(self, engine: CurveEngine, name: str) -> PlanResult:
        cfg = self.cfg
        n_blocks = max(1, -(-cfg.horizon // cfg.commit_ticks))  # ceil

        # Ein Beam-Eintrag ist eine Folge von Blockaktionen. `beam` haelt die
        # Praefixe als (B, gefuellte_ticks) Delta-Matrix; erweitert wird um je
        # einen Block pro Aktion, danach wird auf beam_width gestutzt.
        beam = np.zeros((1, 0), dtype=np.float32)
        beam_first: list[int] = [STRAIGHT]  # erste Aktion je Beam-Eintrag
        best: PlanResult | None = None
        n_eval = 0

        for block in range(n_blocks):
            ticks = min(cfg.commit_ticks, cfg.horizon - block * cfg.commit_ticks)
            n_prefix = beam.shape[0]

            # kartesisches Produkt Praefix x 3 Aktionen
            cand = np.repeat(beam, 3, axis=0)
            new_block = np.tile(np.array([[_DELTA[a]] * ticks for a in _ACTIONS], dtype=np.float32), (n_prefix, 1))
            cand = np.concatenate([cand, new_block], axis=1)
            cand_first = (
                [a for a in _ACTIONS] if block == 0 else [f for f in beam_first for _ in _ACTIONS]
            )

            positions, survived = sensors.simulate_deltas(engine, name, cand)
            n_eval += cand.shape[0]
            filled = cand.shape[1]

            # Endpunkt = letzte ueberlebte Position (bei Tod die davor); dort wird
            # der Freiraum gemessen. Ein toter Kandidat bekommt so keinen
            # Freiraum-Bonus fuer einen Punkt, den er nie erreicht.
            end_idx = np.clip(survived - 1, 0, filled - 1)
            endpoints = positions[np.arange(cand.shape[0]), end_idx]
            alive = survived >= filled
            clearance = np.zeros(cand.shape[0], dtype=np.float32)
            if alive.any():
                clearance[alive] = sensors.clearance_at(
                    engine, name, endpoints[alive], cfg.clearance_rays, cfg.clearance_range_px
                )

            straightness = 1.0 - np.abs(cand).mean(axis=1)
            score = (
                survived.astype(np.float32)
                + cfg.clearance_weight * clearance * cfg.commit_ticks
                + cfg.straight_bonus * straightness
            )

            order = np.argsort(-score)
            top = int(order[0])
            if best is None or score[top] > best.score:
                best = PlanResult(
                    action=cand_first[top], survived=int(survived[top]), score=float(score[top]), n_evaluated=n_eval
                )

            # Nur Kandidaten weiterverfolgen, die den bisherigen Block ueberlebt
            # haben - ein toter Praefix kann durch keine Fortsetzung wieder
            # lebendig werden. Sind alle tot, ist der Zustand verloren; dann
            # bleibt der bestbewertete stehen (er stirbt am spaetesten).
            keep = order[alive[order]][: cfg.beam_width]
            if keep.size == 0:
                break
            beam = cand[keep]
            beam_first = [cand_first[int(i)] for i in keep]

        assert best is not None
        if cfg.epsilon > 0.0 and self.rng.random() < cfg.epsilon:
            best = PlanResult(
                action=self.rng.choice(_ACTIONS), survived=best.survived, score=best.score, n_evaluated=best.n_evaluated
            )
        return best

    def act(self, engine: CurveEngine, name: str) -> int:
        return self.plan(engine, name).action


class PlannerController(Controller):
    """Adapter, damit der Planer jeden Sitz in `CurveEnv` besetzen kann - als
    Gegner im Curriculum oder als Experte bei der Datensammlung."""

    def __init__(self, cfg: PlannerConfig | None = None, rng: random.Random | None = None):
        self.planner = BeamPlanner(cfg, rng)

    def reset(self, seat_name: str) -> None:
        pass

    def act(self, engine: CurveEngine, name: str, frame_hwc: np.ndarray | None) -> int:
        return self.planner.act(engine, name)
