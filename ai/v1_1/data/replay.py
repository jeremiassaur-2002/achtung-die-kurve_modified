"""Sequenz-Replay ueber die vom Planer gesammelten Shards.

Das Weltmodell lernt nicht aus Einzelbildern, sondern aus SEQUENZEN: der RSSM
rollt seinen rekurrenten Zustand ueber `batch_length` Ticks aus und wird darauf
trainiert, den jeweils naechsten Schritt vorherzusagen. Ein Batch ist deshalb
(B, L, ...) und nicht (B, ...).

**Ladestrategie.** Die Shards komprimieren auf ~0,2 KB/Tick; ein Datensatz mit
200k Ticks sind rund 40 MB gepackt, entpackt (uint8-Frames) etwa 2,5 GB. Das
passt bei kleinen Datensaetzen komplett in den RAM, bei grossen nicht mehr.
Deshalb ein LRU-Cache ueber Episoden statt "alles laden" oder "jedes Mal von
Platte": im Steady State werden immer wieder dieselben Episoden gesampelt, und
`np.load` auf einem komprimierten npz ist teuer genug (Dekompression), dass es
den Trainingsschritt sonst dominiert.

**Episodengrenzen.** Eine gesampelte Sequenz liegt IMMER vollstaendig innerhalb
einer Episode. Ueber einen Episodenwechsel hinweg zu sampeln waere ein
Kunstfehler: der RSSM wuerde lernen, dass auf einen Aufprall unmittelbar ein
frisch zurueckgesetztes Spielfeld folgt - eine Dynamik, die es im Spiel nicht
gibt und die die Fortsetzungs-Vorhersage vergiftet.

**Kurze Episoden.** Wer stirbt, stirbt frueh; viele Episoden sind kuerzer als
`batch_length`. Die einfach zu ueberspringen wuerde ausgerechnet die Todesfaelle
aus dem Datensatz filtern - also genau das, was das Weltmodell lernen soll. Sie
werden stattdessen am Ende mit `is_padding=False` maskiert aufgefuellt, und der
Verlust ignoriert die Padding-Schritte.
"""

from __future__ import annotations

import json
import random
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

N_ACTIONS = 3


@dataclass
class Batch:
    """(B, L, ...) - alles float32/bool, bereit fuer torch.from_numpy."""

    image: np.ndarray  # (B, L, H, W, 3) uint8
    vector: np.ndarray  # (B, L, V) float32
    action: np.ndarray  # (B, L, A) float32, one-hot
    reward: np.ndarray  # (B, L) float32
    is_first: np.ndarray  # (B, L) bool - hier muss der RSSM-Zustand zurueckgesetzt werden
    is_terminal: np.ndarray  # (B, L) bool - echter Tod (Fortsetzungsziel = 0)
    mask: np.ndarray  # (B, L) float32 - 0 fuer Padding, 1 sonst
    ticks_to_death: np.ndarray  # (B, L) float32 - Restlebensdauer, gedeckelt

    def __len__(self) -> int:
        return self.image.shape[0]


class ShardReplay:
    def __init__(
        self,
        dataset_dir: Path | str,
        cache_episodes: int = 64,
        seed: int = 0,
        terminal_fraction: float = 0.25,
        ttd_cap: float = 120.0,
        split: str = "all",
        val_fraction: float = 0.1,
    ):
        self.dir = Path(dataset_dir)
        shard_dir = self.dir / "shards"
        self.shards = sorted(shard_dir.glob("*.npz")) if shard_dir.is_dir() else []
        if not self.shards:
            raise FileNotFoundError(
                f"Keine Shards unter {shard_dir}. Erst sammeln:\n"
                f"  python -m ai.run collect --version v1_1 --config ai/v1_1/config/dreamer.yaml"
            )
        # Train/Val-Split ueber ganze EPISODEN, nicht ueber Schritte: zwei
        # Fenster derselben Episode ueberlappen sich stark: laege eines in Train
        # und eines in Val, misst die Validierung nur noch Auswendiglernen.
        if split != "all":
            n_val = max(1, int(len(self.shards) * val_fraction))
            val_set = self.shards[::-1][:n_val]
            if split == "val":
                self.shards = sorted(val_set)
            elif split == "train":
                self.shards = [s for s in self.shards if s not in set(val_set)]
            else:
                raise ValueError(f"split muss 'all', 'train' oder 'val' sein, war {split!r}")
            if not self.shards:
                raise ValueError(f"Split {split!r} ist leer - zu wenige Shards fuer val_fraction={val_fraction}")
        self.split = split
        self.rng = random.Random(seed)
        self._cache: OrderedDict[int, dict[str, np.ndarray]] = OrderedDict()
        self._cache_size = max(1, cache_episodes)
        # Laengenindex einmal aufbauen: nur die Header der npz lesen, nicht die
        # Frames dekomprimieren - sonst kostet das Oeffnen so viel wie das
        # Training selbst.
        self._lengths = [self._peek_length(p) for p in self.shards]
        self.total_ticks = int(sum(self._lengths))
        self.ttd_cap = float(ttd_cap)
        self.terminal_fraction = float(terminal_fraction)
        # Episoden, die wirklich mit einem Tod enden (nicht per Zeitlimit
        # abgeschnitten) - aus ihnen wird gezielt nachgesampelt.
        self._fatal = [i for i, p_ in enumerate(self.shards) if self._peek_fatal(p_)]

    @staticmethod
    def _peek_length(path: Path) -> int:
        with np.load(path) as d:
            return int(d["actions"].shape[0])

    @staticmethod
    def _peek_fatal(path: Path) -> bool:
        with np.load(path) as d:
            dones = d["dones"]
            return bool(len(dones) and dones[-1])

    @staticmethod
    def _ticks_to_death(dones: np.ndarray, cap: float) -> np.ndarray:
        """Restlebensdauer je Schritt, rueckwaerts aus der Episode gezaehlt.

        Der Grund, warum es diese Groesse braucht: Terminals machen unter 0,3%
        der Schritte aus. Ein Fortsetzungskopf, der darauf trainiert wird,
        erreicht ueber 99% Trefferquote, indem er immer "geht weiter" sagt - und
        der Actor sieht in der Imagination nie einen Tod. `ticks_to_death` ist
        dieselbe Information, aber DICHT: jeder einzelne Schritt traegt sie.

        Gedeckelt bei `cap`, weil die exakte Restdauer weit vom Tod entfernt
        weder vorhersagbar noch interessant ist - relevant ist die Naehe.
        Endet das Fenster ohne Terminal (Zeitlimit statt Tod), wird ab dort mit
        `cap` aufgefuellt statt ein Ende zu erfinden, das es nicht gab.
        """
        n = dones.shape[0]
        out = np.full(n, cap, dtype=np.float32)
        end = n - 1 if not dones.any() else int(np.argmax(dones))
        died = bool(dones.any())
        if died:
            idx = np.arange(end + 1, dtype=np.float32)
            out[: end + 1] = np.minimum(end - idx, cap)
        return out

    def _episode(self, idx: int) -> dict[str, np.ndarray]:
        cached = self._cache.get(idx)
        if cached is not None:
            self._cache.move_to_end(idx)
            return cached
        with np.load(self.shards[idx]) as d:
            ep = {k: d[k] for k in ("frames", "vectors", "actions", "rewards", "dones")}
        self._cache[idx] = ep
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return ep

    # ------------------------------------------------------------------

    def sample(self, batch_size: int, batch_length: int) -> Batch:
        images, vectors, actions, rewards, firsts, terminals, masks, ttds = [], [], [], [], [], [], [], []
        # Wie viele Sequenzen des Batches sollen GARANTIERT ein Terminal
        # enthalten. Ohne diese Quote sieht das Modell den Tod in etwa jedem
        # achten Fenster - viel zu selten, um ihn vorherzusagen.
        n_forced = int(round(batch_size * self.terminal_fraction)) if self._fatal else 0
        # Laengengewichtete Auswahl: eine 3000-Tick-Episode enthaelt 30x so viele
        # Startpunkte wie eine mit 100 Ticks. Gleichverteilt ueber EPISODEN zu
        # ziehen wuerde kurze (also toedlich endende) Episoden massiv
        # ueberrepraesentieren und das Weltmodell pessimistisch machen.
        weights = self._lengths
        for slot in range(batch_size):
            force_terminal = slot < n_forced
            if force_terminal:
                idx = self.rng.choice(self._fatal)
            else:
                idx = self.rng.choices(range(len(self.shards)), weights=weights, k=1)[0]
            ep = self._episode(idx)
            n = int(ep["actions"].shape[0])
            take = min(batch_length, n)
            if force_terminal:
                # Fenster so legen, dass der letzte Schritt der Episode drin ist.
                start = n - take
            else:
                start = self.rng.randint(0, n - take) if n > take else 0
            sl = slice(start, start + take)

            pad = batch_length - take
            img = ep["frames"][sl]
            vec = ep["vectors"][sl]
            act = np.eye(N_ACTIONS, dtype=np.float32)[ep["actions"][sl].astype(np.int64)]
            rew = ep["rewards"][sl].astype(np.float32)
            # `dones` ist True auch bei Zeitlimit-Abbruch (truncated). Als
            # Terminal zaehlt nur der letzte Schritt einer Episode, die auch
            # wirklich zu Ende gespielt wurde - ein abgeschnittenes Fenster
            # mitten in der Episode hat kein Terminal.
            term = ep["dones"][sl].astype(bool)
            first = np.zeros(take, dtype=bool)
            first[0] = start == 0
            m = np.ones(take, dtype=np.float32)
            ttd = self._ticks_to_death(ep["dones"], self.ttd_cap)[sl]

            if pad > 0:
                img = np.concatenate([img, np.zeros((pad, *img.shape[1:]), img.dtype)])
                vec = np.concatenate([vec, np.zeros((pad, vec.shape[1]), vec.dtype)])
                act = np.concatenate([act, np.zeros((pad, N_ACTIONS), np.float32)])
                rew = np.concatenate([rew, np.zeros(pad, np.float32)])
                term = np.concatenate([term, np.zeros(pad, bool)])
                first = np.concatenate([first, np.zeros(pad, bool)])
                m = np.concatenate([m, np.zeros(pad, np.float32)])
                ttd = np.concatenate([ttd, np.zeros(pad, np.float32)])

            images.append(img)
            vectors.append(vec)
            actions.append(act)
            rewards.append(rew)
            terminals.append(term)
            firsts.append(first)
            masks.append(m)
            ttds.append(ttd)

        return Batch(
            image=np.stack(images),
            vector=np.stack(vectors).astype(np.float32),
            action=np.stack(actions),
            reward=np.stack(rewards),
            is_first=np.stack(firsts),
            is_terminal=np.stack(terminals),
            mask=np.stack(masks),
            ticks_to_death=np.stack(ttds).astype(np.float32),
        )

    # ------------------------------------------------------------------

    def stats(self) -> dict:
        idx_files = sorted(self.dir.glob("index_w*.json"))
        meta = [json.loads(p.read_text()) for p in idx_files]
        return {
            "episodes": len(self.shards),
            "total_ticks": self.total_ticks,
            "mean_episode_ticks": self.total_ticks / max(1, len(self.shards)),
            "fatal_episodes": len(self._fatal),
            "terminal_ratio": len(self._fatal) / max(1, self.total_ticks),
            "split": self.split,
            "min_episode_ticks": int(min(self._lengths)),
            "max_episode_ticks": int(max(self._lengths)),
            "workers": len(meta),
        }

    def shapes(self) -> dict[str, tuple]:
        """Bild-/Vektorform aus dem ersten Shard - das Modell muss seine
        Eingangsgroessen aus den DATEN nehmen, nicht aus einer Config, die
        vielleicht seit dem Sammeln geaendert wurde."""
        ep = self._episode(0)
        return {"image": tuple(ep["frames"].shape[1:]), "vector": (int(ep["vectors"].shape[1]),), "action": (N_ACTIONS,)}
