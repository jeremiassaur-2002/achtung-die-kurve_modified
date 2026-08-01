"""One place that knows which RL algorithms exist, how to construct them from a
phase YAML, and how to load a saved zip back with the RIGHT class.

Supported (config key `algo`):
  ppo            - stable-baselines3 PPO (the default; unchanged behavior)
  maskable_ppo   - sb3-contrib MaskablePPO (auto-selected when use_action_masking)
  recurrent_ppo  - sb3-contrib RecurrentPPO with an LSTM (MultiInputLstmPolicy):
                   gives the policy real memory instead of only frame_stack's
                   4-tick window - relevant for "where did I come from / where is
                   my own line" once the trail leaves the stacked frames.
  qrdqn          - sb3-contrib QR-DQN: off-policy + replay buffer, i.e. every
                   simulated tick is reused many times. On a Colab budget where
                   wall time (not environment speed) is the bottleneck, that
                   sample reuse is the whole point. Discrete(3) actions fit DQN
                   exactly. Caveat: the replay buffer stores raw uint8 frame
                   stacks, so buffer_size is a RAM knob, not a free parameter -
                   build_algo_kwargs prints the estimate at startup.

Deliberately NOT in this list: switching to CleanRL / sample-factory / RLlib.
Those are whole-framework moves (new training loop, new checkpoint format, no
curriculum/league/resume compatibility) - what they'd buy is instead brought
into THIS pipeline: off-policy sample reuse (qrdqn), recurrence (recurrent_ppo),
and a faster engine (see engine._stamp_segment).
"""

from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

ALGO_CHOICES = ("ppo", "maskable_ppo", "recurrent_ppo", "qrdqn")


@dataclass(frozen=True)
class AlgoSpec:
    name: str            # one of ALGO_CHOICES
    policy_name: str     # SB3 policy alias for Dict observations
    is_recurrent: bool
    is_off_policy: bool

    @property
    def cls(self):
        return _algo_class(self.name)


def _algo_class(name: str):
    if name == "ppo":
        from stable_baselines3 import PPO

        return PPO
    if name == "maskable_ppo":
        from sb3_contrib import MaskablePPO

        return MaskablePPO
    if name == "recurrent_ppo":
        from sb3_contrib import RecurrentPPO

        return RecurrentPPO
    if name == "qrdqn":
        from sb3_contrib import QRDQN

        return QRDQN
    raise ValueError(f"unknown algo {name!r}, expected one of {ALGO_CHOICES}")


def resolve_algo(cfg: dict) -> AlgoSpec:
    """Maps a phase config to an AlgoSpec. `use_action_masking: true` implies
    MaskablePPO (as before); combining masking with recurrent_ppo/qrdqn is
    rejected loudly instead of silently ignoring the mask - curve_env's
    action_masks() hook is only wired up for the (Maskable)PPO family."""
    name = str(cfg.get("algo", "ppo")).lower()
    if name not in ALGO_CHOICES:
        raise SystemExit(f"[algo] unbekanntes algo: {name!r} - erlaubt: {', '.join(ALGO_CHOICES)}")
    if cfg.get("use_action_masking", False):
        if name in ("ppo", "maskable_ppo"):
            name = "maskable_ppo"
        else:
            raise SystemExit(
                f"[algo] use_action_masking: true funktioniert nur mit algo: ppo/maskable_ppo, nicht mit {name!r} "
                f"(RecurrentPPO/QR-DQN haben in SB3 keinen Masking-Pfad). Entweder Masking ausschalten oder algo wechseln."
            )
    return AlgoSpec(
        name=name,
        policy_name="MultiInputLstmPolicy" if name == "recurrent_ppo" else "MultiInputPolicy",
        is_recurrent=name == "recurrent_ppo",
        is_off_policy=name == "qrdqn",
    )


_QRDQN_DEFAULTS = dict(
    learning_rate=1e-4,
    buffer_size=25_000,
    learning_starts=5_000,
    batch_size=64,
    train_freq=4,
    gradient_steps=1,
    target_update_interval=2_000,
    exploration_fraction=0.2,
    exploration_final_eps=0.05,
    gamma=0.99,
)


def build_algo_kwargs(cfg: dict, spec: AlgoSpec, obs_shape_image: tuple[int, ...] | None = None) -> dict:
    """Constructor kwargs for spec.cls, read from the phase YAML.
    PPO family: the existing `ppo:` block (RecurrentPPO shares it; an optional
    `recurrent_ppo:` block overrides individual keys). QR-DQN: `qrdqn:` block on
    top of safe defaults - and because the replay buffer stores TWO uint8 frame
    stacks per transition (obs + next_obs), the RAM footprint is printed so a
    Colab session doesn't die 20 minutes in with a silent OOM."""
    if spec.name == "qrdqn":
        kwargs = dict(_QRDQN_DEFAULTS)
        kwargs["gamma"] = cfg.get("ppo", {}).get("gamma", kwargs["gamma"])  # keep shaping_gamma alignment
        kwargs.update(cfg.get("qrdqn", {}))
        if obs_shape_image is not None:
            import numpy as np

            bytes_per = 2 * int(np.prod(obs_shape_image))  # obs + next_obs, uint8
            gb = kwargs["buffer_size"] * bytes_per / 1024**3
            print(
                f"[algo] QR-DQN Replay-Buffer: {kwargs['buffer_size']:,} Transitionen x {bytes_per / 1024:.0f} KB "
                f"(Bild {obs_shape_image} als obs+next_obs, uint8) ~= {gb:.1f} GB RAM"
            )
        return kwargs

    kwargs = dict(cfg.get("ppo", {}))
    if spec.name == "recurrent_ppo":
        kwargs.update(cfg.get("recurrent_ppo", {}))
    return kwargs


def sniff_algo_name(path: str | Path) -> str | None:
    """Reads the SB3 zip's `data` JSON and identifies which algorithm family
    saved it, via the serialized policy_class string. None = unrecognized."""
    try:
        with zipfile.ZipFile(path) as zf:
            with zf.open("data") as f:
                raw = io.TextIOWrapper(f, encoding="utf-8").read()
        blob = json.loads(raw)
        policy = json.dumps(blob.get("policy_class", ""))
        if "Recurrent" in policy or "Lstm" in policy:
            return "recurrent_ppo"
        if "Maskable" in policy:
            return "maskable_ppo"
        if "QRDQN" in policy or "Quantile" in policy or "qrdqn" in policy:
            return "qrdqn"
        if "ActorCriticPolicy" in policy or "ActorCritic" in policy:
            return "ppo"
    except Exception:
        return None
    return None


def load_trained(path: str | Path, env=None, device: str = "auto", **load_kwargs):
    """Loads a saved model with the correct algorithm class: zip sniffing first,
    then a try-all fallback chain. Replaces every hardcoded `PPO.load(...)` so
    self-play pools, the league and --resume/--init-from keep working no matter
    which of the four algorithms produced the checkpoint."""
    sniffed = sniff_algo_name(path)
    order = [n for n in ([sniffed] if sniffed else []) + list(ALGO_CHOICES) if n]
    seen: list[str] = []
    last_exc: Exception | None = None
    for name in order:
        if name in seen:
            continue
        seen.append(name)
        try:
            return _algo_class(name).load(path, env=env, device=device, **load_kwargs)
        except Exception as exc:  # wrong class, incompatible spaces, ... - try the next
            last_exc = exc
    raise RuntimeError(f"Konnte {path} mit keiner Algorithmusklasse laden ({ALGO_CHOICES}): {last_exc!r}")
