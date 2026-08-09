"""Der trainierte Agent in der ECHTEN Engine - die Brueckezurueck aus dem Traum.

Ohne diese Datei ist alles davor unbeweisbar. Ein Weltmodell kann seine eigene
Fantasie perfekt vorhersagen und ein Actor darin brillieren, ohne dass irgendetwas
davon im Spiel funktioniert. Erst der Vergleich gegen `rules_bot` und den
Beam-Planer auf demselben Spielfeld sagt, ob v1_1 etwas taugt.

**Warum der Agent zustandsbehaftet ist.** Anders als die PPO-Policy in v1_0, die
aus einem 4er-Frame-Stack heraus jedes Mal neu entscheidet, traegt der Dreamer
einen rekurrenten Zustand ueber die ganze Episode. Jeder Tick ist ein
`obs_step`: beobachten, Zustand fortschreiben, Aktion daraus ziehen. `reset()`
MUSS zwischen Episoden aufgerufen werden - ein uebriggebliebener Zustand aus der
letzten Runde ist genau die Sorte Fehler, die sich als "der Agent spielt am
Anfang komisch" aeussert und schwer zu finden ist.

**Deterministisch oder gesampelt.** Beim Bewerten wird per Default die
wahrscheinlichste Aktion genommen (`sample=False`). Die Stichprobe gehoerte ins
Training; beim Messen fuegt sie nur Varianz hinzu, die man dann faelschlich fuer
Schwaeche der Policy haelt.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ai.core.env import renderer, sensors
from ai.core.env.engine import CurveEngine
from ai.core.env.observation import ObsConfig, ObservationBuilder
from ai.core.env.opponents import Controller
from ai.v1_1.models.actor_critic import Actor, ActorCriticConfig
from ai.v1_1.models.world_model import WorldModel, WorldModelConfig


class DreamerAgent:
    """Weltmodell + Actor, mit ueber die Episode fortgeschriebenem RSSM-Zustand."""

    def __init__(self, world_model: WorldModel, actor: Actor, obs_cfg: ObsConfig, device=None):
        self.device = device or torch.device("cpu")
        self.world_model = world_model.to(self.device).eval()
        self.actor = actor.to(self.device).eval()
        self.obs_cfg = obs_cfg
        self.builder = ObservationBuilder(obs_cfg)
        self._state = None
        self._prev_action = None

    @classmethod
    def load(cls, policy_path: Path | str, world_model_path: Path | str, obs_cfg: ObsConfig, device=None) -> "DreamerAgent":
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        wm_state = torch.load(world_model_path, map_location=device, weights_only=False)
        wm_cfg = dict(wm_state["cfg"])
        wm_cfg["image_shape"] = tuple(wm_cfg["image_shape"])
        world_model = WorldModel(WorldModelConfig(**wm_cfg))
        world_model.load_state_dict(wm_state["model"])

        pol = torch.load(policy_path, map_location=device, weights_only=False)
        ac_cfg = ActorCriticConfig(**pol["ac_cfg"]) if "ac_cfg" in pol else ActorCriticConfig()
        actor = Actor(world_model.feat_dim, ac_cfg)
        actor.load_state_dict(pol["actor"])

        # Die Beobachtungsform muss zu der passen, mit der trainiert wurde -
        # sonst laeuft der Encoder auf einer anderen Vektorlaenge und liefert
        # stillen Unsinn statt eines Fehlers.
        if obs_cfg.frame_stack != 1:
            raise ValueError(f"v1_1 arbeitet ohne Frame-Stack (der RSSM ist rekurrent), war frame_stack={obs_cfg.frame_stack}")
        return cls(world_model, actor, obs_cfg, device)

    # ------------------------------------------------------------------

    def reset(self, seat_name: str | None = None) -> None:
        self._state = None
        self._prev_action = None

    @torch.no_grad()
    def act(self, engine: CurveEngine, name: str, frame_hwc: np.ndarray | None = None, sample: bool = False) -> int:
        frame = frame_hwc if frame_hwc is not None else renderer.render_frame(engine, self.obs_cfg.obs_resolution)
        vector = self.builder.observe(engine, name)["vector"]

        image = torch.from_numpy(np.array(frame, copy=True)).to(self.device)[None, None]
        vec = torch.from_numpy(np.asarray(vector, dtype=np.float32)).to(self.device)[None, None]
        action_dim = self.actor.cfg.action_dim
        prev = (
            torch.zeros(1, action_dim, device=self.device)
            if self._prev_action is None
            else torch.eye(action_dim, device=self.device)[self._prev_action][None]
        )

        embed = self.world_model.encode(image, vec)[:, 0]
        if self._state is None:
            self._state = self.world_model.rssm.initial(1, self.device)
        post, _ = self.world_model.rssm.obs_step(self._state, prev, embed)
        self._state = post

        logits = self.actor.logits(post.feature)
        idx = int(torch.distributions.Categorical(logits=logits).sample()) if sample else int(logits.argmax(-1))
        self._prev_action = idx
        return idx


class DreamerController(Controller):
    """Adapter fuer das Controller-Interface, damit der Agent jeden Sitz besetzen
    und in der bestehenden Arena gegen v1_0-Modelle antreten kann."""

    def __init__(self, agent: DreamerAgent, sample: bool = False):
        self.agent = agent
        self.sample = sample

    def reset(self, seat_name: str) -> None:
        self.agent.reset(seat_name)

    def act(self, engine: CurveEngine, name: str, frame_hwc: np.ndarray | None) -> int:
        return self.agent.act(engine, name, frame_hwc, sample=self.sample)
