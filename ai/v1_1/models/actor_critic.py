"""Actor und Critic - beide arbeiten ausschliesslich auf dem Weltmodell-Zustand.

Sie sehen nie ein Bild und nie einen Sensorwert, nur `feature = [h, flatten(z)]`.
Das ist der ganze Witz des Verfahrens: der Encoder laeuft einmal pro echtem Tick,
das Policy-Training dagegen auf tausenden getraeumten Schritten, in denen gar
kein Bild existiert. Ein Imaginationsschritt kostet deshalb einen Bruchteil eines
Engine-Schritts.

**Critic mit twohot statt Regression.** Dieselbe Ueberlegung wie beim
Belohnungskopf, nur schaerfer: die Renditen hier sind stark schief (lange Ketten
von +0,01, dann einmal -1). Eine Regression zieht ihre Vorhersage zum Mittelwert
und macht die Todesnaehe unsichtbar - genau das Signal, auf das es ankommt.

**Langsamer Ziel-Critic (EMA).** Der Critic lernt gegen Ziele, die er selbst
erzeugt hat. Ohne Entkopplung fuettert sich der Fehler zurueck und die
Wertschaetzung driftet weg. Eine exponentiell gemittelte Kopie als Ziel bricht
diese Rueckkopplung; DreamerV3 nutzt dafuer 0,98 pro Schritt.

**REINFORCE statt Reparametrisierung.** Die Aktionen sind diskret (links,
geradeaus, rechts), durch eine kategoriale Stichprobe fliesst kein sauberer
Pfadgradient. Also Score-Function-Schaetzer mit dem Critic als Baseline plus
Entropie-Bonus gegen zu fruehes Festfahren auf eine Kurvenrichtung.

**Return-Normalisierung ueber Perzentile.** `alive_bonus` 0,01 und
`death_penalty` -1 unterscheiden sich um zwei Groessenordnungen; ohne
Skalierung haengt die brauchbare Lernrate an der Belohnungsskala. Normiert wird
mit der Spannweite zwischen 5. und 95. Perzentil - robuster als Mittelwert und
Standardabweichung, weil einzelne Tode die Statistik sonst dominieren. Der Nenner
wird bei 1 gedeckelt, damit kleine Renditen nicht kuenstlich aufgeblasen werden.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from ai.v1_1.models.nets import TwoHotEncoding, _mlp


@dataclass
class ActorCriticConfig:
    action_dim: int = 3
    hidden_dim: int = 512
    layers: int = 2
    value_bins: int = 255
    imagination_horizon: int = 15
    gamma: float = 0.997
    lambda_: float = 0.95
    entropy_scale: float = 3e-4
    actor_lr: float = 3e-5
    critic_lr: float = 3e-5
    slow_critic_decay: float = 0.98
    return_norm_decay: float = 0.99
    unimix: float = 0.01


class Actor(nn.Module):
    def __init__(self, feat_dim: int, cfg: ActorCriticConfig):
        super().__init__()
        self.cfg = cfg
        self.net = _mlp(feat_dim, cfg.hidden_dim, cfg.action_dim, layers=cfg.layers)
        # Nullinitialisierung: der Actor startet exakt gleichverteilt statt mit
        # einer zufaelligen Vorliebe fuer eine Kurvenrichtung, die er sich in
        # den ersten Schritten selbst verstaerken wuerde.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def logits(self, feat: torch.Tensor) -> torch.Tensor:
        logits = self.net(feat)
        if self.cfg.unimix > 0.0:
            # Keine Aktion darf Wahrscheinlichkeit exakt 0 bekommen - sonst wird
            # log_prob zu -inf und der REINFORCE-Term explodiert.
            probs = torch.softmax(logits, -1)
            probs = (1.0 - self.cfg.unimix) * probs + self.cfg.unimix / self.cfg.action_dim
            logits = torch.log(probs)
        return logits

    def distribution(self, feat: torch.Tensor) -> torch.distributions.Categorical:
        return torch.distributions.Categorical(logits=self.logits(feat))

    def sample(self, feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """-> (one-hot Aktion, log_prob)."""
        dist = self.distribution(feat)
        idx = dist.sample()
        return F.one_hot(idx, self.cfg.action_dim).float(), dist.log_prob(idx)


class Critic(nn.Module):
    def __init__(self, feat_dim: int, cfg: ActorCriticConfig):
        super().__init__()
        self.cfg = cfg
        self.net = _mlp(feat_dim, cfg.hidden_dim, cfg.value_bins, layers=cfg.layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.twohot = TwoHotEncoding(bins=cfg.value_bins)
        # Ziel-Kopie; nimmt am Optimierer NICHT teil.
        self.slow = _mlp(feat_dim, cfg.hidden_dim, cfg.value_bins, layers=cfg.layers)
        self.slow.load_state_dict(self.net.state_dict())
        for p in self.slow.parameters():
            p.requires_grad_(False)

    def logits(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat)

    def value(self, feat: torch.Tensor) -> torch.Tensor:
        self.twohot.to(feat.device)
        return self.twohot.decode(self.net(feat))

    @torch.no_grad()
    def slow_value(self, feat: torch.Tensor) -> torch.Tensor:
        self.twohot.to(feat.device)
        return self.twohot.decode(self.slow(feat))

    @torch.no_grad()
    def update_slow(self) -> None:
        d = self.cfg.slow_critic_decay
        for tgt, src in zip(self.slow.parameters(), self.net.parameters()):
            tgt.mul_(d).add_(src, alpha=1.0 - d)


class ReturnNormalizer:
    """Laufende Perzentilspanne der Renditen, exponentiell geglaettet."""

    def __init__(self, decay: float = 0.99, low: float = 5.0, high: float = 95.0):
        self.decay = decay
        self.low = low
        self.high = high
        self.scale: float | None = None

    def update(self, returns: torch.Tensor) -> float:
        flat = returns.detach().flatten()
        lo = torch.quantile(flat, self.low / 100.0)
        hi = torch.quantile(flat, self.high / 100.0)
        span = float((hi - lo).abs())
        self.scale = span if self.scale is None else self.decay * self.scale + (1 - self.decay) * span
        # Deckel bei 1: bei sehr kleinen Renditen wuerde eine winzige Spanne die
        # Vorteile sonst um Groessenordnungen aufblasen und das Training sprengen.
        return max(1.0, self.scale)


def lambda_return(
    reward: torch.Tensor, value: torch.Tensor, continue_: torch.Tensor, gamma: float, lam: float
) -> torch.Tensor:
    """(B, H) -> (B, H) TD(lambda)-Renditen, rueckwaerts aufgebaut.

    `continue_` in [0,1] ist die vom Weltmodell VORHERGESAGTE Fortsetzungs-
    wahrscheinlichkeit, nicht ein Flag. Im Traum gibt es kein hartes
    Episodenende - das Modell aeussert seine Unsicherheit ueber den Tod als
    Zahl zwischen 0 und 1, und die diskontiert die Zukunft entsprechend.
    """
    horizon = reward.shape[1]
    out = torch.zeros_like(reward)
    # Bootstrap am Horizont: der letzte Wert ist die beste Schaetzung fuer alles
    # danach. Ohne ihn waere die Rendite systematisch zu pessimistisch und der
    # Actor lernte, kurz vor dem Horizont aufzugeben.
    acc = value[:, -1]
    for t in reversed(range(horizon)):
        disc = gamma * continue_[:, t]
        boot = value[:, t + 1] if t + 1 < horizon else value[:, -1]
        acc = reward[:, t] + disc * ((1 - lam) * boot + lam * acc)
        out[:, t] = acc
    return out
