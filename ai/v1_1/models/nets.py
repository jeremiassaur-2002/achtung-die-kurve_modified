"""Netz-Bausteine des Weltmodells: symlog/twohot, Encoder, Decoder, Vorhersagekoepfe.

**symlog.** Belohnungen und Sensorwerte haben sehr unterschiedliche
Groessenordnungen (alive_bonus 0,01 gegen death_penalty -1, Strahlen in [0,1],
Positionen in [-1,1]). Ein MSE-Verlust darauf laesst die groessten Werte alles
dominieren. symlog(x) = sign(x)·log(|x|+1) staucht grosse Betraege, laesst
kleine fast unveraendert und ist - anders als eine laufende Normalisierung -
zustandslos und damit ueber einen Neustart hinweg identisch. Genau der Grund,
warum DreamerV3 ohne datensatzspezifisches Tuning auskommt.

**twohot.** Der Belohnungskopf sagt keine Zahl vorher, sondern eine Verteilung
ueber Stuetzstellen im symlog-Raum, und der Zielwert wird auf die zwei
benachbarten Stuetzstellen verteilt. Warum nicht einfach MSE: Belohnungen sind
hier extrem schief - fast immer +0,01, sehr selten -1. Eine Regression darauf
konvergiert gegen den Mittelwert und sagt den Tod nie voraus. Eine Klassifikation
mit Kreuzentropie hat dieses Problem nicht; die seltene Klasse behaelt Gewicht.

**Kategoriale Latents mit Straight-Through.** Der stochastische Teil des
Zustands sind 32 Gruppen à 32 Klassen statt einer Gaussverteilung. Diskrete
Latents passen zu einem Spiel, dessen Zukunft echte Verzweigungen hat ("Gegner
biegt links oder rechts ab") - eine Gaussverteilung muesste das zu einem
verwaschenen Mittelwert verschmieren. Gradienten fliessen per Straight-Through
durch die Stichprobe hindurch.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


# --------------------------------------------------------------- Transformationen


def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.expm1(torch.abs(x))


class TwoHotEncoding:
    """Stuetzstellen im symlog-Raum + Hin-/Rueckrechnung."""

    def __init__(self, low: float = -20.0, high: float = 20.0, bins: int = 255, device=None):
        self.bins = bins
        self.support = torch.linspace(low, high, bins, device=device)

    def to(self, device) -> "TwoHotEncoding":
        self.support = self.support.to(device)
        return self

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """(...) -> (..., bins): Gewicht auf die zwei umschliessenden Stuetzstellen,
        linear interpoliert. Werte ausserhalb landen vollstaendig auf dem Rand."""
        x = symlog(x).unsqueeze(-1)
        support = self.support.view(*([1] * (x.dim() - 1)), -1)
        below = (support <= x).sum(-1) - 1
        above = self.bins - (support > x).sum(-1)
        below = below.clamp(0, self.bins - 1)
        above = above.clamp(0, self.bins - 1)
        equal = below == above
        d_below = torch.where(equal, torch.ones_like(x.squeeze(-1)), (x.squeeze(-1) - self.support[below]).abs())
        d_above = torch.where(equal, torch.ones_like(x.squeeze(-1)), (self.support[above] - x.squeeze(-1)).abs())
        total = d_below + d_above
        w_below = d_above / total
        w_above = d_below / total
        return (
            F.one_hot(below, self.bins) * w_below.unsqueeze(-1)
            + F.one_hot(above, self.bins) * w_above.unsqueeze(-1)
        ).float()

    def decode(self, logits: torch.Tensor) -> torch.Tensor:
        """(..., bins) Logits -> Erwartungswert, zurueck im Originalraum."""
        probs = torch.softmax(logits, dim=-1)
        return symexp((probs * self.support).sum(-1))

    def loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Kreuzentropie gegen das twohot-Ziel, (...) ohne Reduktion."""
        tgt = self.encode(target).detach()
        return -(tgt * torch.log_softmax(logits, dim=-1)).sum(-1)


# ----------------------------------------------------------------------- Bausteine


def _mlp(in_dim: int, hidden: int, out_dim: int, layers: int = 2) -> nn.Sequential:
    """LayerNorm + SiLU wie in DreamerV3: LayerNorm haelt die Aktivierungen ueber
    lange RSSM-Ausrollungen stabil, wo BatchNorm wegen der Sequenzstruktur
    ausscheidet."""
    mods: list[nn.Module] = []
    d = in_dim
    for _ in range(layers):
        mods += [nn.Linear(d, hidden, bias=False), nn.LayerNorm(hidden), nn.SiLU()]
        d = hidden
    mods.append(nn.Linear(d, out_dim))
    return nn.Sequential(*mods)


class ImageEncoder(nn.Module):
    """64x64x3 -> Merkmalsvektor. Vier Faltungen mit Schritt 2 (64->32->16->8->4)."""

    def __init__(self, in_channels: int = 3, depth: int = 32, resolution: int = 64):
        super().__init__()
        if resolution % 16 != 0:
            raise ValueError(f"obs_resolution muss durch 16 teilbar sein, war {resolution}")
        chans = [in_channels, depth, depth * 2, depth * 4, depth * 8]
        layers: list[nn.Module] = []
        for i in range(4):
            layers += [
                nn.Conv2d(chans[i], chans[i + 1], 4, stride=2, padding=1, bias=False),
                nn.GroupNorm(1, chans[i + 1]),
                nn.SiLU(),
            ]
        self.net = nn.Sequential(*layers)
        self.spatial = resolution // 16
        self.out_dim = chans[4] * self.spatial * self.spatial

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, C, H, W) in [0, 255] uint8 oder float."""
        # /255 - 0.5 statt symlog: Pixel sind schon beschraenkt und
        # gleichverteilt, da bringt symlog nichts und kostet nur Aufloesung.
        x = x.float() / 255.0 - 0.5
        return self.net(x).flatten(1)


class ImageDecoder(nn.Module):
    def __init__(self, in_dim: int, out_channels: int = 3, depth: int = 32, resolution: int = 64):
        super().__init__()
        self.spatial = resolution // 16
        self.chans = depth * 8
        self.fc = nn.Linear(in_dim, self.chans * self.spatial * self.spatial)
        chans = [depth * 8, depth * 4, depth * 2, depth]
        layers: list[nn.Module] = []
        for i in range(3):
            layers += [
                nn.ConvTranspose2d(chans[i], chans[i + 1], 4, stride=2, padding=1, bias=False),
                nn.GroupNorm(1, chans[i + 1]),
                nn.SiLU(),
            ]
        layers.append(nn.ConvTranspose2d(depth, out_channels, 4, stride=2, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """-> (N, C, H, W), Vorhersage von (Pixel/255 - 0.5)."""
        x = self.fc(feat).view(-1, self.chans, self.spatial, self.spatial)
        return self.net(x)


class VectorEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256, out_dim: int = 256):
        super().__init__()
        self.net = _mlp(in_dim, hidden, out_dim)
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(symlog(x))


class VectorDecoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 256):
        super().__init__()
        self.net = _mlp(in_dim, hidden, out_dim)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """Vorhersage IM symlog-Raum - der Verlust vergleicht gegen symlog(Ziel)."""
        return self.net(feat)


class RewardHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256, bins: int = 255):
        super().__init__()
        self.net = _mlp(in_dim, hidden, bins)
        # Nullinitialisierung der letzten Schicht: am Anfang sagt der Kopf
        # gleichverteilt "Belohnung 0" vorher statt zufaelligen Unsinn, dessen
        # Gradienten den RSSM in den ersten Schritten in eine falsche Richtung
        # ziehen wuerden.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat)


class ContinueHead(nn.Module):
    """Sagt vorher, ob die Episode weitergeht (1) oder endet (0)."""

    def __init__(self, in_dim: int, hidden: int = 256):
        super().__init__()
        self.net = _mlp(in_dim, hidden, 1)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat).squeeze(-1)
