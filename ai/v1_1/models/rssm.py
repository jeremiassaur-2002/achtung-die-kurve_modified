"""RSSM - das rekurrente Zustandsraummodell, das Herz des Weltmodells.

Der Zustand zerfaellt in zwei Teile:

  h_t  deterministisch, ein GRU-Zustand. Traegt alles, was sicher aus der
       Vergangenheit folgt (wo bin ich, wie schnell, welche Spur habe ich
       gezogen).
  z_t  stochastisch, 32 Gruppen à 32 Klassen. Traegt, was NICHT sicher ist -
       vor allem, was die Gegner als naechstes tun.

Zwei Verteilungen ueber z_t:

  Prior      p(z_t | h_t)          "was erwarte ich, ohne hinzusehen"
  Posterior  q(z_t | h_t, e_t)     "was war es wirklich" (e_t = Beobachtung)

Trainiert wird darauf, dass der Prior dem Posterior nahekommt. Genau dieser
Prior ist es, der spaeter in der Imagination laeuft: der Actor traeumt sich
Trajektorien, ohne dass die Engine je gefragt wird.

**Warum das KL-Ziel geteilt wird (KL-Balancing).** Ein einzelner KL-Term hat
zwei Wege, klein zu werden: der Prior lernt den Posterior - erwuenscht - oder der
Posterior verarmt zum Prior hin, bis z_t nichts mehr ueber die Beobachtung
aussagt. Letzteres ist der Posterior-Kollaps, und er ist der bequemere Weg.
Deshalb zwei Terme mit gestopptem Gradienten auf jeweils einer Seite und
unterschiedlichem Gewicht: der Dynamik-Term (beta 0,5) zieht den Prior zum
Posterior, der Repraesentations-Term (beta 0,1) nur schwach umgekehrt.

**free nats.** Unter 1 nat pro Schritt wird der KL gar nicht mehr bestraft. Ohne
das gibt das Modell die letzten Bits Information auf, um den KL auf null zu
druecken, und verliert dabei genau die feinen Unterschiede, auf die es ankommt.

**unimix.** 1% Gleichverteilung in jede kategoriale Verteilung gemischt. Damit
kann keine Klassenwahrscheinlichkeit exakt 0 werden - sonst waere der KL
unendlich, sobald der Posterior eine Klasse waehlt, die der Prior ausgeschlossen
hat, und der Verlust explodiert.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from ai.v1_1.models.nets import _mlp


@dataclass
class RSSMState:
    deter: torch.Tensor  # (..., deter_dim)      h
    logits: torch.Tensor  # (..., groups, classes) Verteilungsparameter von z
    stoch: torch.Tensor  # (..., groups, classes) gezogenes z (one-hot, ST)

    @property
    def feature(self) -> torch.Tensor:
        """[h, flatten(z)] - was Decoder, Koepfe und spaeter der Actor sehen."""
        return torch.cat([self.deter, self.stoch.flatten(-2)], dim=-1)

    def detach(self) -> "RSSMState":
        return RSSMState(self.deter.detach(), self.logits.detach(), self.stoch.detach())

    def __getitem__(self, idx) -> "RSSMState":
        return RSSMState(self.deter[idx], self.logits[idx], self.stoch[idx])


def _stack_states(states: list[RSSMState], dim: int = 1) -> RSSMState:
    return RSSMState(
        torch.stack([s.deter for s in states], dim=dim),
        torch.stack([s.logits for s in states], dim=dim),
        torch.stack([s.stoch for s in states], dim=dim),
    )


class RSSM(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        action_dim: int,
        deter_dim: int = 512,
        stoch_groups: int = 32,
        stoch_classes: int = 32,
        hidden_dim: int = 512,
        unimix: float = 0.01,
    ):
        super().__init__()
        self.deter_dim = deter_dim
        self.groups = stoch_groups
        self.classes = stoch_classes
        self.stoch_dim = stoch_groups * stoch_classes
        self.unimix = unimix

        # Eingang der Rekurrenz: [z_{t-1}, a_{t-1}] -> versteckte Groesse
        self.pre_gru = nn.Sequential(
            nn.Linear(self.stoch_dim + action_dim, hidden_dim, bias=False), nn.LayerNorm(hidden_dim), nn.SiLU()
        )
        self.gru = nn.GRUCell(hidden_dim, deter_dim)
        self.prior_net = _mlp(deter_dim, hidden_dim, self.stoch_dim, layers=1)
        self.post_net = _mlp(deter_dim + embed_dim, hidden_dim, self.stoch_dim, layers=1)

    # ------------------------------------------------------------ Verteilungen

    def _logits(self, raw: torch.Tensor) -> torch.Tensor:
        logits = raw.view(*raw.shape[:-1], self.groups, self.classes)
        if self.unimix > 0.0:
            probs = torch.softmax(logits, dim=-1)
            probs = (1.0 - self.unimix) * probs + self.unimix / self.classes
            logits = torch.log(probs)
        return logits

    def _sample(self, logits: torch.Tensor) -> torch.Tensor:
        """One-hot-Stichprobe mit Straight-Through-Gradient: vorwaerts diskret,
        rueckwaerts fliesst der Gradient durch die Wahrscheinlichkeiten."""
        probs = torch.softmax(logits, dim=-1)
        if self.training:
            idx = torch.multinomial(probs.reshape(-1, self.classes), 1).view(*probs.shape[:-1])
        else:
            idx = probs.argmax(-1)
        onehot = F.one_hot(idx, self.classes).float()
        return onehot + probs - probs.detach()

    def initial(self, batch_size: int, device) -> RSSMState:
        return RSSMState(
            torch.zeros(batch_size, self.deter_dim, device=device),
            torch.zeros(batch_size, self.groups, self.classes, device=device),
            torch.zeros(batch_size, self.groups, self.classes, device=device),
        )

    # ------------------------------------------------------------- ein Schritt

    def img_step(self, prev: RSSMState, prev_action: torch.Tensor) -> RSSMState:
        """Ein Schritt OHNE Beobachtung - der Traumschritt. Nur diese Funktion
        laeuft spaeter in der Imagination."""
        x = self.pre_gru(torch.cat([prev.stoch.flatten(-2), prev_action], dim=-1))
        deter = self.gru(x, prev.deter)
        logits = self._logits(self.prior_net(deter))
        return RSSMState(deter, logits, self._sample(logits))

    def obs_step(self, prev: RSSMState, prev_action: torch.Tensor, embed: torch.Tensor) -> tuple[RSSMState, RSSMState]:
        """Ein Schritt MIT Beobachtung. Liefert (Posterior, Prior) - der Prior
        wird fuer den KL-Term gebraucht, der Posterior wird weitergereicht."""
        prior = self.img_step(prev, prev_action)
        logits = self._logits(self.post_net(torch.cat([prior.deter, embed], dim=-1)))
        post = RSSMState(prior.deter, logits, self._sample(logits))
        return post, prior

    # ---------------------------------------------------------------- Sequenz

    def observe(
        self, embed: torch.Tensor, action: torch.Tensor, is_first: torch.Tensor, state: RSSMState | None = None
    ) -> tuple[RSSMState, RSSMState]:
        """Ueber (B, L, ...) ausrollen. Liefert (Posteriors, Priors), je (B, L, ...).

        `is_first` setzt Zustand UND vorherige Aktion auf null zurueck: am Beginn
        einer Episode gibt es keine vorherige Aktion, und die des letzten Ticks
        der vorigen Episode wuerde eine Kausalitaet vortaeuschen, die es nicht gibt.
        """
        b, seq_len = embed.shape[0], embed.shape[1]
        state = state or self.initial(b, embed.device)
        posts, priors = [], []
        for t in range(seq_len):
            reset = (~is_first[:, t]).float().unsqueeze(-1)
            state = RSSMState(state.deter * reset, state.logits * reset.unsqueeze(-1), state.stoch * reset.unsqueeze(-1))
            prev_action = action[:, t] * reset
            post, prior = self.obs_step(state, prev_action, embed[:, t])
            posts.append(post)
            priors.append(prior)
            state = post
        return _stack_states(posts), _stack_states(priors)

    def imagine(self, state: RSSMState, actions: torch.Tensor) -> RSSMState:
        """(B, H, A) Aktionen traeumen -> (B, H, ...) Zustaende. Ohne jede
        Beobachtung; hier ersetzt das Modell die Engine."""
        states = []
        for t in range(actions.shape[1]):
            state = self.img_step(state, actions[:, t])
            states.append(state)
        return _stack_states(states)

    # -------------------------------------------------------------------- KL

    @staticmethod
    def _kl(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        """KL(lhs || rhs) ueber kategoriale Gruppen, summiert ueber die Gruppen.
        (..., G, C) -> (...)"""
        p = torch.softmax(lhs, -1)
        return (p * (torch.log_softmax(lhs, -1) - torch.log_softmax(rhs, -1))).sum(-1).sum(-1)

    def kl_loss(
        self, post: RSSMState, prior: RSSMState, free_nats: float = 1.0, beta_dyn: float = 0.5, beta_rep: float = 0.1
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(gesamt, dyn, rep) - je (B, L), ohne Reduktion, damit der Aufrufer die
        Padding-Maske anwenden kann."""
        dyn = self._kl(post.logits.detach(), prior.logits)  # Prior lernt
        rep = self._kl(post.logits, prior.logits.detach())  # Posterior gibt nach
        dyn_c = torch.clamp(dyn, min=free_nats)
        rep_c = torch.clamp(rep, min=free_nats)
        return beta_dyn * dyn_c + beta_rep * rep_c, dyn, rep
