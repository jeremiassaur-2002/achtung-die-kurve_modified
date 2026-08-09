"""Das Weltmodell: Encoder + RSSM + Decoder + Belohnungs-/Fortsetzungskopf.

Ein Trainingsschritt:

  1. Beobachtung (Bild + Sensorvektor) -> Einbettung e_t
  2. RSSM rollt ueber die Sequenz aus und liefert Posteriors und Priors
  3. Aus dem Posterior-Zustand werden rekonstruiert bzw. vorhergesagt:
     Bild, Sensorvektor, Belohnung, Fortsetzung
  4. Verlust = Rekonstruktion + Vorhersage + gewichteter KL

Die Rekonstruktion ist kein Selbstzweck. Sie ist das einzige Signal, das den
Zustand zwingt, ueberhaupt Information ueber die Welt zu tragen - ohne sie wuerde
ein Modell, das nur Belohnung vorhersagt, in einen konstanten Zustand kollabieren
(die Belohnung ist fast immer +0,01) und in der Imagination nichts Sinnvolles
liefern.

**Warum der Sensorvektor ein eigener Kopf ist und nicht im Bild aufgeht.** Die
36 Strahlen und die drei Arc-Ueberlebenszeiten sind genau die Nahfeldgeometrie,
die aus einem 64x64-Bild nicht mehr ablesbar ist - der Kopfpunkt ist dort
kleiner als ein Pixel. Ein Modell, das sie separat vorhersagen muss, kann sie im
Traum auch separat weiterrechnen. Der Actor bekommt sie damit in der Imagination
genauso wie in der echten Umgebung.

**Gewichtung der Verlustanteile.** Der Bildterm summiert ueber 64x64x3 = 12288
Werte, der Belohnungsterm ueber einen. Ohne Gegenmassnahme waere der Belohnungs-
kopf numerisch bedeutungslos. DreamerV3 loest das nicht ueber Gewichte, sondern
darueber, dass alle Terme SUMMIERT (nicht gemittelt) werden und die Koepfe
ausreichend Kapazitaet haben; die Bildrekonstruktion dominiert dann zwar den
Zahlenwert, aber die Gradienten der kleinen Koepfe erreichen den RSSM trotzdem
ungedaempft, weil sie auf einem eigenen Pfad liegen.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn

from ai.v1_1.models.nets import (
    ContinueHead,
    ImageDecoder,
    ImageEncoder,
    RewardHead,
    TwoHotEncoding,
    VectorDecoder,
    VectorEncoder,
    symlog,
)
from ai.v1_1.models.rssm import RSSM, RSSMState


@dataclass
class WorldModelConfig:
    image_shape: tuple[int, int, int] = (64, 64, 3)  # H, W, C
    vector_dim: int = 61
    action_dim: int = 3
    deter_dim: int = 512
    stoch_groups: int = 32
    stoch_classes: int = 32
    hidden_dim: int = 512
    cnn_depth: int = 32
    vector_embed: int = 256
    reward_bins: int = 255
    kl_free_nats: float = 1.0
    kl_beta_dyn: float = 0.5
    kl_beta_rep: float = 0.1
    unimix: float = 0.01
    # Klassengewicht fuer den Fortsetzungskopf. Terminals machen unter 0,3% der
    # Schritte aus; ohne Gewichtung minimiert der Kopf seinen Verlust am besten,
    # indem er ausnahmslos "geht weiter" sagt.
    continue_pos_weight: float = 50.0
    ttd_cap: float = 120.0
    survival_weight: float = 1.0


@dataclass
class WorldModelLoss:
    total: torch.Tensor
    parts: dict = field(default_factory=dict)


class WorldModel(nn.Module):
    def __init__(self, cfg: WorldModelConfig):
        super().__init__()
        self.cfg = cfg
        h, w, c = cfg.image_shape
        if h != w:
            raise ValueError(f"quadratische Bilder erwartet, war {h}x{w}")

        self.image_encoder = ImageEncoder(in_channels=c, depth=cfg.cnn_depth, resolution=h)
        self.vector_encoder = VectorEncoder(cfg.vector_dim, hidden=cfg.vector_embed, out_dim=cfg.vector_embed)
        embed_dim = self.image_encoder.out_dim + cfg.vector_embed

        self.rssm = RSSM(
            embed_dim=embed_dim,
            action_dim=cfg.action_dim,
            deter_dim=cfg.deter_dim,
            stoch_groups=cfg.stoch_groups,
            stoch_classes=cfg.stoch_classes,
            hidden_dim=cfg.hidden_dim,
            unimix=cfg.unimix,
        )
        feat_dim = cfg.deter_dim + cfg.stoch_groups * cfg.stoch_classes
        self.feat_dim = feat_dim

        self.image_decoder = ImageDecoder(feat_dim, out_channels=c, depth=cfg.cnn_depth, resolution=h)
        self.vector_decoder = VectorDecoder(feat_dim, cfg.vector_dim, hidden=cfg.vector_embed)
        self.reward_head = RewardHead(feat_dim, hidden=cfg.hidden_dim, bins=cfg.reward_bins)
        self.continue_head = ContinueHead(feat_dim, hidden=cfg.hidden_dim)
        # Restlebensdauer-Kopf: dieselbe Information wie der Fortsetzungskopf,
        # aber als DICHTES Ziel an jedem Schritt statt als 0,3%-Ereignis. Das
        # ist der Haupthebel dafuer, dass der latente Zustand ueberhaupt
        # Todesnaehe kodiert - und damit dafuer, dass der Actor in der
        # Imagination etwas zu vermeiden hat.
        self.survival_head = RewardHead(feat_dim, hidden=cfg.hidden_dim, bins=cfg.reward_bins)
        self.twohot = TwoHotEncoding(bins=cfg.reward_bins)

    # ------------------------------------------------------------------

    def encode(self, image: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
        """(B, L, H, W, C) uint8 + (B, L, V) -> (B, L, E)."""
        b, seq_len = image.shape[0], image.shape[1]
        img = image.reshape(b * seq_len, *image.shape[2:]).permute(0, 3, 1, 2)
        e_img = self.image_encoder(img).view(b, seq_len, -1)
        e_vec = self.vector_encoder(vector)
        return torch.cat([e_img, e_vec], dim=-1)

    def observe(self, image, vector, action, is_first, state: RSSMState | None = None):
        embed = self.encode(image, vector)
        return self.rssm.observe(embed, action, is_first, state)

    # ------------------------------------------------------------------

    def loss(
        self,
        image: torch.Tensor,
        vector: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        is_first: torch.Tensor,
        is_terminal: torch.Tensor,
        mask: torch.Tensor,
        ticks_to_death: torch.Tensor | None = None,
    ) -> tuple[WorldModelLoss, RSSMState]:
        cfg = self.cfg
        self.twohot.to(image.device)
        post, prior = self.observe(image, vector, action, is_first)
        feat = post.feature
        b, seq_len = feat.shape[0], feat.shape[1]
        flat = feat.reshape(b * seq_len, -1)

        # --- Bild ---
        target_img = image.reshape(b * seq_len, *image.shape[2:]).permute(0, 3, 1, 2).float() / 255.0 - 0.5
        pred_img = self.image_decoder(flat)
        # Summe ueber Pixel, Mittel erst ganz am Ende ueber (B, L) - siehe
        # Modul-Docstring zur Gewichtung.
        img_loss = 0.5 * ((pred_img - target_img) ** 2).sum(dim=(1, 2, 3)).view(b, seq_len)

        # --- Sensorvektor (im symlog-Raum) ---
        pred_vec = self.vector_decoder(flat).view(b, seq_len, -1)
        vec_loss = 0.5 * ((pred_vec - symlog(vector)) ** 2).sum(-1)

        # --- Belohnung (twohot-Klassifikation) ---
        rew_logits = self.reward_head(flat).view(b, seq_len, -1)
        rew_loss = self.twohot.loss(rew_logits, reward)

        # --- Fortsetzung ---
        cont_logits = self.continue_head(flat).view(b, seq_len)
        cont_target = (~is_terminal).float()
        # pos_weight gewichtet hier die SELTENE Klasse (Terminal = Ziel 0), also
        # per weight auf dem Negativ-Term - binary_cross_entropy_with_logits
        # gewichtet nur Positive, daher das Gewicht ueber `weight` von Hand.
        cont_w = torch.where(is_terminal, torch.full_like(cont_target, cfg.continue_pos_weight), torch.ones_like(cont_target))
        cont_loss = F.binary_cross_entropy_with_logits(cont_logits, cont_target, weight=cont_w, reduction="none")

        # --- Restlebensdauer (twohot, dichtes Todesnaehe-Signal) ---
        if ticks_to_death is None:
            surv_loss = torch.zeros_like(cont_loss)
            surv_mae = torch.zeros((), device=cont_loss.device)
        else:
            surv_logits = self.survival_head(flat).view(b, seq_len, -1)
            surv_loss = cfg.survival_weight * self.twohot.loss(surv_logits, ticks_to_death)
            surv_mae = (self.twohot.decode(surv_logits).detach() - ticks_to_death).abs()

        # --- KL ---
        kl, kl_dyn, kl_rep = self.rssm.kl_loss(
            post, prior, free_nats=cfg.kl_free_nats, beta_dyn=cfg.kl_beta_dyn, beta_rep=cfg.kl_beta_rep
        )

        per_step = img_loss + vec_loss + rew_loss + cont_loss + surv_loss + kl
        # Durch die Anzahl GUELTIGER Schritte teilen, nicht durch B*L: sonst
        # bekaeme ein Batch mit vielen kurzen (also toedlichen) Episoden
        # kuenstlich kleinen Verlust, und genau die sind die wichtigen.
        denom = mask.sum().clamp(min=1.0)
        total = (per_step * mask).sum() / denom

        def m(x: torch.Tensor) -> float:
            return float((x.detach() * mask).sum() / denom)

        parts = {
            "image": m(img_loss),
            "vector": m(vec_loss),
            "reward": m(rew_loss),
            "continue": m(cont_loss),
            "survival": m(surv_loss),
            "kl": m(kl),
            "kl_dyn": m(kl_dyn),
            "kl_rep": m(kl_rep),
            "reward_mae": float(
                ((self.twohot.decode(rew_logits).detach() - reward).abs() * mask).sum() / denom
            ),
            "continue_acc": float((((cont_logits.detach() > 0).float() == cont_target).float() * mask).sum() / denom),
            # Trefferquote NUR auf den Terminals - die Gesamtquote liegt wegen
            # der Seltenheit ohnehin ueber 99% und sagt nichts aus.
            "terminal_recall": float(
                (((cont_logits.detach() <= 0).float() * is_terminal.float()) * mask).sum()
                / (is_terminal.float() * mask).sum().clamp(min=1.0)
            ),
            "survival_mae": (m(surv_mae) if ticks_to_death is not None else 0.0),
        }
        return WorldModelLoss(total=total, parts=parts), post

    # ------------------------------------------------------------------

    @torch.no_grad()
    def dream(self, state: RSSMState, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        """Aus einem Zustand heraus eine Aktionsfolge traeumen - ohne Engine.

        Das ist die Diagnose, die das Weltmodell UNABHAENGIG von jeder Policy
        pruefbar macht: einige Ticks Kontext einspielen, dann die tatsaechlich
        gespielten Aktionen weitertraeumen und Traumbild gegen echtes Bild
        halten. Faellt das auseinander, ist eine spaetere Actor-Ausbildung in
        dieser Imagination sinnlos - und man weiss es, bevor man GPU-Stunden
        dafuer ausgibt.
        """
        self.twohot.to(actions.device)
        states = self.rssm.imagine(state, actions)
        feat = states.feature
        b, horizon = feat.shape[0], feat.shape[1]
        flat = feat.reshape(b * horizon, -1)
        img = (self.image_decoder(flat) + 0.5).clamp(0, 1)
        return {
            "image": (img * 255).byte().view(b, horizon, *img.shape[1:]).permute(0, 1, 3, 4, 2),
            "vector": self.vector_decoder(flat).view(b, horizon, -1),
            "reward": self.twohot.decode(self.reward_head(flat)).view(b, horizon),
            "continue": torch.sigmoid(self.continue_head(flat)).view(b, horizon),
            "ticks_to_death": self.twohot.decode(self.survival_head(flat)).view(b, horizon),
        }
