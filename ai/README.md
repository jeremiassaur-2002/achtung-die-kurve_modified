# Achtung, die Kurve! - RL Training System

A from-scratch, headless Python reimplementation of the game's physics (`ai/env`), paired with a CNN+PPO training pipeline (Stable-Baselines3 + PyTorch) that goes through curriculum learning -> self-play -> multiplayer -> league training -> items (`ai/training`), automatic evaluation (`ai/evaluation`) and reporting (`ai/reporting`), and export back into the actual browser game (`ai/export`, `ai_bot.js`).

See `C:\Users\jerem\.claude\plans\fizzy-whistling-backus.md` (or ask for a summary) for the full design rationale and the exact script.js formulas this was ported from.

## Versionen

Seit dem Umbau liegt jede Trainingsversion in einem eigenen Paket, der gemeinsame
Spielkern genau einmal darunter:

| | Algorithmus | Stand |
| --- | --- | --- |
| `ai/v1_0/` | PPO / RecurrentPPO / MaskablePPO / QR-DQN (Stable-Baselines3) | vollstaendig, unveraendert lauffaehig |
| `ai/v1_1/` | Dreamer (RSSM-Weltmodell) + Beam-Pfadplaner | Planer und Datensammlung fertig, Weltmodell folgt |

**Warum Engine und Env NICHT mitkopiert wurden.** Naheliegend waere gewesen, `ai/env`
in beide Versionen zu duplizieren. Das waere ein Fehler: Physik, Renderer, Sensoren
und Gegner-Controller sind nicht PPO-spezifisch, und jeder Bugfix an der Engine
muesste doppelt gemacht werden - die Fehlerklasse, an der schon die
Browser/Python-Diskrepanz beim Selbstkollisions-Bug haing. Ausserdem waere ein
Vergleich "v1_0 gegen v1_1" wertlos, wenn beide gegen leicht verschiedene Physik
antreten. Deshalb: gemeinsamer Kern, getrennte Algorithmen.

```
ai/
  run.py                     DER Einstiegspunkt (train | collect | info)
  core/                      alles Algorithmus-Unabhaengige
    config/game_constants.py aus script.js extrahierte Konstanten
    env/                     CurveEngine, renderer, sensors, observation, opponents,
                             curve_env (Gymnasium), rules_bot, vec_factory
    evaluation/              Arena, Bewertungsbatterie
    export/                  ONNX-Export fuer ai_bot.js
    utils/                   paths (Ausgabelayout), timing (Phasenzeiten), sysinfo (GPU)
  v1_0/                      PPO: config/phase1..5.yaml, models/, training/, reporting/, entry.py
  v1_1/                      Dreamer: config/dreamer.yaml, planner/ (Beam-Suche),
                             data/collect.py (Expertendaten), training/, entry.py
  output/                    ALLE Laufartefakte, <version>/<lauf>/ (gitignored)
  tests/                     pytest-Suite
scripts/                     bootstrap.sh, train.sh (tmux), sync_output.sh
```

## Ein Befehl fuer alles

```bash
python -m ai.run info                                                    # Hardware + vorhandene Laeufe
python -m ai.run train   --version v1_0 --config ai/v1_0/config/phase1.yaml
python -m ai.run train   --version v1_1 --config ai/v1_1/config/dreamer.yaml
python -m ai.run collect --version v1_1 --config ai/v1_1/config/dreamer.yaml --episodes 200
```

`--run-name auto` (Default) greift den juengsten Lauf der Version wieder auf statt
einen zweiten danebenzulegen; zusammen mit `--resume auto` ist derselbe Befehl nach
einem Absturz einfach nochmal aufrufbar. Genau darauf baut die Neustartschleife in
`scripts/train.sh`.

Jeder Lauf schreibt nach `ai/output/<version>/<lauf>/` mit identischem Layout:
`checkpoints/ best/ tensorboard/ videos/ reports/ metrics/ timing/ config_used.yaml`.
Ein einziges `tensorboard --logdir ai/output` zeigt beide Versionen nebeneinander.
`AI_OUTPUT_ROOT` verschiebt das Ganze auf ein persistentes Volume.

## Zeitlogging

`ai/core/utils/timing.py` misst, WOFUER die Zeit draufgeht - TensorBoard zeigt nur,
WAS dabei herauskommt. Nach jedem Abschnitt geschrieben (nicht erst am Ende, ein
abgestuerzter Lauf soll seine Zeiten behalten), atomar per temp+rename:

- `timing/timing.md` - die Tabelle, die man anschaut
- `timing/timing.json` - maschinenlesbar
- `timing/sysinfo.json` - GPU, VRAM, CPU-Zahl, torch-Version

Ohne `sysinfo.json` waere die Tabelle wertlos: "world_model: 4h" heisst auf einer
4090 etwas anderes als auf einer L4.

## v1_1: Planer und statisches Training

Der Beam-Planer (`ai/v1_1/planner/beam.py`) sucht die Aktionsfolge, die am laengsten
ueberlebt, und kann ohne jedes Training spielen. Gemessen im Solo-Lauf:

| Steuerung | ø ueberlebte Ticks |
| --- | --- |
| Zufall | 247 |
| `rules_bot` (hard) | 1436 |
| Beam (Horizont 60) | 2434 |
| Beam (Horizont 90) | 2581 |

Beam statt MCTS, weil der teure Teil nicht die Suchstrategie ist, sondern das
Vorwaertssimulieren gegen das Gitter - und das erledigt `sensors.simulate_deltas`
fuer hunderte Kandidaten in EINEM numpy-Gather. MCTS mit sequentiellen
Einzel-Rollouts verschenkt genau diese Vektorisierung.

Damit laeuft das statische Training: der Planer erzeugt Trajektorien, das
Weltmodell lernt daraus offline, ohne dass das Netz je selbst spielen muss.
Praktischer Nebeneffekt der Trennung - Sammeln ist CPU-gebunden und
parallelisierbar, Modelltraining GPU-gebunden. Man kann also auf billigen Kernen
sammeln und die GPU nur fuer die Stunden mieten, in denen sie wirklich rechnet.

Gespeichert wird pro Tick ein EINZELNER Frame, nicht der 4er-Stack von v1_0: der
RSSM fuehrt einen rekurrenten Zustand und baut sich die Historie selbst. Gemessen
komprimieren die Shards auf ~0,2 KB/Tick, 200k Ticks sind also rund 40 MB.

## Sensoren

`n_rays: 36` in `ai/v1_1/config/dreamer.yaml` - 360/36 = alle 10 Grad ein Abstand,
kopfrelativ (Strahl 0 zeigt geradeaus). Der Beobachtungsvektor waechst damit von 38
auf 61 Werte. v1_0 bleibt bei 16 Strahlen, damit vorhandene Checkpoints weiter
ladbar sind - eine Aenderung der Beobachtungsgroesse macht jeden alten Checkpoint
unbrauchbar.

## Training auf gemieteter GPU

```bash
bash scripts/bootstrap.sh                                   # Pakete, Repo, Selbsttest
bash scripts/train.sh v1_0 ai/v1_0/config/phase1.yaml       # tmux-Sitzung
tmux attach -t achtung                                      # loesen: Strg-B, dann D
```

Die tmux-Sitzung haelt drei Fenster: Training (in einer Neustartschleife),
TensorBoard auf 6006, und `nvidia-smi` + Timing-Tabelle. Zugeklappter Laptop,
geschlossener Browser und getrennte SSH-Verbindung beenden nichts davon.

`scripts/sync_output.sh <version>` sichert Ergebnisse in den separaten Branch
`results` - und zwar nur `best/`, `reports/`, `timing/`, `metrics/*.csv`. Die
rotierenden Checkpoints bleiben lokal: sie sind gross, kurzlebig und nur fuer die
Wiederaufnahme auf DIESER Maschine gedacht. Ein Push von 200-MB-Dateien nach `main`
macht jeden spaeteren Klon dauerhaft langsam, weil Git-History nichts vergisst.

## Local setup

Windows path lengths matter here: torch's own package contents can exceed `MAX_PATH` inside a deeply nested
project folder, so the working venv for this project lives outside the repo:

```
py -3.13 -m venv C:\venvs\achtung_kurve
C:\venvs\achtung_kurve\Scripts\pip install -r ai\requirements.txt
```

Run the test suite (constants fidelity, engine collision/scoring/items, Gym API compliance, a tiny PPO smoke-train):

```
C:\venvs\achtung_kurve\Scripts\python -m pytest ai\tests -v
```

## Training

One `train.py` call = one phase, driven entirely by its YAML config. Chain phases with `--init-from`:

```
python -m ai.v1_0.training.train --config ai/v1_0/config/phase1.yaml
python -m ai.v1_0.training.train --config ai/v1_0/config/phase2.yaml --init-from ai/runs/phase1_.../final_model.zip
python -m ai.v1_0.training.train --config ai/v1_0/config/phase3.yaml --init-from ai/runs/phase2_.../final_model.zip
python -m ai.v1_0.training.train --config ai/v1_0/config/phase4.yaml --init-from ai/runs/phase3_.../final_model.zip
python -m ai.v1_0.training.train --config ai/v1_0/config/phase5.yaml --init-from ai/runs/phase4_.../final_model.zip
```

- **Phase 1** - solo survival, no opponents/items.
- **Phase 2** - self-play, 1 opponent drawn from a rotating pool of recent snapshots.
- **Phase 3** - multiplayer curriculum, 2 to 5 opponents (capped at 5, not the 6 the brief mentions, because the
  actual game only has 6 named player slots total - 1 hero + at most 5 others - see script.js's `players` object).
  The curriculum soft-blends into each harder stage as rolling win rate clears a threshold.
- **Phase 4** - league training: checkpoints saved and Elo-rated throughout, opponents mixed from current +
  historical checkpoints + every rule-based difficulty + random.
- **Phase 5** - same as Phase 4, with the full item set active.

Real multi-million-step runs belong on Colab (`ai/colab/Train_Achtung_Kurve_AI.ipynb`) - locally this is only meant
for short smoke-training (`--timesteps 5000` to override a config's `total_timesteps`).

Each run writes everything under `ai/runs/<run-name>/`: `final_model.zip`, `tensorboard/`, `metrics/metrics.csv`
+ `.jsonl`, `reports/milestone_*/report.md` (auto-generated at 10k/50k/100k/500k/1M/5M/10M steps), and (if enabled)
`league/` + `self_play_pool/`.

## Evaluation

```
python -m ai.core.evaluation.evaluate --checkpoint ai/runs/phase5_.../final_model.zip --league ai/runs/phase5_.../league --matches 30
```

Reports win rate, average placement, average survival ticks, kills, item usage, and death-cause breakdown against
random play, every rule-based difficulty, and league history.

## Export + game integration

```
python -m ai.core.export.export_onnx --checkpoint <final_model.zip> --out ai/exported/model.onnx
python -m ai.core.export.export_weights --checkpoint <final_model.zip> --out-dir ai/exported
```

`export_onnx.py` also writes a `model_data.js` sidecar next to the `.onnx` file (the model bytes, base64-embedded
as `const AI_MODEL_BASE64 = "..."`). Copy both `model.onnx` and `model_data.js` next to `index.html` - `model_data.js`
is what lets the game work by just double-clicking `index.html` (`file://`): onnxruntime-web normally loads the
model via `fetch()`, which browsers block for local files under `file://`, but a plain `<script src="model_data.js">`
has no such restriction, so `ai_bot.js` prefers the embedded bytes when they're present (falling back to fetching
`model.onnx` by URL only if you skip the sidecar and serve the page over a local server instead).

On the start screen, tick a player's **KI** checkbox (next to Left/Right) to make that seat an AI player - or from
the browser console, `addAI('fred')` (any other player name). See `ai_bot.js`'s header comment for how its
observation construction mirrors `ai/core/env/observation.py`.

## A note on `engine_resolution`

`GameConstants(S)` treats `S` as the arena's side length in pixels, exactly like script.js's `h`. Keep `S >= ~200`:
below that, `hitbox_size` drops under ~1px and the engine's plain integer-pixel rasterization (no canvas-style
anti-aliasing) makes straight-line movement spuriously clip its own just-drawn trail. The default (256) clears this
with margin - this was found empirically (see `ai/tests/test_engine.py`), not assumed.
