# Achtung, die Kurve! - RL Training System

A from-scratch, headless Python reimplementation of the game's physics (`ai/env`), paired with a CNN+PPO training pipeline (Stable-Baselines3 + PyTorch) that goes through curriculum learning -> self-play -> multiplayer -> league training -> items (`ai/training`), automatic evaluation (`ai/evaluation`) and reporting (`ai/reporting`), and export back into the actual browser game (`ai/export`, `ai_bot.js`).

See `C:\Users\jerem\.claude\plans\fizzy-whistling-backus.md` (or ask for a summary) for the full design rationale and the exact script.js formulas this was ported from.

## Layout

```
ai/
  config/game_constants.py   every ratio/timer extracted from script.js, plus GameConstants(engine_resolution)
  config/phase1..5.yaml      per-phase env/PPO/curriculum config
  env/                       CurveEngine (physics), renderer (RGB obs), rules_bot (heuristic opponent),
                             observation (image+vector obs), opponents (Controllers), curve_env (Gymnasium env),
                             vec_factory (parallel envs)
  models/                    swappable CNN backbones + SB3 policy wiring
  training/                  train.py entrypoint, curriculum, self-play pool, league, Elo, callbacks
  evaluation/                arena (headless matches), evaluate (win rate/survival/placement/kills/items battery)
  reporting/                 matplotlib charts + auto-generated Markdown reports at step milestones
  export/                    ONNX export (for ai_bot.js) + plain weights export
  colab/                     Train_Achtung_Kurve_AI.ipynb
  tests/                     pytest suite (constants fidelity, engine behavior, Gym API compliance)
ai_bot.js                    in-browser ONNX inference, drop-in alongside the game (see its header comment)
```

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
python -m ai.training.train --config ai/config/phase1.yaml
python -m ai.training.train --config ai/config/phase2.yaml --init-from ai/runs/phase1_.../final_model.zip
python -m ai.training.train --config ai/config/phase3.yaml --init-from ai/runs/phase2_.../final_model.zip
python -m ai.training.train --config ai/config/phase4.yaml --init-from ai/runs/phase3_.../final_model.zip
python -m ai.training.train --config ai/config/phase5.yaml --init-from ai/runs/phase4_.../final_model.zip
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
python -m ai.evaluation.evaluate --checkpoint ai/runs/phase5_.../final_model.zip --league ai/runs/phase5_.../league --matches 30
```

Reports win rate, average placement, average survival ticks, kills, item usage, and death-cause breakdown against
random play, every rule-based difficulty, and league history.

## Export + game integration

```
python -m ai.export.export_onnx --checkpoint <final_model.zip> --out ai/exported/model.onnx
python -m ai.export.export_weights --checkpoint <final_model.zip> --out-dir ai/exported
```

`export_onnx.py` also writes a `model_data.js` sidecar next to the `.onnx` file (the model bytes, base64-embedded
as `const AI_MODEL_BASE64 = "..."`). Copy both `model.onnx` and `model_data.js` next to `index.html` - `model_data.js`
is what lets the game work by just double-clicking `index.html` (`file://`): onnxruntime-web normally loads the
model via `fetch()`, which browsers block for local files under `file://`, but a plain `<script src="model_data.js">`
has no such restriction, so `ai_bot.js` prefers the embedded bytes when they're present (falling back to fetching
`model.onnx` by URL only if you skip the sidecar and serve the page over a local server instead).

On the start screen, tick a player's **KI** checkbox (next to Left/Right) to make that seat an AI player - or from
the browser console, `addAI('fred')` (any other player name). See `ai_bot.js`'s header comment for how its
observation construction mirrors `ai/env/observation.py`.

## A note on `engine_resolution`

`GameConstants(S)` treats `S` as the arena's side length in pixels, exactly like script.js's `h`. Keep `S >= ~200`:
below that, `hitbox_size` drops under ~1px and the engine's plain integer-pixel rasterization (no canvas-style
anti-aliasing) makes straight-line movement spuriously clip its own just-drawn trail. The default (256) clears this
with margin - this was found empirically (see `ai/tests/test_engine.py`), not assumed.
