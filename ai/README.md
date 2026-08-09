# Achtung, die Kurve! - RL Training System

A from-scratch, headless Python reimplementation of the game's physics (`ai/core/env`), paired with a CNN+PPO training pipeline (Stable-Baselines3 + PyTorch) that goes through curriculum learning -> self-play -> multiplayer -> league training -> items (`ai/v1_0/training`), automatic evaluation (`ai/core/evaluation`) and reporting (`ai/v1_0/reporting`), and export back into the actual browser game (`ai/core/export`, `ai_bot.js`). A second version, `ai/v1_1`, trains a Dreamer-style world model + actor/critic on the same physics core.

See `C:\Users\jerem\.claude\plans\fizzy-whistling-backus.md` (or ask for a summary) for the full design rationale and the exact script.js formulas this was ported from.

## Versionen

Seit dem Umbau liegt jede Trainingsversion in einem eigenen Paket, der gemeinsame
Spielkern genau einmal darunter:

| | Algorithmus | Stand |
| --- | --- | --- |
| `ai/v1_0/` | PPO / RecurrentPPO / MaskablePPO / QR-DQN (Stable-Baselines3) | vollstaendig, unveraendert lauffaehig |
| `ai/v1_1/` | Dreamer (RSSM-Weltmodell) + Beam-Pfadplaner | vollstaendige Pipeline (Sammeln -> Weltmodell -> Actor/Critic) lauffaehig |

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
                             data/ (collect.py, replay.py), models/ (world_model.py,
                             actor_critic.py), training/ (train_world_model.py,
                             train_policy.py), agent.py, evaluate.py, entry.py
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

## v1_1: das Weltmodell

`ai/v1_1/models/` enthaelt einen DreamerV3-artigen RSSM. Der Zustand zerfaellt in
einen deterministischen GRU-Teil `h` (was sicher aus der Vergangenheit folgt) und
32x32 kategoriale Latents `z` (was offen ist - vor allem, was die Gegner tun).
Trainiert wird der Prior `p(z|h)` darauf, dem Posterior `q(z|h,e)` nahezukommen;
genau dieser Prior laeuft spaeter in der Imagination, wenn der Actor sich
Trajektorien traeumt, ohne dass die Engine gefragt wird.

Vier Entscheidungen, die den Unterschied machen:

- **KL-Balancing** (beta_dyn 0,5 / beta_rep 0,1). Ein einzelner KL-Term hat zwei
  Wege, klein zu werden: der Prior lernt den Posterior - erwuenscht - oder der
  Posterior verarmt, bis `z` nichts mehr ueber die Beobachtung aussagt. Letzteres
  ist der bequemere Weg. Zwei Terme mit gestopptem Gradienten auf je einer Seite
  verhindern den Kollaps.
- **free nats 1,0.** Unter 1 nat wird der KL nicht mehr bestraft - sonst gibt das
  Modell die letzten Bits auf, um den KL auf null zu druecken, und verliert genau
  die feinen Unterschiede, auf die es ankommt.
- **twohot-Belohnungskopf statt Regression.** Belohnungen sind hier extrem schief
  (fast immer +0,01, sehr selten -1). Eine Regression konvergiert gegen den
  Mittelwert und sagt den Tod nie voraus; eine Klassifikation mit Kreuzentropie
  behaelt der seltenen Klasse ihr Gewicht.
- **Eigener Kopf fuer den Sensorvektor.** Die 36 Strahlen sind aus einem
  64x64-Bild nicht ablesbar - der Kopfpunkt ist dort kleiner als ein Pixel. Ein
  Modell, das sie separat vorhersagen muss, kann sie im Traum separat
  weiterrechnen; der Actor bekommt sie in der Imagination genau wie in echt.

### Woran man erkennt, ob es funktioniert

Nicht am Gesamtverlust - der wird vom Bildterm dominiert (12288 Pixel gegen einen
Belohnungswert) und sinkt auch dann, wenn das Modell nur den schwarzen Hintergrund
gelernt hat. Aussagekraeftig sind die Einzelgroessen im Log und in TensorBoard:

| Groesse | was sie bedeutet |
| --- | --- |
| `reward_mae` | muss deutlich unter 0,1 fallen, sonst kann der Critic nichts damit anfangen |
| `kl_dyn` | faellt auf die free nats (1,0) und bleibt dort = gut. Faellt auf ~0 = Posterior kollabiert, Traum wertlos |
| `continue_acc` | Vorsicht: der Tod ist selten, "immer weiter" kommt schon ueber 99% |
| `dream_vec_mae` | **der eigentliche Test** - Kontext einspielen, dann N Ticks blind traeumen, getraeumte gegen echte Sensorwerte |

`dream_vec_mae` misst die Imagination unabhaengig von jeder Policy. Taugt sie
nichts, waere jede darin ausgebildete Policy verschwendete Rechenzeit - und man
weiss es, bevor GPU-Stunden dafuer draufgehen.

### Replay

`ai/v1_1/data/replay.py` sampelt (B, L)-Sequenzen aus den Shards. Zwei Details,
die leicht falsch laufen:

- **Laengengewichtete Auswahl.** Gleichverteilt ueber Episoden zu ziehen wuerde
  kurze - also toedlich endende - Episoden massiv ueberrepraesentieren und das
  Weltmodell pessimistisch machen.
- **Kurze Episoden werden maskiert aufgefuellt, nicht uebersprungen.** Wer stirbt,
  stirbt frueh; sie zu verwerfen wuerde ausgerechnet die Todesfaelle
  herausfiltern, also das, was gelernt werden soll.

Sequenzen liegen immer vollstaendig innerhalb einer Episode - sonst wuerde der
RSSM lernen, dass auf einen Aufprall ein frisch zurueckgesetztes Spielfeld folgt.

## v1_1: Actor/Critic in der Imagination

`ai/v1_1/models/actor_critic.py` + `training/train_policy.py`. Actor und Critic
sehen nie ein Bild und nie einen Sensorwert, nur `feature = [h, flatten(z)]`.
Genau darin liegt der Gewinn: der Encoder laeuft einmal pro echtem Tick, das
Policy-Training auf tausenden getraeumten Schritten, in denen gar kein Bild
existiert.

Ein Schritt: echte Sequenz -> Weltmodell rollt aus -> jeder Posterior-Zustand ist
ein Startpunkt -> `imagination_horizon` Schritte traeumen -> lambda-Renditen ->
Critic lernt sie, Actor lernt per REINFORCE mit dem Critic als Baseline.

**Das Weltmodell ist dabei eingefroren.** Zwei Optimierer gleichzeitig waeren ein
bewegliches Ziel - der Actor optimierte gegen ein Modell, das sich unter ihm
veraendert, und beide koennten gemeinsam in eine Fantasie abdriften, in der es
sich bequem lebt. Ein Test prueft, dass kein Gradient ins Weltmodell laeuft.

Weitere Entscheidungen:

- **REINFORCE statt Reparametrisierung** - durch eine kategoriale Stichprobe
  ueber drei Aktionen fliesst kein sauberer Pfadgradient.
- **Langsamer Ziel-Critic (EMA 0,98).** Der Critic lernt gegen Ziele, die er
  selbst erzeugt hat; ohne Entkopplung fuettert sich der Fehler zurueck.
- **Return-Normalisierung ueber Perzentile (5/95), gedeckelt bei 1.**
  `alive_bonus` 0,01 und `death_penalty` -1 liegen zwei Groessenordnungen
  auseinander. Perzentile statt Mittelwert/Streuung, weil einzelne Tode die
  Statistik sonst dominieren; der Deckel verhindert, dass winzige Renditen
  kuenstlich aufgeblasen werden.
- **BC-Warmstart auf die Planer-Aktionen.** Ohne ihn startet der Actor
  gleichverteilt, stirbt in den ersten getraeumten Schritten und lernt aus lauter
  kurzen, gleich schlechten Traeumen fast nichts. Reine Verhaltensklonung waere
  aber zu wenig: sie kopiert den Planer samt Fehlern und kann ihn nie
  uebertreffen. BC liefert den Startpunkt, die Imagination den Fortschritt
  darueber hinaus.

### Zurueck in die echte Engine

`ai/v1_1/agent.py` macht die Policy spielbar. Anders als die PPO-Policy aus v1_0,
die aus einem Frame-Stack heraus jedes Mal neu entscheidet, traegt der Dreamer
einen rekurrenten Zustand ueber die Episode - `reset()` zwischen Episoden ist
Pflicht. `DreamerController` besetzt jeden Sitz und kann in der bestehenden Arena
gegen v1_0-Modelle antreten.

```bash
python -m ai.v1_1.evaluate --run ai/output/v1_1/<lauf> --episodes 20
```

Misst `dreamer`, `planner` und `rules` auf identischen Startseeds - gepaart, weil
die Startlage die Ueberlebensdauer stark streut.

**Wie das Ergebnis zu lesen ist.** Den Planer solo zu schlagen ist schwer und
nicht der eigentliche Massstab: der Planer sieht das exakte Gitter, der Dreamer
nur 64x64 Pixel plus Sensoren. Sein Vorteil liegt woanders - er kann lernen, was
Gegner als naechstes zeichnen, was der Planer prinzipiell nicht kann, weil er das
Gitter einfriert. Bleibt der Dreamer solo unter dem Planer, mit Gegnern aber
darueber, ist genau das eingetreten. Das ist der Erfolg, nicht ein Widerspruch.

## Vier Korrekturen am Lernaufbau (Patch 4)

Alle vier betreffen Fehler, bei denen das Training weiterhin sauber durchlaeuft
und plausible Zahlen ausgibt - sie faellt ohne Test niemandem auf.

### 1. Terminals sind unter 0,3% der Schritte

Gemessen an einem echten Datensatz: 6 Tode auf 3041 Ticks. Der Fortsetzungskopf
erreichte damit 99,8% Trefferquote, indem er ausnahmslos "geht weiter" sagte -
und der Actor sah **in der Imagination nie einen Tod**. Drei Gegenmassnahmen:

- `terminal_fraction: 0.25` - ein Viertel der Batch-Sequenzen wird so gelegt,
  dass ein Tod garantiert darin liegt (vorher: etwa jedes achte Fenster)
- `continue_pos_weight: 50.0` - Klassengewichtung im Fortsetzungsverlust
- `terminal_recall` als eigene Kennzahl; die Gesamtquote liegt ohnehin ueber 99%
  und sagt nichts

Wirkung im Rauchtest: die getraeumte Fortsetzungswahrscheinlichkeit faellt von
konstant 0,994 auf 0,835 - das Modell sagt jetzt tatsaechlich Tode voraus.

### 2. Restlebensdauer als dichtes Ziel

Das in Patch 1 mitgespeicherte `planner_survived` war **wertlos**: konstant 60,
also immer gleich dem Planer-Horizont - der Planer weicht ja aus und sieht nie
einen Tod kommen. Ersetzt durch `ticks_to_death`, rueckwaerts aus jeder Episode
gezaehlt und bei 120 gedeckelt. Dieselbe Information wie das Terminal, aber an
JEDEM Schritt statt als 0,3%-Ereignis - der Haupthebel dafuer, dass der latente
Zustand ueberhaupt Todesnaehe kodiert.

Wird im Replay berechnet, nicht beim Sammeln: **bestehende Datensaetze
funktionieren unveraendert weiter.** Eine per Zeitlimit abgeschnittene Episode
zaehlt bewusst nicht runter - sonst lernte das Modell, dass die Zeit an sich
toetet.

Der neue `survival_head` liefert nebenbei die wichtigste Diagnose waehrend des
Policy-Trainings: faellt `Traum-ttd` waehrend die Rendite steigt, laeuft der
Actor in gefaehrlichere Zustaende und nutzt ein Leck im Weltmodell aus.

### 3. Gemischte Experten-Qualitaeten (`planner_mix`)

Ein einheitlich starker Planer ueberlebt tausende Ticks und stirbt fast nie -
die Ursache von Problem 1 liegt schon im Datensatz. Jetzt werden 50% der
Episoden mit dem starken Planer gesammelt (Vorbild fuer die Verhaltensklonung),
30% mittel, 20% mit kurzem Horizont und schmalem Beam. Die schwachen sterben oft
und liefern die Aufpralle. Gemessen: Anteil kurzer Episoden (<300 Ticks) steigt
von 17% auf 33%, kuerzeste Episode von 146 auf 23 Ticks.

### 4. KL-Anker gegen Abdriften (`bc_kl_scale`)

Das klassische Offline-RL-Problem: der Actor optimiert gegen ein Weltmodell, das
nur die Zustaende kennt, die der Planer besucht hat. Verlaesst er diese Region,
sagt das Modell dort irgendetwas vorher - oft etwas zu Optimistisches, denn
nichts hat ihm je widersprochen - und der Actor lernt begeistert, genau dorthin
zu laufen. Im Traum sieht das nach Fortschritt aus, in der Engine stirbt er.
Gegenmittel: eine eingefrorene Kopie des BC-Actors und ein KL-Term, der die
Policy in ihrer Naehe haelt. `bc_kl_scale: 0` schaltet ihn ab - sinnvoll erst
mit online nachgesammelten Daten.

### Ausserdem: Train/Val-Split

Getrennt ueber ganze **Episoden**, nicht ueber Schritte: zwei Fenster derselben
Episode ueberlappen stark, ein schrittweiser Split wuerde nur Auswendiglernen
messen. Die Traum-Diagnose laeuft seither auf dem Val-Split. Ist er zu klein fuer
aussagekraeftige Zahlen, warnt das Training explizit.

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
