#!/usr/bin/env bash
# Training in einer tmux-Sitzung starten, die einen geschlossenen Browser,
# eine getrennte SSH-Verbindung und einen zugeklappten Laptop ueberlebt.
#
#   bash scripts/train.sh v1_0 ai/v1_0/config/phase1.yaml
#   bash scripts/train.sh v1_1 ai/v1_1/config/dreamer.yaml
#
# Danach:  tmux attach -t achtung     (loesen: Strg-B, dann D)
#
# Die tmux-Sitzung bekommt drei Fenster:
#   0 train    das Training selbst, in einer Neustartschleife
#   1 board    TensorBoard auf Port 6006
#   2 watch    nvidia-smi + Timing-Tabelle nebeneinander
#
# Die Neustartschleife ist der Kern der Ausfallsicherheit: stirbt das Training
# (OOM, CUDA-Fehler, unterbrochene Netzverbindung beim Speichern), startet es
# nach 30 Sekunden neu und nimmt dank `--resume auto` den juengsten Checkpoint
# wieder auf. Ein sauberer Abschluss (Exit-Code 0) beendet die Schleife.
set -euo pipefail

VERSION="${1:?Version fehlt: v1_0 oder v1_1}"
CONFIG="${2:?Config fehlt}"
SESSION="${SESSION:-achtung}"
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export AI_OUTPUT_ROOT="${AI_OUTPUT_ROOT:-$REPO_DIR/ai/output}"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "[train] Sitzung '$SESSION' laeuft bereits. Anhaengen: tmux attach -t $SESSION"
  exit 0
fi

TRAIN_CMD="cd $REPO_DIR && export AI_OUTPUT_ROOT=$AI_OUTPUT_ROOT && \
  until python3 -m ai.run train --version $VERSION --config $CONFIG --resume auto; do \
    echo '[train] abgestuerzt - Neustart in 30s'; sleep 30; \
  done; echo '[train] regulaer beendet'; exec bash"

tmux new-session  -d -s "$SESSION" -n train "bash -lc \"$TRAIN_CMD\""
tmux new-window   -t "$SESSION" -n board "bash -lc 'cd $REPO_DIR && tensorboard --logdir $AI_OUTPUT_ROOT --host 0.0.0.0 --port 6006'"
tmux new-window   -t "$SESSION" -n watch "bash -lc 'watch -n 5 \"nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,temperature.gpu --format=csv; echo; tail -n 25 $AI_OUTPUT_ROOT/$VERSION/*/timing/timing.md 2>/dev/null\"'"

cat <<MSG
[train] Sitzung '$SESSION' gestartet ($VERSION, $CONFIG)

  tmux attach -t $SESSION      Fenster wechseln: Strg-B dann 0/1/2
  Strg-B dann D                loesen (Training laeuft weiter)
  TensorBoard                  Port 6006 weiterleiten:
                                 ssh -L 6006:localhost:6006 <host>
MSG
