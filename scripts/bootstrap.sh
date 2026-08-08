#!/usr/bin/env bash
# Frisch gemietete GPU-Maschine (RunPod / Vast.ai / Lambda) einsatzbereit machen.
#
#   bash scripts/bootstrap.sh
#
# Idempotent: laesst sich nach einem Neustart des Containers gefahrlos erneut
# aufrufen. Auf RunPod ueberlebt NUR das Volume unter /workspace einen Neustart -
# deshalb liegen Repo und Ausgabe dort, nicht unter /root.
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
REPO_DIR="${REPO_DIR:-$WORKSPACE/achtung-die-kurve_modified}"
REPO_URL="${REPO_URL:-https://github.com/jeremiassaur-2002/achtung-die-kurve_modified.git}"

echo "[bootstrap] Systempakete"
apt-get update -qq
apt-get install -y -qq tmux git ffmpeg htop >/dev/null

if [ ! -d "$REPO_DIR/.git" ]; then
  echo "[bootstrap] Repo klonen -> $REPO_DIR"
  git clone "$REPO_URL" "$REPO_DIR"
else
  echo "[bootstrap] Repo vorhanden, aktualisiere"
  git -C "$REPO_DIR" pull --ff-only || echo "[bootstrap] pull uebersprungen (lokale Aenderungen)"
fi

cd "$REPO_DIR"
echo "[bootstrap] Python-Abhaengigkeiten"
pip install -q --upgrade pip
pip install -q -r ai/requirements.txt

# Ausgabe auf das persistente Volume legen. Ohne das landen Checkpoints auf der
# Container-Disk und sind nach dem naechsten Neustart weg.
export AI_OUTPUT_ROOT="${AI_OUTPUT_ROOT:-$WORKSPACE/output}"
mkdir -p "$AI_OUTPUT_ROOT"
grep -q AI_OUTPUT_ROOT ~/.bashrc 2>/dev/null || echo "export AI_OUTPUT_ROOT=$AI_OUTPUT_ROOT" >> ~/.bashrc

echo "[bootstrap] Selbsttest"
cd "$REPO_DIR" && python3 -m pytest ai/tests -q --timeout=300 2>/dev/null || python3 -m pytest ai/tests -q
python3 -m ai.run info

cat <<MSG

[bootstrap] fertig.
  Repo:    $REPO_DIR
  Ausgabe: $AI_OUTPUT_ROOT

Training starten (ueberlebt geschlossenen Browser):
  bash scripts/train.sh v1_0 ai/v1_0/config/phase1.yaml
MSG
