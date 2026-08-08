#!/usr/bin/env bash
# Ergebnisse aus ai/output sichern, ohne die Repo-History mit Gewichten zu fluten.
#
#   bash scripts/sync_output.sh v1_0            einmalig
#   watch -n 900 bash scripts/sync_output.sh v1_0   alle 15 Minuten
#
# Gepusht wird in den SEPARATEN Branch `results` und nur, was klein und
# wertvoll ist: das beste Modell, die Reports, das Timing, die Metriken. Die
# rotierenden Zwischen-Checkpoints bleiben lokal - sie sind gross, kurzlebig
# und nur fuer die Wiederaufnahme auf DIESER Maschine gedacht.
#
# Warum ein eigener Branch: ein `git push` von 200-MB-Dateien in main macht
# jeden spaeteren Klon des Repos dauerhaft langsam, weil Git-History nichts
# vergisst. `results` kann man notfalls verwerfen und neu anlegen.
set -euo pipefail

VERSION="${1:?Version fehlt: v1_0 oder v1_1}"
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
OUT="${AI_OUTPUT_ROOT:-$REPO_DIR/ai/output}/$VERSION"
BRANCH="${RESULTS_BRANCH:-results}"
STAGE="${STAGE_DIR:-/tmp/achtung-results}"

[ -d "$OUT" ] || { echo "[sync] $OUT existiert nicht"; exit 1; }

rm -rf "$STAGE"
git -C "$REPO_DIR" worktree prune
if git -C "$REPO_DIR" show-ref --verify --quiet "refs/heads/$BRANCH"; then
  git -C "$REPO_DIR" worktree add "$STAGE" "$BRANCH"
else
  git -C "$REPO_DIR" worktree add -b "$BRANCH" "$STAGE"
fi

DEST="$STAGE/$VERSION"
mkdir -p "$DEST"
# --delete waere hier falsch: ein abgebrochener Lauf soll seine bereits
# gesicherten Ergebnisse nicht verlieren.
rsync -a --include='*/' \
  --include='best/***' --include='reports/***' --include='timing/***' \
  --include='metrics/*.csv' --include='config_used.yaml' \
  --exclude='*' "$OUT/" "$DEST/"

cd "$STAGE"
git add -A
if git diff --cached --quiet; then
  echo "[sync] nichts Neues"
else
  git -c user.name="training-bot" -c user.email="bot@local" \
      commit -q -m "results $VERSION $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  git push -q origin "$BRANCH" && echo "[sync] gepusht nach $BRANCH"
fi
