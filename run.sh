#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-harness}"
STATE_DIR="${STATE_DIR:-$HOME/harness-state}"
CPUS="${CPUS:-8}"
MEMORY="${MEMORY:-16g}"

# usage:
#   ./run.sh /path/to/project                    # TUI (default)
#   ./run.sh once "prompt" /path/to/project      # headless one-shot

MODE="tui"
if [ "${1:-}" = "once" ]; then
    MODE="once"
    shift
    PROMPT="${1:?usage: ./run.sh once \"prompt\" /path/to/project}"
    shift
fi

WORKSPACE="${1:?usage: ./run.sh [once \"prompt\"] /path/to/project}"
NAME="harness-$(basename "$WORKSPACE")"

mkdir -p "$WORKSPACE" "$STATE_DIR"

if [ "$MODE" = "tui" ]; then
    ENTRY_ARGS=(--entrypoint python "$IMAGE" -m harness.tui)
else
    ENTRY_ARGS=(--entrypoint python "$IMAGE" -m harness "$PROMPT")
fi

podman run -it --rm \
    --init \
    --name "$NAME" \
    --cpus "$CPUS" \
    --memory "$MEMORY" \
    -v "$WORKSPACE:/workspace:z" \
    -v "$STATE_DIR:/state:z" \
    --add-host "ai.kenhome.lan:$(getent hosts ai.kenhome.lan | awk '{print $1}')" \
    -e HARNESS_TLS_VERIFY=0 \
    "${ENTRY_ARGS[@]}"

