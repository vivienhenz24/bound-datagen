#!/usr/bin/env bash
set -euo pipefail
set -x

# Manual Runpod workflow.
# Assumes you have already created a pod and cloned this repo on the pod.
# Usage:
#   SSH_HOST=ssh.runpod.io SSH_USER=<podHostId> SSH_PORT=22 \
#   REMOTE_BASE=/workspace/bound-datagen \
#   bash runpod_tools/manual_pod_flow.sh

SSH_HOST="${SSH_HOST:-}"
SSH_PORT="${SSH_PORT:-22}"
SSH_USER="${SSH_USER:-root}"
REMOTE_BASE="${REMOTE_BASE:-/workspace/bound-datagen}"
LOCAL_OUTPUT="${LOCAL_OUTPUT:-finetuned_models/qwen3-1.5b-unsloth}"

if [[ -z "$SSH_HOST" ]]; then
  echo "SSH_HOST is required."
  exit 1
fi

SSH_BASE=(ssh -p "$SSH_PORT" -o StrictHostKeyChecking=no "${SSH_USER}@${SSH_HOST}")
RSYNC_SSH=(ssh -p "$SSH_PORT" -o StrictHostKeyChecking=no)

echo "[manual] Verifying remote repo..."
"${SSH_BASE[@]}" "ls -la ${REMOTE_BASE} && ls -la ${REMOTE_BASE}/runpod_tools"

echo "[manual] Syncing dataset and scripts..."
rsync -avv --progress -e "${RSYNC_SSH[*]}" finetune-data.jsonl runpod_tools/ "${SSH_USER}@${SSH_HOST}:${REMOTE_BASE}/"

echo "[manual] Starting remote training..."
"${SSH_BASE[@]}" "cd ${REMOTE_BASE} && bash runpod_tools/setup_and_train.sh"

echo "[manual] Pulling artifacts..."
mkdir -p "${LOCAL_OUTPUT}"
rsync -avv --progress -e "${RSYNC_SSH[*]}" \
  "${SSH_USER}@${SSH_HOST}:${REMOTE_BASE}/output/qwen3-1.5b-unsloth/" \
  "${LOCAL_OUTPUT}/"

echo "[manual] Done. Artifacts in ${LOCAL_OUTPUT}"
