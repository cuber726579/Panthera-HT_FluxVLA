#!/usr/bin/env bash
set -euo pipefail

# 参数见 deploy/config/rollout_panthera.yaml；可从任意目录执行。
# 用法：bash deploy/rollout_panthera.sh [check|smoke|dry-run|rollout]
#       bash deploy/rollout_panthera.sh smoke --config /path/to/config.yaml
#       bash deploy/rollout_panthera.sh --print-command
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
exec python3 "$ROOT/deploy/launch_panthera.py" "$@"
