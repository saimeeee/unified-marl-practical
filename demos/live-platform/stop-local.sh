#!/usr/bin/env bash
set -euo pipefail
PLATFORM_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PLATFORM_DIR"
docker compose --project-name unified-marl-live --file compose.yaml down
echo 'Local platform stopped. Models and local-state run data were retained.'
