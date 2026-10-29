#!/usr/bin/env bash
set -euo pipefail
PLATFORM_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PLATFORM_DIR"
command -v docker >/dev/null || { echo 'Docker with Compose v2 is required.' >&2; exit 1; }
docker compose version
test "$(docker info --format '{{.OSType}}')" = linux || { echo 'Use a Linux Docker engine.' >&2; exit 1; }
for scene in warehouse mpe traffic powergrid; do
    test -f "bundle/models/$scene/policy.pt" || { echo "Missing local model for $scene. Use the complete GitHub local-platform download." >&2; exit 1; }
done
test -f bundle/vendor/CityFlow/setup.py || { echo 'Missing bundled CityFlow source.' >&2; exit 1; }
mkdir -p local-state
if [[ "${1:-}" == --rebuild ]] || ! docker image inspect unified-marl-live:local-v1 >/dev/null 2>&1; then
    docker compose --project-name unified-marl-live --file compose.yaml build platform
fi
docker compose --project-name unified-marl-live --file compose.yaml up --detach --no-build platform
echo "Local platform: http://127.0.0.1:${MARL_LOCAL_PORT:-18765}/"
echo 'Initialization may take a moment. Run docker compose down in this folder to stop; run data is retained.'
