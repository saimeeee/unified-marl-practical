#!/usr/bin/env bash
set -euo pipefail
PLATFORM_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY="$(cd -- "$PLATFORM_DIR/../.." && pwd)"
cd "$PLATFORM_DIR"
rebuild=false
case "${1:-}" in
    --rebuild) rebuild=true ;;
    '') ;;
    *) echo 'Usage: bash start-local.sh [--rebuild]. Set MARL_LOCAL_PORT to change the port.' >&2; exit 1 ;;
esac
command -v docker >/dev/null || { echo 'Docker with Compose v2 is required.' >&2; exit 1; }
docker compose version
engine="$(docker info --format '{{.OSType}}')" || {
    echo 'The Docker engine is not ready. Open Docker Desktop; resolve any WSL/data-disk startup error before retrying. This launcher does not reset Docker/WSL data.' >&2
    exit 1
}
test "$engine" = linux || { echo 'Use Linux containers in Docker Desktop, then retry.' >&2; exit 1; }
for scene in warehouse mpe traffic powergrid; do
    test -f "bundle/models/$scene/policy.pt" || { echo "Missing local model for $scene. Use the complete GitHub local-platform download." >&2; exit 1; }
done
test -f bundle/vendor/CityFlow/setup.py || { echo 'Missing bundled CityFlow source.' >&2; exit 1; }
mkdir -p local-state
check_source() {
    docker run --rm --network none --entrypoint python \
        --mount "type=bind,source=$REPOSITORY,target=/marl-source,readonly" \
        unified-marl-live:local-v1 /marl-source/demos/live-platform/source_fingerprint.py --check /marl-source
}
image_id="$(docker image ls --quiet unified-marl-live:local-v1)"
if [[ -z "$image_id" ]]; then
    rebuild=true
elif [[ "$rebuild" == false ]]; then
    result=0
    check_source || result=$?
    if [[ "$result" -gt 1 ]]; then
        echo 'Could not verify source inputs. Check the error above.' >&2
        exit 1
    fi
    [[ "$result" == 0 ]] || rebuild=true
fi
if [[ "$rebuild" == true ]]; then
    echo 'Building the local CPU environment. Changed source is rebuilt automatically.'
    docker compose --project-name unified-marl-live --file compose.yaml build platform
    check_source || { echo 'Source changed during the build or the image is incomplete. Retry the launcher.' >&2; exit 1; }
fi
docker compose --project-name unified-marl-live --file compose.yaml up --detach --no-build platform || {
    echo "Could not start the platform. Check whether port ${MARL_LOCAL_PORT:-18765} is occupied; try MARL_LOCAL_PORT=18766 bash start-local.sh." >&2
    exit 1
}
container_id="$(docker compose --project-name unified-marl-live --file compose.yaml ps --quiet platform)"
ready=false
for ((attempt=0;attempt<90;attempt++)); do
    [[ "$(docker inspect --format '{{.State.Running}}' "$container_id")" == true ]] || break
    if docker exec "$container_id" python -c "import json,urllib.request; r=json.load(urllib.request.urlopen('http://127.0.0.1:18765/api/catalog',timeout=2)); assert r['mode']=='live_inference' and len(r['models'])>=4" >/dev/null 2>&1; then
        ready=true
        break
    fi
    sleep 2
done
if [[ "$ready" != true ]]; then
    docker compose --project-name unified-marl-live --file compose.yaml logs --tail 60 platform
    echo 'Local service did not become ready; check the log above.' >&2
    exit 1
fi
echo "Local platform: http://127.0.0.1:${MARL_LOCAL_PORT:-18765}/"
echo 'Run bash stop-local.sh to stop; run data is retained.'
