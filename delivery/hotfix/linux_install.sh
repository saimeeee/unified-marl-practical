#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
PYTHON="$ROOT/.venv/bin/python"
[[ -x "$PYTHON" ]] || { printf 'Existing engineering .venv required. Install the base bundle first.\n' >&2; exit 1; }
exec "$PYTHON" -X utf8 -I "$ROOT/.unified-marl-hotfix/0.8.2/feature_install.py" --bundle "$ROOT" "$@"
