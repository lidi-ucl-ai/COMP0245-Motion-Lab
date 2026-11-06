#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ -n "${COMP0245_PYTHON:-}" ]]; then
  python_bin="$COMP0245_PYTHON"
  export LD_LIBRARY_PATH="$(dirname -- "$(dirname -- "$python_bin")")/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  exec "$python_bin" webapp/server.py "$@"
fi
if command -v pixi >/dev/null 2>&1; then
  exec pixi run demo "$@"
fi
exec python3 webapp/server.py "$@"
