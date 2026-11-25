#!/usr/bin/env sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
cd "$repo_root"

config="examples/research/three_free_providers.json"
if [ "$#" -gt 0 ]; then
  config=$1
  shift
fi

if ! command -v llm-compat >/dev/null 2>&1; then
  echo "error: llm-compat is not installed. From the checkout, run: python -m pip install -e ." >&2
  exit 1
fi

llm-compat research validate "$config"
printf '\nStarting the research run. It requires explicit authorization: --yes or an interactive "yes".\n'
exec llm-compat research run "$config" "$@"
