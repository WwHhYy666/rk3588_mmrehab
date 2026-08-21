#!/usr/bin/env bash
set -euo pipefail

exec "$(dirname "$0")/switch_rehab_mode.sh" cpu_8082 "$@"
