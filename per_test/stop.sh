#!/usr/bin/env bash
# Graceful counterpart to run.sh: never uses pkill -9 / kill -9.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
exec python3 "$HERE/job_control.py" stop "$@"
