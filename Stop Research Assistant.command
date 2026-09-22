#!/bin/zsh
set -e
cd -- "$(dirname -- "$0")"
exec .venv/bin/python stop.py
