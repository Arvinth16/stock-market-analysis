#!/bin/bash
set -e
cd "$(dirname "$0")"
git init
git checkout -b main
git add -A
git commit -m "chore(init): project skeleton and dependencies"
echo "---"
python3 -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
echo "=== SETUP COMPLETE ==="
