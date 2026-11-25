#!/usr/bin/env bash
# Reproduce every experiment reported in results/RESULTS.md for
# "A Kernel Framework for Linear Operator Equations" (Filipovic and
# Tissot-Daguette, arXiv:2611.31626).
set -e

cd "$(dirname "$0")"

python -m pytest -m "not market_data"

python -m exp.exp1
python -m exp.exp2
python -m exp.exp3
python -m exp.exp4
python -m exp.exp5
python -m exp.exp6
python -m exp.exp7

echo "Done. Metrics and figures are in results/; see results/RESULTS.md for the paper comparison."
