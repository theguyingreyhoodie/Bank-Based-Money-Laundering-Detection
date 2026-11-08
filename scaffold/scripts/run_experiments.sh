#!/usr/bin/env bash
# Every experiment behind the report, in one go:
#   main runs (configs/default.yaml):              baselines + the 4 FL modes, 5 seeds
#   sensitivity (configs/sensitivity_signal0.1.yaml): baselines + hybrid + zero-filled, 3 seeds
# then the seed summaries, the report figures and the dashboard's alerts. About an hour on a laptop.
#
# A run whose results file already exists is skipped, so after an interruption just start it again.
# To redo everything, first move the old experiments/results/*_s*.csv files out of the way.
#
#   bash scripts/run_experiments.sh
#   PYTHON=~/.venvs/aml-fl/bin/python bash scripts/run_experiments.sh   # a venv outside paths with spaces
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
MAIN=configs/default.yaml
SENS=configs/sensitivity_signal0.1.yaml

results_dir() {
  "$PY" -c "import sys; sys.path.insert(0, 'src'); from aml_fl.utils import load_config, results_dir; print(results_dir(load_config('$1'), '.'))"
}

experiments() {  # experiments <config> "<seeds>" <fl modes...>
  local cfg=$1 seeds=$2 res
  shift 2
  res=$(results_dir "$cfg")
  for seed in $seeds; do
    [ -f "$res/baselines_s$seed.csv" ] || "$PY" scripts/run_baselines.py --config "$cfg" --seed "$seed"
    for mode in "$@"; do
      [ -f "$res/fl_${mode}_s$seed.csv" ] || "$PY" scripts/run_fl.py --config "$cfg" --mode "$mode" --seed "$seed"
    done
  done
  "$PY" scripts/summarize_seeds.py --config "$cfg"
}

# Data preparation takes under a minute per config and gives the same files every time.
"$PY" scripts/prepare_data.py --config "$MAIN"
"$PY" scripts/prepare_data.py --config "$SENS"

experiments "$MAIN" "42 43 44 45 46" hybrid zero_filled encoder_head naive
experiments "$SENS" "42 43 44" hybrid zero_filled

"$PY" scripts/make_figures.py
"$PY" scripts/build_alerts.py fl_hybrid_s42
echo "All experiments done."
