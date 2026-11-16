#!/usr/bin/env bash
# Train (auto-resume on crash, e.g. transient CUDA OOM on a shared laptop GPU) then evaluate runs.
# Usage: scripts/neural/run_chain.sh <run_name> [<run_name> ...]   (config = configs/neural/<run>.toml)
set -u
PY="${PY:-$LOCALAPPDATA/mtah/venv-neural/Scripts/python.exe}"
export PYTHONPATH=src
mkdir -p artifacts/neural/logs
for run in "$@"; do
  ok=0
  for attempt in 1 2 3 4; do
    "$PY" scripts/neural/train.py --config "configs/neural/$run.toml" --resume \
      >> "artifacts/neural/logs/$run.train.log" 2>&1 && { ok=1; break; }
    echo "[$run] train attempt $attempt failed; resuming" >> "artifacts/neural/logs/$run.train.log"
  done
  [ "$ok" = 1 ] || { echo "$run FAILED"; continue; }
  "$PY" scripts/neural/evaluate.py --run "$run" > "artifacts/neural/logs/$run.eval.log" 2>&1
  [ "${SKIP_SC:-0}" = 1 ] || "$PY" scripts/neural/evaluate.py --run "$run" --tag _sc_only --test skillcorner_1925299 \
    skillcorner_1986691 skillcorner_2006363 skillcorner_2011166 skillcorner_2017461 \
    > "artifacts/neural/logs/$run.eval_sc.log" 2>&1
  echo "$run done (eval exit $?)"
done
