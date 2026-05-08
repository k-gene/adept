#!/usr/bin/env bash
#SBATCH -p pdebug
#SBATCH -N 1
#SBATCH -A wbronze
#SBATCH --exclusive
#SBATCH -J sbsbs-smoke
#SBATCH -o /usr/WS2/kur1/ML_backscatter/adept/benchmarks/out/slurm-%j.out
#SBATCH -e /usr/WS2/kur1/ML_backscatter/adept/benchmarks/out/slurm-%j.err

set -euo pipefail

# Interactive allocation example:
#   salloc -p pdebug -N 1 -A wbronze --exclusive
# Then run:
#   bash /usr/WS2/kur1/ML_backscatter/adept/benchmarks/run_sbsbs_smoke_matrix.sh
#
# One-shot interactive example:
#   srun -p pdebug -N 1 -A wbronze --exclusive --pty \
#     bash /usr/WS2/kur1/ML_backscatter/adept/benchmarks/run_sbsbs_smoke_matrix.sh
#
# Batch submission example (works from any directory because the script uses absolute paths):
#   sbatch /usr/WS2/kur1/ML_backscatter/adept/benchmarks/run_sbsbs_smoke_matrix.sh

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "This script must be run inside a Slurm allocation." >&2
  echo "Examples:" >&2
  echo "  salloc -p pdebug -N 1 -A wbronze --exclusive" >&2
  echo "  sbatch /usr/WS2/kur1/ML_backscatter/adept/benchmarks/run_sbsbs_smoke_matrix.sh" >&2
  exit 1
fi

REPO_ROOT="/usr/WS2/kur1/ML_backscatter/adept"
PYTHON_BIN="/usr/WS2/kur1/ML_backscatter/mlbs_predmodel/myenv_mlbs_local/bin/python"
TRAIN_SCRIPT="$REPO_ROOT/adept/sbsbs_train.py"
OUTDIR_ROOT="${1:-$REPO_ROOT/benchmarks/out/sbsbs_smoke_matrix}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="$OUTDIR_ROOT/$TIMESTAMP"
mkdir -p "$RUN_ROOT"

BASE_ARGS=(
  "$TRAIN_SCRIPT"
  --no-mlflow
  --max-shots 1
  --max-timepoints 8
  --timepoints-per-task 1
  --save-every 1000000
  --log-every 1
)

# CASES=(
#   "w1_c1_xlaoff|--workers 1 --cores-per-worker 1 --no-xla-cpu-multi-thread-eigen"
#   "w1_c4_xlaoff|--workers 1 --cores-per-worker 4 --no-xla-cpu-multi-thread-eigen"
#   "w1_c4_xlaon|--workers 1 --cores-per-worker 4 --xla-cpu-multi-thread-eigen"
#   "w4_c1_xlaoff|--workers 4 --cores-per-worker 1 --no-xla-cpu-multi-thread-eigen"
#   "w4_c4_xlaoff|--workers 4 --cores-per-worker 4 --no-xla-cpu-multi-thread-eigen"
#   "w4_c4_xlaon|--workers 4 --cores-per-worker 4 --xla-cpu-multi-thread-eigen"
# )
CASES=(
  "w8_c8_xlaon|--workers 8 --cores-per-worker 8 --xla-cpu-multi-thread-eigen"
  "w8_c4_xlaon|--workers 8 --cores-per-worker 4 --xla-cpu-multi-thread-eigen"
  "w8_c2_xlaon|--workers 8 --cores-per-worker 2 --xla-cpu-multi-thread-eigen"
  "w8_c1_xlaon|--workers 8 --cores-per-worker 1 --xla-cpu-multi-thread-eigen"
)

SUMMARY_CSV="$RUN_ROOT/summary.csv"
printf 'case,status,elapsed_wall_s,outdir,log_file,time_file\n' > "$SUMMARY_CSV"

echo "Writing benchmark artifacts under: $RUN_ROOT"
echo "Using Python: $PYTHON_BIN"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"

to_seconds() {
  local raw="$1"
  python3 - <<'PY' "$raw"
import sys
raw = sys.argv[1].strip()
mins, secs = raw.split(':')
print(int(mins) * 60 + float(secs))
PY
}

for entry in "${CASES[@]}"; do
  case_name="${entry%%|*}"
  case_args="${entry#*|}"
  case_dir="$RUN_ROOT/$case_name"
  stdout_log="$case_dir/stdout.log"
  time_log="$case_dir/time.log"
  mkdir -p "$case_dir"

  echo
  echo "=== Running $case_name ==="
  echo "Args: $case_args"

  status="success"
  if ! /usr/bin/time -v -o "$time_log" \
    "$PYTHON_BIN" "${BASE_ARGS[@]}" --outdir "$case_dir/runs" $case_args \
    > "$stdout_log" 2>&1; then
    status="failure"
  fi

  elapsed_raw="$(python3 - <<'PY' "$time_log"
import sys
from pathlib import Path
text = Path(sys.argv[1]).read_text()
needle = 'Elapsed (wall clock) time (h:mm:ss or m:ss): '
for line in text.splitlines():
    if needle in line:
        print(line.split(needle, 1)[1].strip())
        break
else:
    print('')
PY
)"

  elapsed_wall_s=""
  if [[ -n "$elapsed_raw" ]]; then
    if [[ "$elapsed_raw" == *:*:* ]]; then
      elapsed_wall_s="$(python3 - <<'PY' "$elapsed_raw"
import sys
raw = sys.argv[1].strip()
hours, mins, secs = raw.split(':')
print(int(hours) * 3600 + int(mins) * 60 + float(secs))
PY
)"
    else
      elapsed_wall_s="$(to_seconds "$elapsed_raw")"
    fi
  fi

  printf '%s,%s,%s,%s,%s,%s\n' \
    "$case_name" "$status" "$elapsed_wall_s" "$case_dir/runs" "$stdout_log" "$time_log" \
    >> "$SUMMARY_CSV"

  echo "Status: $status"
  echo "Elapsed wall time (s): ${elapsed_wall_s:-unknown}"
  echo "Log: $stdout_log"
  echo "Time: $time_log"

  grep -E "Training settings|Thread controls|Resolved run output directory|epoch=|shot_wall_s" "$stdout_log" || true
done

echo
echo "Done. Summary written to: $SUMMARY_CSV"
