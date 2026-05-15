#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_ACTIVATE="${ENV_ACTIVATE:-/usr/WS2/kur1/ML_backscatter/mlbs_predmodel/myenv_mlbs_local/bin/activate}"

if [[ ! -f "${ENV_ACTIVATE}" ]]; then
  echo "Python environment activation script not found: ${ENV_ACTIVATE}" >&2
  exit 2
fi

# shellcheck disable=SC1090
source "${ENV_ACTIVATE}"

CONFIG_PATH="${CONFIG_PATH:-/usr/WS2/kur1/ML_backscatter/adept/configs/sbsbs-1d/sbsbs_cbet_nbeams.yaml}"
DATA_PATH="${DATA_PATH:-${SCRIPT_DIR}/Fake_ML_Backscatter_Data.npz}"
OUTDIR="${OUTDIR:-/g/g15/kur1/ws/ML_backscatter/mlbs_predmodel/runs/sbsbs_train/debug_eval}"
SHOT_INDICES="${SHOT_INDICES:-0,1}"
TIME_INDICES="${TIME_INDICES:-0,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64,68,72,76,80,84,88,92,96,100,104,108,112,116,120,124}"
WORKERS="${WORKERS:-8}"
CORES_PER_WORKER="${CORES_PER_WORKER:-1}"
SEED="${SEED:-0}"
BASE_TEMPDIR="${BASE_TEMPDIR:-/tmp/kur1}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR_NAME="$(basename -- "${RUN_OUTDIR:-${OUTDIR}/debug_eval_${RUN_ID}}")"
RUN_OUTDIR="${RUN_OUTDIR:-${OUTDIR}/debug_eval_${RUN_ID}}"
LOG_DIR="${LOG_DIR:-${OUTDIR}/logs}"
STDOUT_LOG="${STDOUT_LOG:-${LOG_DIR}/${RUN_DIR_NAME}.out}"
STDERR_LOG="${STDERR_LOG:-${LOG_DIR}/${RUN_DIR_NAME}.err}"

mkdir -p "${LOG_DIR}"

CMD=(
  srun -p pdebug -N 1 -A wbronze --exclusive
  -o "${STDOUT_LOG}"
  -e "${STDERR_LOG}"
  python "${SCRIPT_DIR}/sbsbs_debug_eval.py"
  --config "${CONFIG_PATH}"
  --data "${DATA_PATH}"
  --outdir "${RUN_OUTDIR}"
  --shot-indices "${SHOT_INDICES}"
  --time-indices "${TIME_INDICES}"
  --workers "${WORKERS}"
  --cores-per-worker "${CORES_PER_WORKER}"
  --seed "${SEED}"
  --base-tempdir "${BASE_TEMPDIR}"
)

if [[ -n "${WEIGHTS_PATH:-}" ]]; then
  CMD+=(--weights "${WEIGHTS_PATH}")
fi

if [[ -n "${MAX_CASES:-}" ]]; then
  CMD+=(--max-cases "${MAX_CASES}")
fi

case "${XLA_CPU_MULTI_THREAD_EIGEN:-default}" in
  1|true|TRUE|yes|YES|on|ON)
    CMD+=(--xla-cpu-multi-thread-eigen)
    ;;
  0|false|FALSE|no|NO|off|OFF)
    CMD+=(--no-xla-cpu-multi-thread-eigen)
    ;;
  default|"")
    ;;
  *)
    echo "Unsupported XLA_CPU_MULTI_THREAD_EIGEN value: ${XLA_CPU_MULTI_THREAD_EIGEN}" >&2
    exit 2
    ;;
esac

CMD+=("$@")

exec "${CMD[@]}"
