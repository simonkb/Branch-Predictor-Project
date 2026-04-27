# Run all benchmarks across selected branch predictors
#!/usr/bin/env bash
set -euo pipefail

# Run from project root (directory containing gem5/, configs/, benchmarks/)
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

GEM5="$ROOT/gem5/build/RISCV/gem5.opt"
CFG="$ROOT/configs/se_riscv_bp.py"

# Predictors (your current setup)
PREDICTORS=(
  "BiModeBP"
  "GshareBP"
  "TAGE"
)

WORKLOADS=(
  "tinymlp:$ROOT/benchmarks/tinymlp/tinymlp_rv_static"
  "matmul:$ROOT/benchmarks/matmul/matmul_rv_static"
)

CPU_TYPE="O3CPU"
MAXINSTS="10000000"
CACHE_FLAGS=(--caches --l2cache)

OUTROOT="$ROOT/results"

# ---- checks
if [[ ! -x "$GEM5" ]]; then
  echo "ERROR: gem5 binary not found/executable at: $GEM5"
  exit 1
fi
if [[ ! -f "$CFG" ]]; then
  echo "ERROR: config script not found at: $CFG"
  exit 1
fi

mkdir -p "$OUTROOT"

echo "== gem5 run matrix =="
echo "gem5 : $GEM5"
echo "cfg  : $CFG"
echo "cpu  : $CPU_TYPE"
echo "I    : $MAXINSTS"
echo

for w in "${WORKLOADS[@]}"; do
  wname="${w%%:*}"
  wbin="${w#*:}"

  if [[ ! -f "$wbin" ]]; then
    echo "ERROR: workload binary not found: $wbin"
    exit 1
  fi

  for p in "${PREDICTORS[@]}"; do
    outdir="$OUTROOT/$wname/$p"
    rm -rf "$outdir"
    mkdir -p "$outdir"

    echo "--> workload=$wname predictor=$p"
    echo "    outdir=$outdir"

    "$GEM5" --outdir="$outdir" \
      "$CFG" \
      --cmd="$wbin" \
      --cpu-type="$CPU_TYPE" \
      --bp-type="$p" \
      "${CACHE_FLAGS[@]}" \
      -I "$MAXINSTS" \
      > "$outdir/run.log" 2>&1

    if [[ ! -s "$outdir/stats.txt" ]]; then
      echo "ERROR: stats.txt missing/empty for $wname $p"
      echo "See: $outdir/run.log"
      exit 1
    fi

    echo "    OK"
  done
done

echo
echo "All runs completed."
echo "See results under: $OUTROOT/<workload>/<predictor>/"