#!/usr/bin/env bash
set -euo pipefail

# Generate connected non-isomorphic graphs in graph6 format using nauty's geng.
# Defaults are conservative for local experimentation and can be overridden.

usage() {
  cat <<'EOF'
Usage:
  scripts/generate_connected_g6.sh [options]

Options:
  --min N            Minimum number of vertices (default: 2)
  --max N            Maximum number of vertices (default: 8)
  --out-dir PATH     Output directory for .g6 files (default: pyrigi/graphDB/outputs/g6)
  --mode MODE        File handling mode: skip|overwrite (default: skip)
  --compress         Gzip each .g6 file after generation
  --dry-run          Print planned actions without running geng
  -h, --help         Show this help message

Environment variable overrides:
  MIN_N, MAX_N, OUT_DIR, MODE, COMPRESS, DRY_RUN

Examples:
  scripts/generate_connected_g6.sh
  scripts/generate_connected_g6.sh --max 10 --mode overwrite
  MIN_N=3 MAX_N=9 scripts/generate_connected_g6.sh --compress
EOF
}

MIN_N="${MIN_N:-2}"
MAX_N="${MAX_N:-8}"
OUT_DIR="${OUT_DIR:-pyrigi/graphDB/outputs/g6}"
MODE="${MODE:-skip}"
COMPRESS="${COMPRESS:-0}"
DRY_RUN="${DRY_RUN:-0}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --min)
      MIN_N="$2"
      shift 2
      ;;
    --max)
      MAX_N="$2"
      shift 2
      ;;
    --out-dir)
      OUT_DIR="$2"
      shift 2
      ;;
    --mode)
      MODE="$2"
      shift 2
      ;;
    --compress)
      COMPRESS=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if ! [[ "$MIN_N" =~ ^[0-9]+$ && "$MAX_N" =~ ^[0-9]+$ ]]; then
  echo "Error: --min and --max must be integers." >&2
  exit 1
fi

if (( MIN_N < 2 )); then
  echo "Error: minimum n must be >= 2." >&2
  exit 1
fi

if (( MAX_N < MIN_N )); then
  echo "Error: --max must be >= --min." >&2
  exit 1
fi

if [[ "$MODE" != "skip" && "$MODE" != "overwrite" ]]; then
  echo "Error: --mode must be either 'skip' or 'overwrite'." >&2
  exit 1
fi

if ! command -v geng >/dev/null 2>&1; then
  echo "Error: geng not found. Install nauty/gtools first (e.g. 'brew install nauty')." >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

echo "Starting generation: n=${MIN_N}..${MAX_N}, mode=${MODE}, compress=${COMPRESS}, dry_run=${DRY_RUN}"

generated=0
skipped=0
failed=0

for (( n=MIN_N; n<=MAX_N; n++ )); do
  out_file="$OUT_DIR/connected_n${n}.g6"

  if [[ -f "$out_file" && "$MODE" == "skip" ]]; then
    echo "[skip] n=$n exists: $out_file"
    skipped=$((skipped + 1))
    continue
  fi

  echo "[run ] n=$n -> $out_file"

  if (( DRY_RUN == 1 )); then
    generated=$((generated + 1))
    continue
  fi

  if geng -c "$n" > "$out_file"; then
    if (( COMPRESS == 1 )); then
      gzip -f "$out_file"
      out_file="${out_file}.gz"
    fi
    if [[ "$out_file" == *.gz ]]; then
      graph_count=$(gzip -dc "$out_file" | wc -l)
    else
      graph_count=$(wc -l < "$out_file")
    fi
    echo "[ ok ] n=$n graphs=$graph_count file=$out_file"
    generated=$((generated + 1))
  else
    echo "[fail] n=$n"
    failed=$((failed + 1))
  fi
done

echo "Done. generated=$generated skipped=$skipped failed=$failed"

if (( failed > 0 )); then
  exit 1
fi
