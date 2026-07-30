#!/usr/bin/env bash
# === Delta (NCSA) driver for demos/academy_mlip_committee.py ===============
#SBATCH -J academy-committee
#SBATCH -A bhhl-delta-gpu
#SBATCH -p gpuA100x4
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -t 01:45:00
#SBATCH -o academy-committee-%j.out
# ==========================================================================
#
# Runs the Academy x Rootstock query-by-committee demo on a Delta GPU node.
# All committee members share the one allocated GPU (device="cuda"); the
# cells are small, so contention is negligible next to model-load time.
#
# PATH propagation: like the nightly driver, this relies on SLURM's default
# --export=ALL carrying the submit shell's PATH into the job — submit from a
# shell where `uv` resolves (`which uv` works).
#
# Stage the demo (from rootstock-scratch/):
#   scp demos/academy_mlip_committee.py demos/mp_seeds.extxyz \
#       demos/academy_committee_delta.sbatch delta:~/
# (mp_seeds.extxyz comes from fetch_mp_structures.py, run wherever you have
#  an MP_API_KEY — the demo itself never talks to Materials Project.)
#
# Kickoff (from a dt-login node, in the staged directory):
#   sbatch academy_committee_delta.sbatch
# Extra args pass through to the script, e.g.:
#   sbatch academy_committee_delta.sbatch --rounds 8 --pool-size 5

set -euo pipefail

# Stream progress into the SLURM .out as it happens — without this, python
# block-buffers stdout under SLURM and the demo looks silent until exit.
export PYTHONUNBUFFERED=1

# Worker-side logging. NOT "stderr": the server redirects each worker's
# stdout/stderr to anonymous temp files that only surface in a
# WorkerDiedError post-mortem, so "stderr" is invisible for healthy
# workers. A file path (opened append, line-buffered) is watchable live;
# all workers share it, so lines interleave.
export ROOTSTOCK_WORKER_LOG="${SLURM_SUBMIT_DIR:-$PWD}/rootstock-workers-${SLURM_JOB_ID:-manual}.log"

# Warm-up throughput note (2026-07-29): the demo script now pins
# rootstock>=1.2, whose prewarm reads with parallel threads (#171,
# ROOTSTOCK_PREWARM_THREADS, default 8) — expect materially faster
# warm-ups than the 946-1322 s/env measured under 1.1.0's single stream.
# Model-weight caches are still NOT prewarmed — a large checkpoint (uma)
# loads cold after prewarm, which is why its warm-up dwarfs its siblings'
# (upstream fix in progress).

# HF token for gated checkpoints (uma). Weights are already cached in the
# install, but some setups re-check gating at load time.
if [[ -f ~/.cache/huggingface/token ]]; then
  export HF_TOKEN="$(cat ~/.cache/huggingface/token)"
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
else
  echo "WARNING: no HF token at ~/.cache/huggingface/token; uma may fail to load." >&2
fi

# Resolve this script's original path ($0 is the spool copy under SLURM) so
# the demo runs from the same directory regardless of submit cwd.
if command -v scontrol >/dev/null 2>&1 && [[ -n "${SLURM_JOB_ID:-}" ]]; then
  SELF_PATH="$(scontrol show job "$SLURM_JOB_ID" \
    | sed -n 's/.*Command=\([^ ]*\).*/\1/p' | head -1)"
fi
SELF_PATH="${SELF_PATH:-$0}"
SCRIPT_DIR="$(cd "$(dirname "$SELF_PATH")" && pwd)"

nvidia-smi | head -4 || true
echo "=== academy_mlip_committee.py (cluster=delta) ==="

# Quorum start: campaign begins at 3 warm members; the rest of the pool
# joins mid-flight. --timeout 2700 because six envs sharing cold Lustre
# bandwidth can push the slowest warm-ups past 1800 s; --rounds/--round-pause
# stretch the campaign so stragglers have rounds left to join. Trailing
# "$@" lets submit-time args override any of these (argparse last-wins).
uv run "$SCRIPT_DIR/academy_mlip_committee.py" \
  --cluster delta \
  --outdir "committee_results_${SLURM_JOB_ID:-manual}" \
  --timeout 2700 \
  --rounds 6 \
  --round-pause 60 \
  "$@"
