#!/usr/bin/env bash
# Pod payload. Runs DETACHED on the RunPod 5090. Phase-1 of
# docs/PLAN_expressive_adversary.md: the feature-conditioned adversary search
# vs B_liu (the competitor), then an optional reduced head-to-head for context.
# Writes a STATUS file the local orchestrator polls; survives SSH disconnect.
set -uo pipefail

RUN_TS="${1:?usage: remote_run.sh RUN_TS}"
RUN_DIR="/workspace/run_${RUN_TS}"
POD="${RUN_DIR}/_pod"
mkdir -p "${POD}"
STATUS="${POD}/STATUS"
export PYTHONUNBUFFERED=1

log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "${POD}/run.log"; }
set_status(){ echo "$1" > "${STATUS}"; log "STATUS=$1"; }
fail(){ log "FATAL: $*"; set_status FAILED; exit 1; }

set_status SETUP
cd "${RUN_DIR}" || fail "no run dir ${RUN_DIR}"

# --- pull_pack helper (LF-safe: written here on the pod) so interim pulls work
cat > "${POD}/pull_pack.sh" <<'PACK'
#!/usr/bin/env bash
RUN_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TS="$(basename "$RUN_DIR" | sed 's/^run_//')"
cd /workspace || exit 1
SETS="run_${TS}/_pod"
[ -d "run_${TS}/results/flip_flop/adversary/quick_feature_controlled" ] && \
  SETS="$SETS run_${TS}/results/flip_flop/adversary/quick_feature_controlled"
[ -d "run_${TS}/results/flip_flop/eval_headtohead_fc" ] && \
  SETS="$SETS run_${TS}/results/flip_flop/eval_headtohead_fc"
tar czf "pull_${TS}.tmp.tgz" $SETS 2>/dev/null
mv -f "pull_${TS}.tmp.tgz" "pull_${TS}.tgz"
echo "/workspace/pull_${TS}.tgz"
PACK
chmod +x "${POD}/pull_pack.sh"

{ echo "run_ts=${RUN_TS}"; date -u;
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader;
  python -c "import torch;print('torch',torch.__version__,'cuda_ok',torch.cuda.is_available(),torch.cuda.get_device_name(0))";
} > "${POD}/metadata.txt" 2>&1 || true

log "pip install transformers (torch already Blackwell-ready)"
python -m pip install --no-input -q --break-system-packages --root-user-action=ignore transformers >> "${POD}/pip.log" 2>&1 || fail "pip transformers"
python - >> "${POD}/pip.log" 2>&1 <<'PY' || fail "import check failed"
import torch, transformers, yaml, numpy
import flip_flop.adversary.run            # exercises the wired pipeline
import expressive_adversary.distribution  # the controller (single source of truth)
print("imports OK", transformers.__version__)
PY
log "env OK"

# ---- Phase 1: feature-controlled adversary search vs B_liu ----
SEARCH_OUT="results/flip_flop/adversary/quick_feature_controlled"
set_status SEARCH
log "Phase-1 search start (target=B_liu, skyline=LSTM)"
if python -u -m flip_flop.scripts.run_adversary \
      --config flip_flop/configs/quick_adversary_feature_controlled.yaml \
      --transformer_ckpt results/flip_flop/full_from_scratch_liu_only/model_final.pt \
      --transformer_cfg  results/flip_flop/full_from_scratch_liu_only/config.yaml \
      --out_dir "${SEARCH_OUT}" \
      > "${POD}/search.log" 2>&1; then
  log "Phase-1 search COMPLETE"
  set_status SEARCH_DONE
else
  fail "search failed (see _pod/search.log)"
fi

# ---- Optional reduced head-to-head context (non-fatal; search is the result) ----
set_status EVAL
log "reduced head-to-head (optional, guarded)"
set +e
python -u -m flip_flop.scripts.eval_headtohead \
  --n 4096 --bootstrap 1000 --n_periodic 6 --models "" \
  --eval_batch_size 128 \
  --out_dir results/flip_flop/eval_headtohead_fc \
  > "${POD}/eval.log" 2>&1
EV=$?
set -e
if [ ${EV} -eq 0 ]; then log "head-to-head OK"; else
  log "head-to-head skipped/failed rc=${EV} (search evidence already captured)"; fi

# ---- collect ----
log "collecting artifacts + manifest"
mkdir -p "${POD}/artifacts"
cp -r "${SEARCH_OUT}" "${POD}/artifacts/quick_feature_controlled" 2>/dev/null || true
cp -r results/flip_flop/eval_headtohead_fc "${POD}/artifacts/eval_headtohead_fc" 2>/dev/null || true
( cd "${RUN_DIR}" && find _pod results/flip_flop/adversary/quick_feature_controlled \
    results/flip_flop/eval_headtohead_fc -type f 2>/dev/null ) > "${POD}/MANIFEST.txt" 2>/dev/null || true

set_status DONE
log "ALL DONE — safe to pull + terminate"
