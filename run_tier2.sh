#!/bin/bash
# Tier 2: full adversary battery against R2-redone-v2 model.
# Budget ~$7 of H100 (~120 min). NO Liu R4 (out of budget; cite paper).
# NO retrain (R2-redone-v2 model is pre-loaded).
#
# Steps:
#   1. Generate adversary configs pointing at R2-redone-v2 checkpoint.
#   2. Run stationary grid (~25 min).
#   3. Run planted grid (~18 min).
#   4. Run piecewise CMA, budget=1500 (~75 min).
#   5. Tar + FINAL_DONE.
#
# NO automatic teardown.
set -uo pipefail
cd /workspace/adversary_ICL
mkdir -p logs flip_flop/configs

# Generate R2-v2 adversary configs (sed substitute baseline ckpt path,
# out_dir, final_eval_n).
CKPT=results/flip_flop/retrain/round_2_redone_v2/model_final.pt
for strategy in stationary piecewise planted; do
  sed -e "s|results/flip_flop/baseline/model_final.pt|${CKPT}|g" \
      -e "s|out_dir: results/flip_flop/adversary/|out_dir: results/flip_flop/adversary_r2v2/|g" \
      -e "s|final_eval_n: 100000|final_eval_n: 20000|g" \
      flip_flop/configs/adversary_${strategy}.yaml > flip_flop/configs/adversary_${strategy}_r2v2.yaml
done

run_adv () {
  local STRAT=$1
  echo "=== R2v2 ADV ${STRAT} $(date -u) ===" | tee -a logs/PROGRESS
  python3 -u -m flip_flop.scripts.run_adversary --config flip_flop/configs/adversary_${STRAT}_r2v2.yaml > logs/r2v2_${STRAT}.log 2>&1
  echo "  exit=$? $(date -u)" | tee -a logs/PROGRESS
}

run_adv stationary
run_adv planted
run_adv piecewise

# Tar all results
tar czf /tmp/tier2_results.tar.gz \
  logs/ \
  results/flip_flop/adversary_r2v2/ \
  flip_flop/configs/adversary_stationary_r2v2.yaml \
  flip_flop/configs/adversary_piecewise_r2v2.yaml \
  flip_flop/configs/adversary_planted_r2v2.yaml \
  2>/dev/null

ls -la /tmp/tier2_results.tar.gz > logs/SAVE_COMPLETE
date -u >> logs/SAVE_COMPLETE
echo "=== TIER2 ALL DONE $(date -u) ===" | tee -a logs/PROGRESS
touch logs/FINAL_DONE
