#!/bin/bash
# Phase B: diagnostic round. R1 adversary + R2 retrain + R2 adversary.
# Goal: validate Fix 2 selection_log.jsonl and Fix 3a cma_trajectory.jsonl on
# real H100 data before committing to the remaining fixes. NO ROUND 3.
# NO automatic teardown; my-side polling drives teardown after FINAL_DONE.
set -uo pipefail
cd /workspace/adversary_ICL
mkdir -p logs flip_flop/configs

# --- R2 retrain config (init from r1 model, families from r1 adversary)
cat > flip_flop/configs/retrain_r2.yaml << 'YAML'
retrain:
  base_p_i: 0.8
  replay_frac: 0.5
  init_from_ckpt: results/flip_flop/retrain/round_1/model_final.pt
  family_sources:
    - log: results/flip_flop/adversary_r1/stationary/adversary_log.jsonl
      top_k: 5
    - log: results/flip_flop/adversary_r1/piecewise/adversary_log.jsonl
      top_k: 5
    - log: results/flip_flop/adversary_r1/planted_decoy/adversary_log.jsonl
      top_k: 5

model:
  family: gpt2
  vocab_size: 5
  n_positions: 512
  n_embd: 512
  n_layer: 6
  n_head: 8
  resid_pdrop: 0.0
  embd_pdrop: 0.0
  attn_pdrop: 0.0

data:
  seq_len: 512
  train_p_i: 0.8
  eval_in_p_i: 0.8
  eval_sparse_p_i: 0.98
  eval_dense_p_i: 0.1
  eval_in_n: 1000
  eval_sparse_n: 10000
  eval_dense_n: 3000
  eval_seed: 1

training:
  lr: 3.0e-5
  weight_decay: 0.1
  beta1: 0.9
  beta2: 0.999
  warmup_steps: 100
  train_steps: 2000
  decay_end_step: 2001
  batch_size: 16
  grad_clip: 1.0
  seed: 0
  eval_every: 200
  training_eval_subset: 1000
  log_every: 50
  save_every: 0
  eval_batch_size: 64
  device: auto
  selection_enabled: true
  lambda_penalty: 1.5
  baseline_in_dist_glitch: 0.0
  in_dist_hard_cap: 0.005
  family_eval_n: 2000
  plateau_window: 5
  plateau_tol: 0.005

output:
  out_dir: results/flip_flop/retrain/round_2
YAML

# --- R1 and R2 adversary configs (sed-substitute from base templates)
for R in 1 2; do
  if [ "$R" = "1" ]; then
    CKPT=results/flip_flop/retrain/round_1/model_final.pt
  else
    CKPT=results/flip_flop/retrain/round_2/model_final.pt
  fi
  for strategy in stationary piecewise planted; do
    sed -e "s|results/flip_flop/baseline/model_final.pt|${CKPT}|g" \
        -e "s|out_dir: results/flip_flop/adversary/|out_dir: results/flip_flop/adversary_r${R}/|g" \
        -e "s|final_eval_n: 100000|final_eval_n: 20000|g" \
        flip_flop/configs/adversary_${strategy}.yaml > flip_flop/configs/adversary_${strategy}_r${R}.yaml
  done
done

run_adv () {
  R=$1
  STRAT=$2
  echo "=== R${R} ADV ${STRAT} $(date -u) ===" | tee -a logs/PROGRESS
  python3 -u -m flip_flop.scripts.run_adversary --config flip_flop/configs/adversary_${STRAT}_r${R}.yaml > logs/r${R}_${STRAT}.log 2>&1
}

# === R1 adversary against existing round-1 model
run_adv 1 stationary
run_adv 1 planted
run_adv 1 piecewise
touch logs/R1_ADV_DONE

# === R2 retrain (with new instrumentation: selection_log.jsonl, hard cap, plateau)
echo "=== R2 RETRAIN $(date -u) ===" | tee -a logs/PROGRESS
python3 -u -m flip_flop.scripts.run_retrain --config flip_flop/configs/retrain_r2.yaml > logs/r2_retrain.log 2>&1
touch logs/R2_RETRAIN_DONE

# === R2 adversary against round-2 model
run_adv 2 stationary
run_adv 2 planted
run_adv 2 piecewise
touch logs/R2_ADV_DONE

# Tar all results. NO teardown — my side does that after pulling.
tar czf /tmp/phase_b_results.tar.gz logs/ results/flip_flop/adversary_r1/ results/flip_flop/adversary_r2/ results/flip_flop/retrain/ 2>/dev/null
ls -la /tmp/phase_b_results.tar.gz > logs/SAVE_COMPLETE
date -u >> logs/SAVE_COMPLETE
touch logs/FINAL_DONE
echo "=== ALL DONE $(date -u) ===" | tee -a logs/PROGRESS
