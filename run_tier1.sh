#!/bin/bash
# Tier-1 diagnostic on H100. Total budget ~7 min, ~$0.41.
#
# Goal: did the 7 fixes stop the R1->R2 forgetting catastrophe? Without
# spending H100 cycles on a new adversary search.
#
# Steps:
#   1. Retrain R2-redone from R1 model with all fixes (~3-5 min).
#      This produces eval_battery.json (FFL test sets + new families)
#      and selection_log.jsonl (in-loop FFL/family/score trajectory).
#   2. Evaluate the retrained model on R1's SAVED top_k.jsonl files
#      (~3 min; 5000 seq * 25 configs * 3 axes * 2 models).
#   3. Consolidate everything into tier1_summary.json.
#
# NO adversary search. NO Liu R4. NO automatic teardown — my-side polling
# triggers runpodctl stop after pulling tier1_summary.json.

set -uo pipefail
cd /workspace/adversary_ICL
mkdir -p logs flip_flop/configs

# === Step 1: write the Tier-1 retrain config ===
# - init_from_ckpt: R1 model
# - family_sources: R1's three adversary logs
# - all 7 fixes are baked into family.py + train.py defaults
# - replay_frac: 0.7 (Step 7)
# - new threshold: 0.01 (Step 1, default)
# - new lambdas: in=5, 98=0.7, 01=0.7 (Step 4 defaults)
# - eval_every: 200 (so plateau check is meaningful)
# - hard caps: in=0.005, 98=0.005, 01=0.005 (Step 4 defaults)
# - memorize_warmup: 800 (Step 5 default)

cat > flip_flop/configs/retrain_r2_redone.yaml << 'YAML'
retrain:
  base_p_i: 0.8
  replay_frac: 0.7
  init_from_ckpt: results/flip_flop/retrain/round_1/model_final.pt
  family_sources:
    - log: results/flip_flop/adversary_r1/stationary/adversary_log.jsonl
      top_k: 10
    - log: results/flip_flop/adversary_r1/piecewise/adversary_log.jsonl
      top_k: 10
    - log: results/flip_flop/adversary_r1/planted_decoy/adversary_log.jsonl
      top_k: 10

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
  lambda_in: 5.0
  lambda_98: 0.7
  lambda_01: 0.7
  baseline_in_dist_glitch: 0.0
  baseline_98_glitch: 0.0599
  baseline_01_glitch: 0.0157
  in_dist_hard_cap: 0.005
  tail_98_hard_cap: 0.005
  tail_01_hard_cap: 0.005
  family_eval_n: 2000
  plateau_window: 5
  plateau_tol: 0.005
  memorize_check_evals: 2
  memorize_warmup_steps: 800

output:
  out_dir: results/flip_flop/retrain/round_2_redone
YAML

# === Step 2: retrain ===
echo "=== TIER1 R2-REDONE RETRAIN $(date -u) ===" | tee -a logs/PROGRESS
python3 -u -m flip_flop.scripts.run_retrain \
    --config flip_flop/configs/retrain_r2_redone.yaml \
    > logs/r2_redone_retrain.log 2>&1
RETRAIN_RC=$?
echo "  exit=${RETRAIN_RC} $(date -u)" | tee -a logs/PROGRESS
if [ $RETRAIN_RC -ne 0 ]; then
  echo "[tier1] retrain failed; aborting" | tee -a logs/PROGRESS
  exit $RETRAIN_RC
fi
touch logs/R2_REDONE_RETRAIN_DONE

# === Step 3: evaluate the retrained model on R1's saved top_k.jsonl files ===
# This is the load-bearing test: does R2-redone preserve R1's adversarial
# robustness? The Phase-B R2 model failed this test at 99.98% (piecewise),
# 38.6% (planted), 14.7% (stationary). With all 7 fixes, we expect numbers
# close to R1's saved final_eval values: ~10% piecewise, ~17% planted,
# ~10% stationary.
echo "=== TIER1 EVAL-ON-SAVED-TOPK $(date -u) ===" | tee -a logs/PROGRESS
python3 -u -m flip_flop.scripts.eval_on_saved_topk \
    --model results/flip_flop/retrain/round_2_redone/model_final.pt \
    --model_cfg flip_flop/configs/baseline.yaml \
    --lstm results/flip_flop/lstm/model_final.pt \
    --lstm_cfg flip_flop/configs/lstm.yaml \
    --topk results/flip_flop/adversary_r1/piecewise/top_k.jsonl:r1_piecewise \
    --topk results/flip_flop/adversary_r1/stationary/top_k.jsonl:r1_stationary \
    --topk results/flip_flop/adversary_r1/planted_decoy/top_k.jsonl:r1_planted \
    --n 5000 \
    --batch_size 64 \
    --seed 0 \
    --out results/flip_flop/retrain/round_2_redone/saved_topk_eval.json \
    > logs/r2_redone_topk_eval.log 2>&1
EVAL_RC=$?
echo "  exit=${EVAL_RC} $(date -u)" | tee -a logs/PROGRESS

# === Step 4: tar everything and mark FINAL_DONE ===
tar czf /tmp/tier1_results.tar.gz \
    logs/ \
    results/flip_flop/retrain/round_2_redone/ \
    flip_flop/configs/retrain_r2_redone.yaml \
    2>/dev/null

ls -la /tmp/tier1_results.tar.gz > logs/SAVE_COMPLETE
date -u >> logs/SAVE_COMPLETE
echo "=== TIER1 ALL DONE $(date -u) ===" | tee -a logs/PROGRESS
touch logs/FINAL_DONE
