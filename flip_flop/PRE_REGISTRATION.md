# Pre-registration — frozen BEFORE the stationary-only necessity control (Rung 1)

Committed before training `stationary_control` so nothing downstream can be
tuned to a desired outcome. Changing anything below after this commit
invalidates the pre-registration.

## Frozen evaluation design
- Eval set: `HELDOUT_PIECEWISE` + 16 `piecewise_band` jitters (seed 20260516),
  `liu_ffl_098`, `liu_ffl_010`, 10 Periodic (full-simplex, seed 42),
  `heldout_anticorr`. All T=512.
- N = 4096 sequences/distribution, eval seed = 42, same sequence tensor for
  every model (paired).
- Paired bootstrap = 4000 resamples, 95 % percentile CI on
  `pooled_err(B_liu) − pooled_err(model)`.
- LSTM-invalid threshold: drop a distribution if LSTM error_rate > 5e-4.

## Frozen decision thresholds
- BEAT (non-stationary band): bootstrap 95 % CI of `(B_liu − our_model)`
  excludes 0 in our favour.
- TIE / non-inferiority (Liu tails + Periodic): `|B_liu − our_model|` within
  δ = 2e-4.
- Control necessity (Rung 1): `stationary_control` band err > 10 × LSTM ⇒
  non-stationarity is NECESSARY; else honest downgrade.
- FALSIFY: B_liu band err ≤ 5 × LSTM ⇒ no beat (report negative).
- `stationary_control` family list frozen in
  `flip_flop/scripts/build_stationary_control.py` (2 Liu tails + fixed 8-point
  p_i grid {0.02,0.05,0.15,0.30,0.50,0.70,0.90,0.97}).

## Rung 0 outcome (recorded verbatim, pre-control)
Pooled over 17 LSTM-valid piecewise-band dists, N=4096:
- B_liu       = 4.060e-02   (LSTM = 0.000)
- full_adv    = 4.327e-04   → (B_liu−full_adv) 95 % CI [+3.97e-2, +4.07e-2] BEAT
- beat_R1     = 8.173e-06   → (B_liu−beat_R1)  95 % CI [+4.01e-2, +4.11e-2] BEAT
Liu tails / Periodic: B_liu = 0; full_adv / beat_R1 ≈ 1e-5 (within δ=2e-4 →
non-inferior). FALSIFY ruled out (B_liu band = 4.06e-2 ≫ 5×LSTM).

## Falsification sentence (verbatim)
"If, at N≥4096, B_liu's pooled error on the piecewise band is ≤ 5× the LSTM
skyline, there is no non-stationary beat and we report the negative result."
