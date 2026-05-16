# pod/ — RunPod 5090 run orchestration

Runs Phase 1 of `docs/PLAN_expressive_adversary.md` (feature-conditioned
adversary search vs **B_liu**, LSTM skyline) on the RunPod 5090, with robust
logging, timed data collection, and gated auto-terminate.

| file | where it runs | what it does |
|---|---|---|
| `remote_run.sh` | the pod (detached) | deps → Phase-1 search → optional reduced head-to-head → STATUS/MANIFEST/artifacts |
| `orchestrate.sh` | local (backgrounded) | poll STATUS; 2h interim pull (no kill); on DONE final pull → terminate; on FAIL pull only |

**Pod:** `root@149.36.1.38 -p 41861` (direct TCP, SCP-capable), key
`~/.ssh/id_ed25519`, pod id `x59vbrbmeyeq4z`.

**Data collection.** Everything lands under `/workspace/run_<TS>/_pod/`:
`STATUS`, `run.log`, `search.log`, `eval.log`, `pip.log`, `metadata.txt`,
`MANIFEST.txt`, and `artifacts/` (copies of the adversary out dir + head-to-head).
The append-only `adversary_log.jsonl` + `cma_trajectory.jsonl` are flushed every
generation, so a pull at ANY time (incl. the 2h interim) is valid evidence even
mid-run. Pulled tarballs land in
`results/pod_pull/run_<TS>/{pull_*.tgz, <tag>/...}` locally.

**Terminate policy.** The pod is terminated **only** after a *verified*
final pull (tarball present, non-trivial size, extracted). A failed run or a
failed final pull leaves the pod UP so nothing is lost. The 2h timer only
triggers an interim snapshot — it never kills the run and never terminates
(matches the explicit instruction "let it keep running, don't kill at 2h").
