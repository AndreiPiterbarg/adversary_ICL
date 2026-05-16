#!/usr/bin/env bash
# LOCAL monitor for the RunPod 5090 run. Backgrounded on the user's machine.
#   * polls pod STATUS every 3 min
#   * at 2h: ONE interim pull (does NOT kill the run, does NOT terminate)
#   * on DONE: final pull -> verify -> terminate pod (gated on a good pull)
#   * on FAILED: pull logs, do NOT terminate (leave pod for debugging)
#   * 6h safety cap: pull + exit WITHOUT terminating (never strand silently)
set -uo pipefail

RUN_TS="${1:?usage: orchestrate.sh RUN_TS}"
KEY="${HOME}/.ssh/id_ed25519"
HOST="root@149.36.1.38"
PORT="41861"
POD_ID="x59vbrbmeyeq4z"
RUN_DIR="/workspace/run_${RUN_TS}"
# git-bash POSIX path (NO drive-colon: scp treats 'C:' as a host otherwise)
LOCAL_OUT="/c/Users/andre/OneDrive - PennO365/Documents/adversary_ICL/results/pod_pull/run_${RUN_TS}"
SSHO=(-i "$KEY" -p "$PORT" -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -o ServerAliveInterval=30 -o ServerAliveCountMax=4)
SCPO=(-i "$KEY" -P "$PORT" -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20)

mkdir -p "$LOCAL_OUT"
MON="${LOCAL_OUT}/monitor.log"
say(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$MON"; }
ssh_pod(){ ssh "${SSHO[@]}" "$HOST" "$@"; }

pull(){  # $1 = tag; returns 0 only on a verified non-trivial pull
  local tag="$1" remote sz
  remote="$(ssh_pod "bash ${RUN_DIR}/_pod/pull_pack.sh" 2>>"$MON" | tail -1)" || { say "pull[$tag]: pack failed"; return 1; }
  [ -n "$remote" ] || { say "pull[$tag]: empty remote path"; return 1; }
  scp "${SCPO[@]}" "${HOST}:${remote}" "${LOCAL_OUT}/pull_${tag}.tgz" 2>>"$MON" || { say "pull[$tag]: scp failed"; return 1; }
  sz="$(stat -c%s "${LOCAL_OUT}/pull_${tag}.tgz" 2>/dev/null || echo 0)"
  say "pull[$tag]: ${LOCAL_OUT}/pull_${tag}.tgz (${sz} bytes)"
  [ "$sz" -gt 200 ] || { say "pull[$tag]: tarball too small — treating as FAILED"; return 1; }
  mkdir -p "${LOCAL_OUT}/${tag}" && tar xzf "${LOCAL_OUT}/pull_${tag}.tgz" -C "${LOCAL_OUT}/${tag}" --no-same-owner --no-same-permissions 2>>"$MON" || true
  return 0
}

START="$(date +%s)"; INTERIM_DONE=0; INTERIM_AT=$((2*3600)); CAP=$((6*3600))
say "monitor start RUN_TS=${RUN_TS} pod=${POD_ID} -> ${LOCAL_OUT}"

while :; do
  ST="$(ssh_pod "cat ${RUN_DIR}/_pod/STATUS 2>/dev/null" 2>/dev/null || echo UNREACHABLE)"
  NOW="$(date +%s)"; EL=$((NOW-START))
  say "status=${ST} elapsed=$((EL/60))m"

  if [ "$ST" = "DONE" ]; then
    say "run DONE -> final pull"
    if pull final; then
      say "================ READY TO TERMINATE ================"
      say "final pull VERIFIED. runpodctl is NOT authed on the pod, so"
      say "termination is MANUAL (user choice). Terminate pod now:"
      say "  RunPod dashboard -> Stop/Terminate, OR (authed machine):"
      say "  runpodctl remove pod ${POD_ID}"
      say "Pod LEFT UP intentionally — nothing is lost; it is billing."
      say "===================================================="
    else
      say "FINAL PULL FAILED -> pod left UP, pull manually before terminating."
    fi
    break
  fi

  if [ "$ST" = "FAILED" ]; then
    say "run FAILED -> pulling logs, NOT terminating (left up for debug)"
    pull failed || true
    break
  fi

  if [ "$INTERIM_DONE" -eq 0 ] && [ "$EL" -ge "$INTERIM_AT" ]; then
    say "2h reached -> INTERIM pull (NOT killing run, NOT terminating)"
    if pull interim2h; then INTERIM_DONE=1; else say "interim pull failed; retry next cycle"; fi
  fi

  if [ "$EL" -ge "$CAP" ]; then
    say "6h safety cap -> pull + EXIT WITHOUT terminate (pod left up for user)"
    pull cap6h || true
    break
  fi
  sleep 180
done
say "monitor exit"
