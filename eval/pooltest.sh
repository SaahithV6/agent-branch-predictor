#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
stop() { for p in $(pgrep -f "^node bridge.mjs|^node --env-file-if-exists=.env src/index.ts"); do kill "$p"; done; sleep 4; }
n=0
for pool in 0 1 0 1; do
  n=$((n+1)); stop
  printf 'HARNESS=claude\nRESEARCH=1\nBIZ=1\nNATIVE_DB=%s/profile.db\nPREDICTOR=off\nCLAUDE_WARM_POOL=%s\n' "$PWD" "$pool" > env.local
  nice -n 5 ./run.sh > "eval/run-pool$n.log" 2>&1 &
  until grep -q "connector registered" "eval/run-pool$n.log" 2>/dev/null; do sleep 2; done
  curl -s -o /dev/null localhost:8787/mcp -d '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"qm_remember","arguments":{"content":"Onboarding: completed v2 on 2026-10-05.","scope":"personal:saahith","acting_user":"saahith","source":"explicit"}}}'
  (cd eval && PROFILE_TAG="pool$pool-block$n" BENCH_MODEL=claude-haiku-4-5 python3 profile.py)
done
stop
printf 'HARNESS=claude\nREAL=1\nPREDICTOR=inject\nCLAUDE_MODEL=claude-haiku-4-5\n' > env.local
