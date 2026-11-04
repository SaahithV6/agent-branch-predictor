#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
stop() {
  for p in $(pgrep -f "^node bridge.mjs|^node --env-file-if-exists=.env src/index.ts"); do kill "$p"; done
  sleep 4
}
start() {
  printf 'HARNESS=claude\n%s\nNATIVE_DB=%s/%s\nPREDICTOR=%s\n' "${BENCH_ENV:-SIM=1}" "$PWD" "${BENCH_DB:-bench.db}" "$1" > env.local
  nice -n 5 ./run.sh > "eval/run-$2.log" 2>&1 &
  until grep -q "connector registered" "eval/run-$2.log" 2>/dev/null; do sleep 2; done
  curl -s -o /dev/null localhost:8787/mcp -d '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"qm_remember","arguments":{"content":"Onboarding: completed v2 on 2026-10-05.","scope":"personal:saahith","acting_user":"saahith","source":"explicit"}}}'
}
cd eval
for phase in "$@"; do
  case $phase in
    warmup) mode=shadow ;;
    baseline) mode=off ;;
    speculate) mode=speculate ;;
    inject) mode=inject ;;
    shadow) mode=shadow ;;
  esac
  stop
  (cd .. && start "$mode" "$phase")
  nice -n 5 python3 "${BENCH_PY:-bench.py}" "$phase"
done
stop
printf 'HARNESS=claude\nPREDICTOR=speculate\n' > ../env.local
