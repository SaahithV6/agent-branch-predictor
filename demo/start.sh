#!/usr/bin/env bash
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
cd "$here/.."
for p in $(pgrep -f "^node bridge.mjs|^node --env-file-if-exists=.env src/index.ts|^node demo/server.mjs"); do kill "$p"; done
sleep 3
ENV_FILE=env.baseline BRIDGE_PORT=8788 QM_PORT=8081 timeout "${DEMO_HOURS:-6}h" nice -n 5 ./run.sh > "$here/plain.log" 2>&1 &
ENV_FILE=env.demo BRIDGE_PORT=8787 QM_PORT=8080 timeout "${DEMO_HOURS:-6}h" nice -n 5 ./run.sh > "$here/ours.log" 2>&1 &
until grep -q "connector registered" "$here/plain.log" 2>/dev/null && grep -q "connector registered" "$here/ours.log" 2>/dev/null; do sleep 2; done
warm() {
  curl -s -o /dev/null -m 120 "localhost:$1/v1/turns" -H 'content-type: application/json' \
    -d "{\"surface\":\"api\",\"actor\":{\"externalId\":\"saahith\"},\"conversation\":{\"kind\":\"dm\",\"threadRef\":\"warm-$1-$(date +%s%N)\"},\"text\":\"Is Zoom at risk of churning before renewal?\"}"
}
warm 8081 & warm 8080 & wait
timeout "${DEMO_HOURS:-6}h" node demo/server.mjs > "$here/server.log" 2>&1 &
sleep 1
echo "demo ready: http://127.0.0.1:8090"
