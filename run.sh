#!/usr/bin/env bash
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
cd "$here"
envfile="${ENV_FILE:-env.local}"
[ -f "$envfile" ] && set -a && . "./$envfile" && set +a
[ -f claude.token ] && export CLAUDE_CODE_OAUTH_TOKEN="$(cat claude.token)"
unset ANTHROPIC_API_KEY ANTHROPIC_AUTH_TOKEN
if [ "${HARNESS:-}" = claude ] && [ -z "${CLAUDE_CODE_OAUTH_TOKEN:-}" ]; then echo "missing ./claude.token (run: claude setup-token)"; exit 1; fi
export BRIDGE_PORT="${BRIDGE_PORT:-8787}"
qm_port="${QM_PORT:-8080}"

nice -n 5 node bridge.mjs &
bridge=$!
trap 'kill $bridge 2>/dev/null' EXIT
until curl -sf -o /dev/null -X POST "localhost:$BRIDGE_PORT/token"; do sleep 0.2; done

export HARNESS="${HARNESS:-mock}"
export ALLOW_UNAUTHENTICATED_CORE=1
export PORT="$qm_port"
export DATA_DIR="$here/qm-data-$qm_port"
export NATIVE_CLIENT_ID=local NATIVE_CLIENT_SECRET=local
export MEMORY_PROVIDER_CONFIG="{\"providers\":[{\"id\":\"native\",\"type\":\"mcp\",\"url\":\"http://127.0.0.1:$BRIDGE_PORT\",\"timeoutMs\":3000,\"read\":{\"tool\":\"qm_recall\",\"scopeArg\":\"scope\",\"maxCharsArg\":\"max_chars\",\"clientIdEnv\":\"NATIVE_CLIENT_ID\",\"clientSecretEnv\":\"NATIVE_CLIENT_SECRET\"},\"write\":{\"tool\":\"qm_remember\",\"scopeArg\":\"scope\",\"sourceArg\":\"source\",\"clientIdEnv\":\"NATIVE_CLIENT_ID\",\"clientSecretEnv\":\"NATIVE_CLIENT_SECRET\"}}],\"routes\":[{\"provider\":\"native\",\"scopes\":[\"personal\",\"channel\",\"group\",\"team\",\"org\"],\"capture\":\"automatic\",\"manage\":true,\"label\":\"Native\"}]}"

cd "${QM_DIR:-$HOME/qm}"
nice -n 5 npm start &
qm=$!
trap 'kill $bridge $qm 2>/dev/null' EXIT
until curl -s -o /dev/null "localhost:$qm_port/"; do sleep 0.5; done
curl -sf -X PUT "localhost:$qm_port/v1/admin/mcp-servers/native" \
  -H 'content-type: application/json' -H 'x-admin-actor: admin-alice@default-org' \
  -d "{\"url\":\"http://127.0.0.1:$BRIDGE_PORT\",\"auth\":\"none\",\"enabled\":true,\"readOnly\":false}" >/dev/null && echo "[qm-native] Native connector registered"
curl -s -o /dev/null "localhost:$BRIDGE_PORT/mcp" -d '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"qm_remember","arguments":{"content":"Onboarding: completed v2 on 2026-10-05.","scope":"personal:saahith","acting_user":"saahith","source":"explicit"}}}'
wait $qm
