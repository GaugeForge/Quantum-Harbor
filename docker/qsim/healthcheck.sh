#!/usr/bin/env bash
# Probe the sidecar's non-MCP GET /health route. It answers only
# when the HTTP event loop is responsive, which is exactly what a wedged
# transport fails; unlike the previous `initialize` POST it does not mint a new
# MCP session per probe.
set -uo pipefail

PORT="${QSIM_PORT:-8123}"

if curl -sSf -m 5 "http://127.0.0.1:${PORT}/health" > /dev/null; then
    exit 0
fi
echo "healthcheck failed: GET /health on port ${PORT} did not answer" >&2
exit 1
