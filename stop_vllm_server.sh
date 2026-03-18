#!/usr/bin/env bash
set -euo pipefail

PID_FILE=".vllm_server.pid"

if [[ -f "${PID_FILE}" ]]; then
  SERVER_PID="$(cat "${PID_FILE}")"
  if kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill -TERM "${SERVER_PID}" 2>/dev/null || true
    for _ in $(seq 1 20); do
      if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        break
      fi
      sleep 1
    done
    if kill -0 "${SERVER_PID}" 2>/dev/null; then
      kill -KILL "${SERVER_PID}" 2>/dev/null || true
    fi
  fi
  rm -f "${PID_FILE}"
fi

pkill -f vllm_server.py || true
