# QIQCBench agent image. Carries:
#   - Python 3.11 + qiskit + qiskit-aer + qcodes  (agent-facing control
#     convenience; qsim engine remains pure-numpy)
#   - Node 22 + nvm + vendored codex CLI  (codex agent driver)
#   - Claude Code CLI (claude agent driver) so Harbor's install template
#     short-circuits on `command -v claude` instead of installing at runtime
#
# Build from the agent image directory:
#   docker build -t qiqcbench-agent docker/agent/

FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get -o Acquire::Retries=5 update && \
    apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        bash \
        procps \
        build-essential \
        cmake \
        ninja-build \
        git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
# build-essential/cmake/ninja/git: pymatching (+vendored stim) has no prebuilt
# linux/aarch64 wheel, so it compiles from source on arm64 hosts (Apple Silicon).
# Its CMake build FetchContent-clones googletest/stim, which needs git.

# Python: qiskit/qiskit-aer for circuit construction & qcodes for transmon control helpers.
# numpy/scipy are useful for fitting (parity oscillation, RB decay, etc.)
# Direct dependencies are pinned so image rebuilds retain the same agent tools.
# Transitive versions still float; treat a bump as a comparability break.
# pymatching: surface-code decoder, so agents can score their own decoder calibrations.
# stim: client-side circuit/DEM tooling for the surface_code_lattice_surgery track (the
# honest calibration campaign needs stim + pymatching agent-side; the hidden noise never
# ships — the science being tested is estimating it, not reimplementing a matcher).
# qualtran: agent-facing FTQC resource-analysis convenience (
# exact pin, beta library; qsim engine and verifiers remain pure-numpy/stdlib and
# score against construction-time-frozen references, never live qualtran). No
# graphviz apt package: bloq diagram drawing is not needed for benchmark work.
# mcp: agent-side MCP client scripting (large final answers are built and
# submitted programmatically through a Python MCP client; task instructions may
# promise the package, and runtime pip installs must not be load-bearing under
# network isolation). Same major as the qsim server's pin in pyproject.toml;
# 2.x has an incompatible client API.
RUN pip install --no-cache-dir \
        "qiskit==1.4.6" \
        "qiskit-aer==0.15.1" \
        "qcodes==0.58.0" \
        "httpx==0.28.1" \
        "numpy==2.4.6" \
        "scipy==1.17.1" \
        "pydantic==2.13.4" \
        "ruamel.yaml==0.19.1" \
        "pymatching==2.4.0" \
        "stim==1.16.0" \
        "qualtran==0.7.0" \
        "pyzx==0.10.5" \
        "mcp==1.29.0"

# nvm + node 22 (mirrors the previous node:22-slim base).
ENV NVM_DIR=/root/.nvm
RUN mkdir -p "$NVM_DIR" && \
    curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.2/install.sh | bash && \
    . "$NVM_DIR/nvm.sh" && \
    nvm install 22 && \
    nvm alias default 22

# Symlink node/npm/npx to /usr/local/bin so harbor's runtime PATH (no nvm
# sourcing) finds them.
RUN . "$NVM_DIR/nvm.sh" && \
    NODE_DIR="$NVM_DIR/versions/node/$(nvm version)" && \
    ln -sf "$NODE_DIR/bin/node" /usr/local/bin/node && \
    ln -sf "$NODE_DIR/bin/npm"  /usr/local/bin/npm && \
    ln -sf "$NODE_DIR/bin/npx"  /usr/local/bin/npx

# codex CLI from npm, pinned to a gpt-5.6-sol-capable version, then symlink the wrapper
# to /usr/local/bin so harbor finds `codex` on PATH. gpt-5.6-sol requires 0.144.4.
# Newer codex ignores the OPENAI_BASE_URL env var but honors `openai_base_url` from
# $CODEX_HOME/config.toml, which harbor writes (agents/installed/codex.py) — so the
# This supports an OpenAI-compatible gateway configured by the agent driver.
ARG CODEX_VERSION=0.144.4
RUN . "$NVM_DIR/nvm.sh" && \
    npm install -g "@openai/codex@${CODEX_VERSION}" && \
    NODE_DIR="$NVM_DIR/versions/node/$(nvm version)" && \
    ln -sf "$NODE_DIR/bin/codex" /usr/local/bin/codex && \
    /usr/local/bin/codex --version

# Claude Code CLI via the official installer (native binary -> ~/.local/bin/claude).
# Pinned to match the host driver for reproducible benchmark runs. Harbor's
# install-claude-code.sh.j2 does `command -v claude` and skips install when present,
# so pre-vendoring avoids a per-run network install. Symlink to /usr/local/bin so the
# harness finds `claude` even before it prepends ~/.local/bin to PATH.
ARG CLAUDE_CODE_VERSION=2.1.186
RUN curl -fsSL https://claude.ai/install.sh | bash -s -- "$CLAUDE_CODE_VERSION" && \
    ln -sf /root/.local/bin/claude /usr/local/bin/claude && \
    /usr/local/bin/claude --version

WORKDIR /app
