# Separate verifier image for logical_cnot_decoder_calibration.
#
# task.toml sets [verifier].environment_mode = "separate", so the verifier runs in this
# image while the agent runs elsewhere. It may therefore carry the hidden noise model
# and the authoritative replay scorer — the agent never runs here.
#
# qiqcbench-qsim:latest bakes the qiqcbench package (importable, incl. stim) plus the
# full configs tree at /app/qiqcbench_configs. The symlink exposes them at /app/configs
# for the loaders. pymatching (the replay decoder) is a core dependency; it is
# installed here explicitly. Harbor copies the agent's qsim evidence into /qsim_logs via
# [[artifacts]] (service = "qsim").
FROM qiqcbench-qsim:latest

# pymatching has no prebuilt linux/aarch64 wheel: it compiles from source on arm64 hosts
# (its CMake build FetchContent-clones googletest/stim, which needs git).
RUN apt-get update && apt-get install -y --no-install-recommends cmake ninja-build git \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir "pymatching>=2.2"

# Bundle the verifier scripts into the image; Harbor execs /tests/test.sh.
COPY . /tests

RUN ln -sfn /app/qiqcbench_configs /app/configs

# Reset the qsim base entrypoint/healthcheck so the verifier container stays idle (no
# MCP server) until Harbor execs the verifier.
ENTRYPOINT []
HEALTHCHECK NONE
