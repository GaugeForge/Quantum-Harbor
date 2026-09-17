# Separate verifier image for adaptive_clustered_clbcs_h2o.
#
# task.toml sets [verifier].environment_mode = "separate", so Harbor runs the
# agent in a different container and this image may carry hidden truth (the
# hidden scorer + the qsim hidden-dynamics state reconstruction). The agent
# never runs here.
#
# qiqcbench-qsim:latest bakes the qiqcbench package (importable) plus the full
# configs tree at /app/qiqcbench_configs (incl. the hidden scorer YAML and the
# public Hamiltonian). The symlink exposes them at /app/configs, which the
# scorer's resolvers probe. Harbor copies the agent's qsim evidence into
# /qsim_logs via [[artifacts]] (service = "qsim").
FROM qiqcbench-qsim:latest

COPY . /tests

# Harbor copies the agent's qsim evidence into /qsim_logs via [[artifacts]]; the
# base qsim image does not create that directory and this image resets the
# entrypoint (which would otherwise mkdir it at runtime), so create it here or
# Harbor's `docker cp .../experiment_log.jsonl main:/qsim_logs/...` fails closed.
RUN mkdir -p /qsim_logs \
    && ln -sfn /app/qiqcbench_configs /app/configs

# Keep the verifier container idle (no MCP server) until Harbor execs test.sh.
ENTRYPOINT []
HEALTHCHECK NONE
