# Separate verifier image for tunable_coupler_cz_netzero. Carries qiqcbench +
# hidden configs + the materialized scorer; the agent never runs here.
FROM qiqcbench-qsim:latest
COPY . /tests
RUN ln -sfn /app/qiqcbench_configs /app/configs
ENTRYPOINT []
HEALTHCHECK NONE
