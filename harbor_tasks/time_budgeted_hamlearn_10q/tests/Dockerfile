# This context is materialized from verifier-context.json. Nothing else in the
# repository, including unrelated hidden material, is available to COPY.
FROM python:3.11.14-slim-bookworm@sha256:65a93d69fa75478d554f4ad27c85c1e69fa184956261b4301ebaf6dbb0a3543d

COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir --require-hashes --only-binary=:all: \
    -r /tmp/requirements.txt

ARG SOURCE_REVISION=unknown
ARG CONTENT_DIGEST=unknown
LABEL qiqcbench.verifier.task="time_budgeted_hamlearn_10q" \
      qiqcbench.verifier.source_revision="${SOURCE_REVISION}" \
      qiqcbench.verifier.content_digest="${CONTENT_DIGEST}"

RUN useradd --system --uid 10001 --no-create-home verifier && \
    mkdir -p /tests /verifier/private /qsim_logs /task_materials /logs/verifier && \
    chown -R verifier:verifier /tests /verifier /qsim_logs /task_materials /logs

COPY --chown=verifier:verifier --chmod=0755 test.sh /tests/test.sh
COPY --chown=verifier:verifier --chmod=0644 score_hamlearn.py /tests/score_hamlearn.py
COPY --chown=verifier:verifier --chmod=0644 hamlearn_contract.py /tests/hamlearn_contract.py
COPY --chown=verifier:verifier --chmod=0400 private/hidden_scorer.yaml /verifier/private/hidden_scorer.yaml

USER verifier

ENTRYPOINT []
HEALTHCHECK NONE
