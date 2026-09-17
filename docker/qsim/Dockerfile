# qsim sidecar image. Hosts the qiqcbench qsim MCP server on :8123/mcp.
# Build:  docker build -t qiqcbench-qsim -f docker/qsim/Dockerfile .
#
# Bind-mounts at runtime:
#   /app/qiqcbench_configs       <- repo configs/ (read-only)
#   /qsim_logs                   <- host run dir's qsim_logs/ (read-write)
#   /app/hidden/hidden.yaml      <- per-run hidden truth (read-only)
#
# The agent sees only /qsim_logs/public_job_results, read-only; /app/hidden remains qsim-only.

FROM python:3.11-slim AS controller-python

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        bash \
        build-essential \
        curl \
        ca-certificates \
        git \
        && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# This install sits above `COPY src` on purpose. It is the
# expensive layer -- ~470 s cold on arm64, where both stim and the pinned qutip
# 5.2.3 predate their first linux-aarch64 wheels and compile from the sdist --
# and Harbor wraps build and `up --wait` together in each bundle's
# build_timeout_sec. With the source below it, a src edit rebuilds only the
# cheap layers (about 9 s) instead of the whole install (363 s measured).
#
# requirements.txt is uv.lock's exact `core + [live]` closure, regenerated with
# `uv export --frozen --no-emit-project --extra live -o
# docker/qsim/requirements.txt` whenever uv.lock moves;
# Every line is
# `==`-pinned with hashes, so `--require-hashes` fetches exact bytes rather than
# re-resolving against PyPI mid-build. The base image's own pip runs it: an
# unpinned `pip install --upgrade pip` would be the one floating PyPI fetch left
# in the layer this pinning exists to make repeatable.
COPY docker/qsim/requirements.txt /app/requirements.txt
RUN pip install --no-deps --require-hashes -r /app/requirements.txt

COPY pyproject.toml /app/pyproject.toml
COPY README.md /app/README.md
COPY src /app/src

# --no-deps: the closure above is complete, so this installs only qiqcbench
# itself (and would fail loudly rather than silently re-resolve if it were not).
# The import check confirms the `[live]` extra really landed.
RUN pip install --no-deps /app && \
    python -c "import qiskit_ibm_runtime"

# Package-free Python root for agent-authored syndrome-controller programs.
# The parent qsim/verifier process retains the simulator and hidden model;
# controller code is chrooted as uid/gid 65534 and sees only its read-only
# source plus writable /work and /tmp.  Copying /usr/local from a pristine
# Python stage prevents the installed qiqcbench package from crossing the
# controller boundary.
COPY --from=controller-python /usr/local /controller_runner_root/usr/local
RUN mkdir -p /controller_runner_root/usr /controller_runner_root/etc \
        /controller_runner_root/tmp /controller_runner_root/work && \
    # Chroot alone does not create a network namespace. Remove the native
    # networking/FFI extensions from the package-free interpreter so submitted
    # Python cannot open sockets or reconstruct raw syscalls through ctypes.
    find /controller_runner_root/usr/local/lib/python3.11/lib-dynload \
        -maxdepth 1 \( -name '_socket*.so' -o -name '_ssl*.so' -o -name '_ctypes*.so' \) \
        -delete && \
    cp -a /lib /controller_runner_root/lib && \
    cp -a /usr/lib /controller_runner_root/usr/lib && \
    ( [ ! -e /lib64 ] || cp -a /lib64 /controller_runner_root/lib64 ) && \
    ( [ ! -d /usr/lib64 ] || cp -a /usr/lib64 /controller_runner_root/usr/lib64 ) && \
    cp /etc/ld.so.cache /controller_runner_root/etc/ld.so.cache && \
    cp /etc/passwd /controller_runner_root/etc/passwd && \
    cp /etc/group /controller_runner_root/etc/group && \
    chmod 1777 /controller_runner_root/tmp /controller_runner_root/work

# configs/ is runtime data (also bind-mounted by the task compose); copy it AFTER
# the dependency install so adding/editing a config never busts the pip layer and
# forces a full reinstall on the next Harbor `compose build`.
COPY configs /app/qiqcbench_configs

# Separate verifiers receive operator-private device files through Harbor's
# artifact ferry at /app/configs/devices, while existing verifier scorers also
# read task materials below /app/configs. Keep that path as an alias of the
# complete baked config root. Harbor uploads through the symlink into the
# verifier's ephemeral filesystem; .dockerignore still excludes operator
# *.hidden.yaml files from the baked source tree.
RUN ln -s /app/qiqcbench_configs /app/configs

ENV QIQCBENCH_CONFIGS=/app/qiqcbench_configs \
    QSIM_PORT=8123 \
    QSIM_HOST=0.0.0.0 \
    QSIM_DEVICE_ID=single_transmon_v0

EXPOSE 8123

# Leave QIQCBENCH_HIDDEN_PATH unset so the entrypoint can prefer the conventional
# per-run bind mount and otherwise use the documented local-development example.
# Explicit runtime-provided instance paths remain authoritative.

COPY docker/qsim/entrypoint.sh /app/entrypoint.sh
COPY docker/qsim/healthcheck.sh /app/healthcheck.sh
RUN chmod +x /app/entrypoint.sh /app/healthcheck.sh

HEALTHCHECK --interval=5s --timeout=10s --retries=20 --start-period=20s \
    CMD /app/healthcheck.sh

ENTRYPOINT ["/app/entrypoint.sh"]

# Source provenance. Declared LAST so a changing revision rebuilds
# only this metadata layer: a LABEL adds no filesystem layer, so re-stamping
# leaves RootFS.Layers untouched and does not invalidate the verifier images
# built FROM this one. tools/build_qsim_images.py supplies the value (with a
# `-dirty` suffix for an edited tree); a plain `docker build` or Harbor's own
# `compose build` leaves it `unknown` rather than claiming a source revision.
ARG SOURCE_REVISION=unknown
LABEL org.opencontainers.image.source="https://github.com/GaugeForge/Quantum-Harbor" \
      org.opencontainers.image.revision="${SOURCE_REVISION}" \
      org.opencontainers.image.description="QIQCBench qsim sidecar"
