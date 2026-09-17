# Installation

Install the tested Harbor version, synchronize the Python package, and build the images required by the released tasks.

You need Docker, uv, Python 3.11 or newer, and a git checkout. Install Harbor following its [installation instructions](https://docs.harborframework.com/getting-started/installation), including the Docker setup. This repository pins Harbor **0.18.0**. Harbor's online documentation follows the latest Harbor release. Where it disagrees with 0.18.0, trust `harbor run --help` from the pinned install.

```bash
uv tool install 'harbor==0.18.0'
git clone https://github.com/GaugeForge/Quantum-Harbor
cd Quantum-Harbor
uv sync
docker build -t qiqcbench-agent docker/agent/
uv run python tools/build_qsim_images.py
```

The first build downloads several gigabytes and takes about 20 minutes. Later builds reuse the Docker cache. The agent image contains Python scientific libraries and the agent CLIs. The image builder builds `qiqcbench-qsim:latest` and the verifier images declared in each [task bundle](../tasks/overview.md). It records the checkout's git revision in the image labels, with a `-dirty` suffix and a warning when the working tree has local edits. Harbor uses a declared separate-verifier image without building the task's `tests/Dockerfile` automatically, so build verifier images before running a job. Rebuild the images after changing code or verifier inputs.

Use `uv run python tools/build_qsim_images.py --tasks time_budgeted_hamlearn_10q` to build the qsim image and one task's verifier image. Use `--qsim-only` when only the qsim image is needed. Neither option builds the agent image.
