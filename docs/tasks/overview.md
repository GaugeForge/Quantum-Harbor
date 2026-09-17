# Task overview

Read one released task bundle and see which of its files the agent can reach.

The Hamiltonian-learning bundle contains these files:

```text
harbor_tasks/time_budgeted_hamlearn_10q/
├── task.toml                  timeouts, MCP server, collected artifacts, verifier image
├── instruction.md             the task statement the agent receives
├── build_instance.py          maintainer script that regenerates the task materials
├── environment/
│   ├── docker-compose.yaml    the main (agent) and qsim (sidecar) services and their mounts
│   └── Dockerfile             the agent image for this task, built on qiqcbench-agent
└── tests/
    ├── Dockerfile             the standalone verifier image
    ├── verifier-context.json  allowlist of files copied into the verifier image
    ├── requirements.txt       hash-pinned verifier dependencies
    ├── test.sh                verifier entry point
    └── score_hamlearn.py      scorer
```

[`task.toml`](../../harbor_tasks/time_budgeted_hamlearn_10q/task.toml) declares timeouts, MCP access, artifact collection, and the separate verifier image. [`instruction.md`](../../harbor_tasks/time_budgeted_hamlearn_10q/instruction.md) gives the agent its goal, tools, submission contract, and public budget. [`environment/docker-compose.yaml`](../../harbor_tasks/time_budgeted_hamlearn_10q/environment/docker-compose.yaml) starts `main` and `qsim` and mounts the public materials. [`tests/test.sh`](../../harbor_tasks/time_budgeted_hamlearn_10q/tests/test.sh) runs the scorer after Harbor collects the evidence. The other four bundles also contain `task.toml`, `instruction.md`, and `environment/`, and the surrogate and composite-measurement bundles also contain `build_instance.py`. Their `tests/` directories hold only a Dockerfile, `test.sh`, and a `score_*.py` entry script. Those verifier images build on `qiqcbench-qsim`, and the entry scripts import the task's scorer from `src/qiqcbench/qsim/hidden_dynamics/<task>/`.

Public hardware specifications live in [`configs/devices/`](../../configs/devices/) as `<device>.public.yaml`. The hidden parameters that the released bundles run on live beside them in `configs/devices/<device>.hidden.example.yaml`, and only qsim and the separate verifier load them. [`configs/tasks/`](../../configs/tasks/) binds a task to a device, backend mode, capabilities, and public surface. Where a task has files, [`configs/task_materials/<task>/public/`](../../configs/task_materials/) is mounted for the agent. Hidden task materials under `configs/task_materials/<task>/hidden/` remain on the qsim or verifier side. The private scorer runs in the separate verifier image, not in the agent environment. Each bundle controls its own mounts, so read its `environment/docker-compose.yaml` before assuming a particular public file exists.

There is no `solution/` directory in the five bundles, so Harbor's `oracle` agent cannot run them as supplied.

For generic bundle mechanics, see Harbor's [task overview](https://docs.harborframework.com/core-concepts/tasks/overview), [task configuration](https://docs.harborframework.com/core-concepts/tasks/configuration), [multi-container environments](https://docs.harborframework.com/core-concepts/tasks/multi-container), [artifacts](https://docs.harborframework.com/core-concepts/tasks/artifacts), and [separate verifier](https://docs.harborframework.com/core-concepts/tasks/separate-verifier).
