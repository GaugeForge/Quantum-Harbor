# Core concepts

Quantum-Harbor adds a laboratory interface, interchangeable hardware backends, and evidence-bound grading to Harbor.

Harbor supplies tasks, agents, trials, jobs, and verifiers. A trial is one agent attempt at a task, and a job gathers trials. Read [Harbor's core concepts](https://docs.harborframework.com/core-concepts/index) for the underlying workflow.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="images/fig_framework_dark.png">
  <img alt="Agent, qsim, and grader boundaries" src="images/fig_framework.png">
</picture>

## Agent environment

Harbor runs the agent in the `main` container, built from `qiqcbench-agent`. The agent can write analysis code and task outputs, but the five released tasks expose device control through MCP, not by importing the qsim implementation. Public task materials are mounted read-only where provided.

## Public surface

The MCP server at `http://qsim:8123/mcp` exposes public hardware specifications, a lab notebook that may contain stale calibration, experiment submission and result polling, and final-answer submission. Task-specific tools determine which experiments an agent can run. Some tasks also expose public files at `/task_materials`.

## Sidecar

The `qsim` container hosts the active qtype, its simulated hardware, hidden parameters, and experiment records. Each experiment tool call creates a qsim experiment job and returns its job ID, which the agent polls for raw measurements. A qsim experiment job is unrelated to a Harbor job, which is a set of trials. The agent container mounts neither the configs tree nor the qsim log volume.

## Qtype

A qtype is one hardware abstraction. It defines public and hidden device schemas, simulator behavior, and task-specific tools. A descriptor in `src/qiqcbench/qsim/qtypes/<name>/qtype.py` registers each qtype automatically. [Browse the qtypes](qsim/qtypes.md).

## Hidden parameters

Device configuration separates agent-visible specifications from hidden simulator parameters. The released task cards permit simulator mode only. The agent container does not receive the hidden configuration or private scorer.

## Lab notebook

`get_lab_notebook` returns a device's characterization notes. Calibration entries can be outdated, so an agent can measure before relying on them.

## Evidence log

Qsim records experiment activity in `/qsim_logs/experiment_log.jsonl` and stores large public job results under `/qsim_logs/public_job_results/`. Harbor collects these qsim-owned records for the verifier. The agent can read the public results but cannot edit the qsim evidence log.

## Final answer

The agent submits a task-shaped answer with `submit_final_answer`. Qsim writes the accepted submission to `/qsim_logs/final_answer.json`. Hamiltonian learning and the 60-qubit surrogate task also support a file-based submission through `/submission`.

## Grader and score report

The grader is the task's verifier in Harbor terms. Harbor runs the task's `tests/test.sh` in a separate verifier environment, and `test.sh` calls the task's Python scorer. The scorer inspects the submitted answer and the collected experimental evidence, then writes a binary reward and a `score_report.json` with grading details. [Read about grading](tasks/grading.md).

## Backend modes

The backend selector supports `simulator`, `provider_replay`, and `live_provider` code paths. All five released tasks allow `simulator` only. Live-provider mode is opt-in and implemented for IBM Quantum only, in the `digital_gate_model` and `transmon_pulse` qtypes. Replay mode exists in those two and in `trapped_ion_chain`. None of these three qtypes has a device file in this repository.

## Paper terms and repository names

| Paper term | Repository implementation |
| --- | --- |
| Agent environment | Harbor `main` service using `qiqcbench-agent` |
| Sidecar | `qsim` service using `qiqcbench-qsim` and `src/qiqcbench/qsim/` |
| Public surface | MCP endpoint `/mcp` and read-only public files under `/task_materials` |
| Qtype | Auto-discovered `src/qiqcbench/qsim/qtypes/<name>/qtype.py` descriptor |
| Hidden parameters | `configs/devices/<device>.hidden.example.yaml` and task-specific hidden materials |
| Evidence log | `/qsim_logs/experiment_log.jsonl`, public job results, and final answer |
| Grader | Harbor verifier executing `harbor_tasks/<task>/tests/test.sh` and the task scorer |
