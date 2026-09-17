# Qsim sidecar

Run qsim without Harbor and see the tools and settings it exposes.

From a synchronized checkout, list devices, inspect one public specification, then start a local sidecar for a released task:

```bash
uv run qiqcbench qsim list-devices
uv run qiqcbench qsim spec hamlearn_10q_chain_v1
uv run qiqcbench qsim serve-mcp --task-id time_budgeted_hamlearn_10q --port 8123 --log-dir jobs/qsim_local
```

The last command is a foreground server. Stop it when finished. Without `--task-id` or `--device-id`, the sidecar falls back to device `single_transmon_v0`, which this repository does not ship, and fails at startup. The streamable HTTP MCP endpoint is `http://localhost:8123/mcp`. `/health` and `/healthz` serve health responses. A local `--log-dir` receives `experiment_log.jsonl` and, by default in simulator mode, a hidden-truth snapshot. Keep that directory out of version control. The `jobs/` directory is already ignored by git. All five Harbor compose bundles disable the hidden-truth snapshot. The agent's public result mount contains only polled job results, not the full qsim log volume.

The always-on MCP tools are `list_devices`, `get_device_spec`, `get_lab_notebook`, `get_job_result`, and `submit_final_answer`. Each qtype registers task-specific measurement tools. Task tools usually return a job ID for an asynchronous experiment. Poll `get_job_result` for raw outcomes. Results too large for an MCP reply are written under `/qsim_logs/public_job_results/`. The task instruction gives the answer schema and public budget.

The sidecar reads `QIQCBENCH_CONFIGS` for the configs root, `QIQCBENCH_TASK_ID` or `QSIM_DEVICE_ID` for selection, and `QIQCBENCH_HIDDEN_PATH` for a hidden device config override. `QSIM_HOST` and `QSIM_PORT` set the server bind address and port in the container entrypoint. `QSIM_SNAPSHOT_HIDDEN_TRUTH` controls the local simulator snapshot. `QIQCBENCH_SURFACES` overrides an allowed agent surface. `QSIM_SUBMISSIONS_DIR` enables file-based final answers for tasks that mount a submission directory.

`QSIM_BACKEND_MODE` selects `simulator`, `provider_replay`, or `live_provider`, subject to the task card and qtype. Replay also uses `QIQCBENCH_REPLAY_ROOT`. Live mode requires `QIQCBENCH_ALLOW_LIVE_PROVIDER=1` and the supported IBM Quantum credentials and limits, including `QISKIT_IBM_TOKEN`. IBM Quantum is the only implemented live provider. Live mode is opt-in, and **all five released tasks accept simulator mode only**. The presence of replay and live code paths does not imply that the released tasks can use them.
