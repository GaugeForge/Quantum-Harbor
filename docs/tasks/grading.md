# Grading

See how a grader scores a trial from the experiments qsim recorded.

Each `task.toml` requests `environment_mode = "separate"` and names a verifier Docker image. Build the [verifier images](../getting-started/installation.md) before a job. Harbor runs the grader outside the agent's `main` environment. The verifier image carries scorer code and any task-owned private reference material that the agent must not read.

Harbor collects declared artifacts from the `qsim` service, including `/qsim_logs/experiment_log.jsonl`, `/qsim_logs/final_answer.json`, and `/qsim_logs/public_job_results/`. Some bundles also collect task-specific materials. The scorer reads those records and checks the final answer against them. Depending on the task, the scorer recomputes a quantity from raw records, checks a prediction against hidden values, or replays a submitted design on fresh simulated measurements.

The task's `tests/test.sh` invokes its scorer and writes a binary `reward.txt` at `/logs/verifier/reward.txt`. A structured `score_report.json` is written under `/logs/verifier/artifacts/` when the scorer produces a report, along with copies of collected evidence. A scored trial writes reward `1` or `0`. A verifier or evidence-transfer infrastructure failure has **no reward file** and exits through a reserved nonzero code. Four tasks reserve exit code `2`. The 60-qubit surrogate task reserves exit codes of `3` and above. Do not interpret a missing reward as a failed scientific answer.

The 60-qubit surrogate grader expects qsim's startup record in the evidence log. The released bundle starts qsim without that record, so a run in which the agent never calls a tool leaves the log empty, and the grader reports an infrastructure failure instead of a reward of `0`. A `nop` run of that task therefore ends without a reward file.

See Harbor's [verifier](https://docs.harborframework.com/core-concepts/tasks/verifier), [separate verifier](https://docs.harborframework.com/core-concepts/tasks/separate-verifier), and [artifact collection](https://docs.harborframework.com/core-concepts/tasks/artifacts) pages for generic mechanics.
