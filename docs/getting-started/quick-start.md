# Quick start

Run an installation check, launch one agent attempt, and inspect the result.

A trial is one agent attempt at one task. A job is a set of trials. Read [Harbor's quick start](https://docs.harborframework.com/getting-started/quick-start) for the general job workflow.

## Check the installation

From the repository root, run one task with Harbor's `nop` agent. The `nop` agent does nothing and needs no API key, so the run exercises the images, the sidecar, and the grader without any model calls.

```bash
harbor run -p harbor_tasks/time_budgeted_hamlearn_10q -a nop
```

The trial takes about two minutes and ends with a reward of `0`, because the agent submitted no answer.

To check all five bundles at once, point Harbor at the whole directory:

```bash
harbor run -p harbor_tasks -a nop
```

Four trials end with reward `0`. The 60-qubit surrogate trial ends without a reward file, and its grader exits with code `3`. That grader reports an infrastructure failure whenever the evidence log is empty, which is what a `nop` agent leaves behind. See [Grading](../tasks/grading.md).

## Run an agent

Set an API key in your shell and run one of the example configs:

```bash
export OPENAI_API_KEY=<your key>
harbor run -c configs/harbor/time_budgeted_hamlearn_10q_codex.yaml
```

The config selects a Codex agent and a model. To change the model, pass both `-a` and `-m`, as described in [Run a job](../jobs/run-a-job.md). A trial can run until the task's agent timeout, which is one hour for this task.

## View the results

```bash
harbor view jobs
```

Jobs are written under `jobs/` by default. [View job results](../results.md) lists where the reward, the score report, and the collected evidence land.

## Next steps

Read the [released tasks](../tasks/released-tasks.md), then [run a job](../jobs/run-a-job.md) for a selected task or [run the sidecar without Harbor](../qsim/sidecar.md).
