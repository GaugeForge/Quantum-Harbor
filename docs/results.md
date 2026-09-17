# View job results

Find each trial's reward, score report, and collected evidence.

```bash
harbor view jobs
```

`harbor view jobs` serves Harbor's results viewer on the first free local port between 8080 and 8089 and runs until you stop it. Read [Harbor's view-job-results guide](https://docs.harborframework.com/core-concepts/results/view-job-results) for the viewer itself.

## Layout on disk

Harbor writes one directory per job under `jobs/` (the default, and the value the example configs set). Each trial gets a directory named after the task, truncated to 32 characters, plus a short random suffix:

```text
jobs/<job name>/
├── config.json  lock.json  result.json  job.log
└── time_budgeted_hamlearn_10q__QaWgDsJ/
    ├── config.json  lock.json  result.json  trial.log
    ├── exception.txt                    only when the trial raised an error
    ├── agent/                           the agent's logs
    ├── artifacts/
    │   ├── manifest.json                what Harbor tried to collect, with a status per path
    │   └── qsim_logs/
    │       ├── experiment_log.jsonl     the evidence log
    │       ├── final_answer.json        present when the agent submitted an answer
    │       └── public_job_results/
    └── verifier/
        ├── reward.txt                   1 or 0
        ├── test-stdout.txt
        └── artifacts/
            ├── score_report.json        the grader's structured report
            └── ...                      copies of the evidence the grader read
```

Harbor mirrors every collected sidecar path under `artifacts/`. For example, `/qsim_logs/final_answer.json` on the `qsim` service lands at `artifacts/qsim_logs/final_answer.json`. Some bundles collect task-specific files as well.

## Reading the outcome

- `verifier/reward.txt` holds the binary reward.
- `verifier/artifacts/score_report.json` records the evidence the grader found and the reason the answer passed or failed.
- A trial without `verifier/reward.txt` did not get a scientific score. Check `exception.txt` and `verifier/test-stdout.txt`. A missing reward with a reserved verifier exit code marks an infrastructure failure, not a wrong answer. See [Grading](tasks/grading.md).
