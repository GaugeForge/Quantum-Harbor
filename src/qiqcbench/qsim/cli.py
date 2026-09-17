from __future__ import annotations

import typer

qsim_app = typer.Typer(no_args_is_help=True, help="qsim commands.")


@qsim_app.command("list-devices")
def list_devices() -> None:
    """List devices known to the local config."""
    from qiqcbench.qsim.devices import list_known_devices

    for device in list_known_devices():
        typer.echo(device)


@qsim_app.command("spec")
def spec(device_id: str) -> None:
    """Print the public spec for a device."""
    import json

    from qiqcbench.qsim.devices import load_public_spec

    typer.echo(json.dumps(load_public_spec(device_id).model_dump(), indent=2))


@qsim_app.command("serve-mcp")
def serve_mcp(
    host: str = typer.Option("0.0.0.0", help="Bind host."),
    port: int = typer.Option(8123, help="Bind port."),
    task_id: str | None = typer.Option(None, help="Task to expose via task-spec bootstrap."),
    device_id: str | None = typer.Option(None, help="Device override for local dev fallback."),
    hidden_path: str | None = typer.Option(
        None,
        help="Path to hidden config YAML. Defaults to configs/devices/<id>.hidden.example.yaml.",
    ),
    log_dir: str | None = typer.Option(
        None, help="Directory for experiment_log.jsonl + hidden_truth_snapshot.yaml."
    ),
) -> None:
    """Backward-compatible MCP command serving the full qsim sidecar."""
    from qiqcbench.qsim.server import serve_sidecar

    serve_sidecar(
        host=host,
        port=port,
        task_id=task_id,
        device_id=device_id,
        hidden_path=hidden_path,
        log_dir=log_dir,
    )


@qsim_app.command("serve-sidecar")
def serve_sidecar_command(
    host: str = typer.Option("0.0.0.0", help="Bind host."),
    port: int = typer.Option(8123, help="Bind port."),
    task_id: str | None = typer.Option(None, help="Task to expose via task-spec bootstrap."),
    device_id: str | None = typer.Option(None, help="Device override for local dev fallback."),
    hidden_path: str | None = typer.Option(
        None,
        help="Path to hidden config YAML. Defaults to configs/devices/<id>.hidden.example.yaml.",
    ),
    log_dir: str | None = typer.Option(
        None, help="Directory for experiment_log.jsonl + hidden_truth_snapshot.yaml."
    ),
) -> None:
    """Start the qsim multi-surface sidecar."""
    from qiqcbench.qsim.server import serve_sidecar

    serve_sidecar(
        host=host,
        port=port,
        task_id=task_id,
        device_id=device_id,
        hidden_path=hidden_path,
        log_dir=log_dir,
    )
