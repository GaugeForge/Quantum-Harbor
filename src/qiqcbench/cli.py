from __future__ import annotations

import typer

from qiqcbench.qsim.cli import qsim_app

app = typer.Typer(no_args_is_help=True, help="QIQCBench developer CLI.")
app.add_typer(qsim_app, name="qsim", help="Quantum simulator commands.")


if __name__ == "__main__":
    app()
