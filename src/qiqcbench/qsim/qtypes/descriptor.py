"""Qtype descriptor type + lazy-loading helpers.

A ``QtypeDescriptor`` is the single object every qtype-agnostic façade
(``runner``, ``devices``, ``backends.factory``, ``mcp.server``) reads to learn
what a qtype can do. It deliberately separates two concerns:

* **Lightweight schema** — ``public_model`` / ``hidden_model`` /
  ``notebook_model`` / ``result_data_models`` plus the capability/mode/surface
  frozensets. These are safe to import at discovery time and are what the
  dynamic device / lab-notebook / capability / ``JobData`` unions are built
  from.
* **Heavyweight runtime** — backend factories, the MCP registrar, runner
  callables, and the instructions string. These pull in engines/backends and
  must NOT be imported at discovery time, or ``core.wire`` (which backends
  import) and the registry would form an import cycle. Pass them as
  :func:`lazy` callables / :func:`lazy_instructions` so the real module is
  imported only on first call.

The contributor contract this enables: a new
qtype is a self-contained ``qtypes/<name>/`` package exposing ``DESCRIPTOR``;
no central file is edited.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any

__all__ = [
    "QtypeDescriptor",
    "lazy",
    "lazy_instructions",
    "runner_not_applicable",
]


class _LazyCallable:
    """A callable that imports ``module`` and resolves ``attr`` on first call.

    Truthy and callable, so it is a drop-in for a real function reference in a
    descriptor slot (e.g. ``desc.build_replay_backend is None`` stays correct
    for qtypes that pass ``None``, while qtypes that pass a ``_LazyCallable``
    read as present).
    """

    __slots__ = ("_module", "_attr", "_resolved")

    def __init__(self, module: str, attr: str) -> None:
        self._module = module
        self._attr = attr
        self._resolved: Callable[..., Any] | None = None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self._resolved is None:
            self._resolved = getattr(importlib.import_module(self._module), self._attr)
        return self._resolved(*args, **kwargs)

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"<lazy {self._module}.{self._attr}>"


class _LazyInstructions:
    """Lazily resolve a module-level string constant (e.g. ``INSTRUCTIONS``)."""

    __slots__ = ("_module", "_attr", "_resolved")

    def __init__(self, module: str, attr: str) -> None:
        self._module = module
        self._attr = attr
        self._resolved: str | None = None

    def get(self) -> str:
        if self._resolved is None:
            value = getattr(importlib.import_module(self._module), self._attr)
            if not isinstance(value, str):
                raise TypeError(
                    f"{self._module}.{self._attr} must be a str, got {type(value).__name__}"
                )
            self._resolved = value
        return self._resolved


def lazy(module: str, attr: str) -> _LazyCallable:
    """Build a lazily-resolved callable for a descriptor runtime slot."""
    return _LazyCallable(module, attr)


def lazy_instructions(module: str, attr: str) -> _LazyInstructions:
    """Build a lazily-resolved instructions string for a descriptor."""
    return _LazyInstructions(module, attr)


def runner_not_applicable(method: str, qtype: str) -> Callable[..., Any]:
    """Build a placeholder runner that fails clearly when called on the wrong qtype."""

    def _raise(*_args: Any, **_kwargs: Any) -> Any:
        raise NotImplementedError(
            f"qtype {qtype!r} does not implement {method!r}; "
            "dispatch through descriptor_for_qtype(hidden.qtype) and call "
            "the runner method that matches the request shape."
        )

    return _raise


class QtypeDescriptor:
    """Per-qtype runtime capabilities resolved by ``descriptor_for_qtype``.

    Runner slots left unset default to a fail-closed
    :func:`runner_not_applicable` for this qtype, so a descriptor only declares
    the runners it actually implements.
    """

    __slots__ = (
        "qtype",
        "public_model",
        "hidden_model",
        "notebook_model",
        "result_data_models",
        "build_lab_notebook",
        "build_simulator_backend",
        "register_mcp_tools",
        "_instructions",
        "supported_backend_modes",
        "supported_surfaces",
        "supported_capabilities",
        "build_replay_backend",
        "build_live_provider_backend",
        "run_pulse_sequence",
        "run_sweep",
        "run_circuit_sequence",
        "run_circuit_sweep_request",
    )

    def __init__(
        self,
        *,
        qtype: str,
        public_model: type,
        hidden_model: type,
        notebook_model: type,
        build_lab_notebook: Callable[[Any], Any],
        build_simulator_backend: Callable[..., Any],
        register_mcp_tools: Callable[..., None],
        instructions: str | _LazyInstructions,
        supported_backend_modes: frozenset[str],
        supported_surfaces: frozenset[str],
        supported_capabilities: frozenset[str],
        result_data_models: tuple[type, ...] = (),
        build_replay_backend: Callable[..., Any] | None = None,
        build_live_provider_backend: Callable[..., Any] | None = None,
        run_pulse_sequence: Callable[..., Any] | None = None,
        run_sweep: Callable[..., Any] | None = None,
        run_circuit_sequence: Callable[..., Any] | None = None,
        run_circuit_sweep_request: Callable[..., Any] | None = None,
    ) -> None:
        self.qtype = qtype
        self.public_model = public_model
        self.hidden_model = hidden_model
        self.notebook_model = notebook_model
        self.result_data_models = result_data_models
        self.build_lab_notebook = build_lab_notebook
        self.build_simulator_backend = build_simulator_backend
        self.register_mcp_tools = register_mcp_tools
        self._instructions = instructions
        self.supported_backend_modes = supported_backend_modes
        self.supported_surfaces = supported_surfaces
        self.supported_capabilities = supported_capabilities
        self.build_replay_backend = build_replay_backend
        self.build_live_provider_backend = build_live_provider_backend
        self.run_pulse_sequence = run_pulse_sequence or runner_not_applicable(
            "run_pulse_sequence", qtype
        )
        self.run_sweep = run_sweep or runner_not_applicable("run_sweep", qtype)
        self.run_circuit_sequence = run_circuit_sequence or runner_not_applicable(
            "run_circuit_sequence", qtype
        )
        self.run_circuit_sweep_request = run_circuit_sweep_request or runner_not_applicable(
            "run_circuit_sweep_request", qtype
        )

    @property
    def instructions(self) -> str:
        """The MCP instructions string (resolved lazily for lazy descriptors)."""
        if isinstance(self._instructions, _LazyInstructions):
            return self._instructions.get()
        return self._instructions

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"QtypeDescriptor(qtype={self.qtype!r})"
