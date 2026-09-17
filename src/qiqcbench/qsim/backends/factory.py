"""Backend runtime selector for simulator, replay, and live-provider modes."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from qiqcbench.qsim.agent_surfaces.manifest import build_surface_manifest
from qiqcbench.qsim.agent_surfaces.policy import resolve_active_surfaces
from qiqcbench.qsim.backends._live_constants import IBM_QUANTUM_PLATFORM_CHANNEL
from qiqcbench.qsim.backends.base import CircuitBackend, PulseBackend
from qiqcbench.qsim.capabilities import resolve_enabled_capabilities
from qiqcbench.qsim.core.device import HiddenDeviceConfig, PublicDeviceSpec
from qiqcbench.qsim.qtypes.registry import descriptor_for_qtype
from qiqcbench.qsim.tasks import AgentSurface, TaskSpec

BackendMode = Literal["simulator", "provider_replay", "live_provider"]
_SUPPORTED_BACKEND_MODES: set[str] = {"simulator", "provider_replay", "live_provider"}


@dataclass(frozen=True)
class BackendRuntime:
    """Resolved backend plus sidecar runtime metadata for qsim state."""

    backend: PulseBackend | CircuitBackend
    backend_mode: BackendMode
    task_id: str | None
    replay_root: Path | None
    log_dir: Path | None
    snapshot_hidden_truth: bool
    active_surfaces: tuple[str, ...]
    active_capabilities: tuple[str, ...]
    surface_manifest: dict[str, object]


def resolve_backend_mode(env: Mapping[str, str] | None = None) -> BackendMode:
    """Resolve and validate the qsim backend mode from environment."""

    env_map = os.environ if env is None else env
    mode = env_map.get("QSIM_BACKEND_MODE") or "simulator"
    if mode not in _SUPPORTED_BACKEND_MODES:
        supported = ", ".join(sorted(_SUPPORTED_BACKEND_MODES))
        raise ValueError(f"Unsupported QSIM_BACKEND_MODE {mode!r}; expected one of: {supported}")
    return cast(BackendMode, mode)


def _bool_env(value: str | None, *, default: bool) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Expected boolean value, got {value!r}")


def _runtime_path(value: str | Path | None) -> Path | None:
    return None if value is None else Path(value)


def _int_env(value: str | None, *, default: int | None = None) -> int | None:
    if value is None or value == "":
        return default
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"Expected positive integer value, got {value!r}")
    return parsed


def _float_env(value: str | None, *, default: float | None = None) -> float | None:
    if value is None or value == "":
        return default
    parsed = float(value)
    if parsed <= 0:
        raise ValueError(f"Expected positive numeric value, got {value!r}")
    return parsed


def _csv_env(value: str | None) -> list[str]:
    if value is None:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def build_runtime(
    hidden: HiddenDeviceConfig,
    public: PublicDeviceSpec,
    log_dir: str | Path | None = None,
    *,
    backend_mode: BackendMode | None = None,
    task_id: str | None = None,
    task_spec: TaskSpec | None = None,
    replay_root: str | Path | None = None,
    snapshot_hidden_truth: bool | None = None,
    env: Mapping[str, str] | None = None,
) -> BackendRuntime:
    """Build the selected qsim backend runtime.

    The selector keeps simulator mode intact while wiring replay and opt-in
    live-provider backends behind explicit runtime modes.
    """

    env_map = os.environ if env is None else env
    if backend_mode is not None:
        resolved_mode = backend_mode
    elif env_map.get("QSIM_BACKEND_MODE"):
        resolved_mode = resolve_backend_mode(env_map)
    elif task_spec is not None and task_spec.default_backend_mode is not None:
        resolved_mode = task_spec.default_backend_mode
    else:
        resolved_mode = resolve_backend_mode(env_map)
    if resolved_mode not in _SUPPORTED_BACKEND_MODES:
        supported = ", ".join(sorted(_SUPPORTED_BACKEND_MODES))
        raise ValueError(
            f"Unsupported backend_mode {resolved_mode!r}; expected one of: {supported}"
        )

    resolved_task_id = (
        task_id
        if task_id is not None
        else task_spec.task_id
        if task_spec is not None
        else env_map.get("QIQCBENCH_TASK_ID")
    )
    resolved_replay_root = _runtime_path(
        replay_root if replay_root is not None else env_map.get("QIQCBENCH_REPLAY_ROOT")
    )
    resolved_log_dir = _runtime_path(log_dir)

    if task_spec is not None and task_spec.device_id != public.device_id:
        raise ValueError(
            f"Task {task_spec.task_id!r} targets device {task_spec.device_id!r}, "
            f"not {public.device_id!r}"
        )
    if task_spec is not None and resolved_mode not in task_spec.allowed_backend_modes:
        raise ValueError(
            f"Task {task_spec.task_id!r} does not allow backend mode {resolved_mode!r}"
        )

    desc = descriptor_for_qtype(public.qtype)
    if resolved_mode not in desc.supported_backend_modes:
        raise ValueError(f"Qtype {public.qtype!r} does not support backend mode {resolved_mode!r}")

    enabled_capabilities = (
        task_spec.enabled_capabilities if task_spec is not None else ("basic_measurement",)
    )
    active_capabilities = resolve_enabled_capabilities(
        enabled_capabilities,
        supported_capabilities=set(desc.supported_capabilities),
    )

    default_surfaces = task_spec.default_surfaces if task_spec is not None else ("mcp",)
    allowed_surfaces = task_spec.allowed_surfaces if task_spec is not None else ("mcp",)
    surface_task_spec = task_spec or TaskSpec(
        task_id=resolved_task_id or "device_fallback",
        device_id=public.device_id,
        description="Device fallback runtime surface policy",
        wall_clock_budget_s=0,
        default_backend_mode=None,
        allowed_backend_modes=tuple(),
        default_surfaces=cast(tuple[AgentSurface, ...], default_surfaces),
        allowed_surfaces=cast(tuple[AgentSurface, ...], allowed_surfaces),
    )
    override = _csv_env(env_map.get("QIQCBENCH_SURFACES")) or None
    active_surfaces = resolve_active_surfaces(
        surface_task_spec,
        override=override,
        supported_surfaces=set(desc.supported_surfaces),
    )
    surface_manifest = build_surface_manifest(
        task_id=resolved_task_id,
        qtype=public.qtype,
        backend_mode=resolved_mode,
        default_surfaces=default_surfaces,
        allowed_surfaces=allowed_surfaces,
        active_surfaces=active_surfaces,
        override_source="QIQCBENCH_SURFACES" if override is not None else None,
    )

    if snapshot_hidden_truth is None:
        snapshot_hidden_truth = _bool_env(
            env_map.get("QSIM_SNAPSHOT_HIDDEN_TRUTH"),
            default=resolved_mode == "simulator",
        )

    if resolved_mode == "simulator":
        return BackendRuntime(
            backend=desc.build_simulator_backend(hidden, public),
            backend_mode="simulator",
            task_id=resolved_task_id,
            replay_root=resolved_replay_root,
            log_dir=resolved_log_dir,
            snapshot_hidden_truth=snapshot_hidden_truth,
            active_surfaces=active_surfaces,
            active_capabilities=active_capabilities,
            surface_manifest=surface_manifest,
        )

    if resolved_mode == "provider_replay":
        if resolved_task_id is None:
            raise ValueError("QIQCBENCH_TASK_ID is required for provider_replay mode")
        if resolved_replay_root is None:
            raise ValueError("QIQCBENCH_REPLAY_ROOT is required for provider_replay mode")
        if desc.build_replay_backend is None:
            raise ValueError(
                f"Qtype {public.qtype!r} does not support backend mode 'provider_replay'"
            )
        return BackendRuntime(
            backend=desc.build_replay_backend(
                task_id=resolved_task_id,
                public=public,
                replay_root=resolved_replay_root,
                log_dir=resolved_log_dir,
            ),
            backend_mode="provider_replay",
            task_id=resolved_task_id,
            replay_root=resolved_replay_root,
            log_dir=resolved_log_dir,
            snapshot_hidden_truth=False,
            active_surfaces=active_surfaces,
            active_capabilities=active_capabilities,
            surface_manifest=surface_manifest,
        )

    if resolved_mode == "live_provider":
        if resolved_task_id is None:
            raise ValueError("QIQCBENCH_TASK_ID is required for live_provider mode")
        if desc.build_live_provider_backend is None:
            raise ValueError(
                f"Qtype {public.qtype!r} does not support backend mode 'live_provider'"
            )
        backend_allowlist = _csv_env(env_map.get("QIQCBENCH_LIVE_BACKEND_ALLOWLIST"))
        backend_name = backend_allowlist[0] if backend_allowlist else public.device_id
        try:
            backend = desc.build_live_provider_backend(
                task_id=resolved_task_id,
                public=public,
                backend_name=backend_name,
                log_dir=resolved_log_dir,
                token=env_map.get("QISKIT_IBM_TOKEN"),
                instance=env_map.get("QISKIT_IBM_INSTANCE"),
                channel=env_map.get("QISKIT_IBM_CHANNEL") or IBM_QUANTUM_PLATFORM_CHANNEL,
                allow_live_provider=env_map.get("QIQCBENCH_ALLOW_LIVE_PROVIDER") == "1",
                max_shots=_int_env(
                    env_map.get("QIQCBENCH_LIVE_MAX_SHOTS"),
                    default=public.max_shots,
                ),
                max_jobs=_int_env(
                    env_map.get("QIQCBENCH_LIVE_MAX_JOBS"),
                    default=1,
                ),
                backend_allowlist=backend_allowlist,
                wall_clock_s=_float_env(
                    env_map.get("QIQCBENCH_LIVE_WALL_CLOCK_S"),
                    default=300.0,
                ),
            )
        except ImportError as exc:
            raise RuntimeError(
                "live_provider mode requires the optional [live] extras; "
                "install with: pip install 'qiqcbench[live]'"
            ) from exc
        return BackendRuntime(
            backend=backend,
            backend_mode="live_provider",
            task_id=resolved_task_id,
            replay_root=resolved_replay_root,
            log_dir=resolved_log_dir,
            snapshot_hidden_truth=False,
            active_surfaces=active_surfaces,
            active_capabilities=active_capabilities,
            surface_manifest=surface_manifest,
        )

    raise AssertionError(f"Unhandled backend mode {resolved_mode!r}")
