"""Materialize the ``time_budgeted_hamlearn_10q`` task onto the analog qtype.

Bridges the task's fixed hidden instance (``construction.py``) to the reusable
``blackbox_analog_dynamics`` device-config shapes, so qsim can run the task and
``get_device_spec`` returns the public term dictionary + probe constraints.

* ``build_public_device_spec()`` -> dict that validates as ``PublicAnalogSpec``
  (the public spec + the 111-term dictionary; no coefficients).
* ``build_hidden_device_config()`` -> dict that validates as
  ``HiddenAnalogConfig`` (the 24 true coefficients, the RNG salt, the stale
  prior). This is hidden truth: it is committed only as ``*.hidden.example.yaml``
  and is never mounted into the agent container.

Task code depending on the reusable qtype models is the intended direction; the
qtype never imports this module.
"""

from __future__ import annotations

import io
from pathlib import Path

from ruamel.yaml import YAML

from qiqcbench.qsim.hidden_dynamics.time_budgeted_hamlearn_10q.construction import (
    HamLearnInstance,
    build_instance,
)
from qiqcbench.qsim.hidden_dynamics.time_budgeted_hamlearn_10q.dictionary import (
    N_TERMS,
    TERM_ORDER,
    pauli_string,
)
from qiqcbench.qsim.qtypes.blackbox_analog_dynamics.device import (
    HiddenAnalogConfig,
    PublicAnalogSpec,
)

__all__ = [
    "DEVICE_ID",
    "NOISE_SEED",
    "analog_configs_from_instance",
    "build_hidden_device_config",
    "build_public_device_spec",
    "write_device_configs",
]

DEVICE_ID = "hamlearn_10q_chain_v1"
# Fixed private salt for the simulator RNG. The backend mixes it with fresh
# per-run entropy, and the Hamiltonian instance does not depend on it.
NOISE_SEED = 2026061719


def analog_configs_from_instance(
    instance: HamLearnInstance,
    *,
    device_id: str = DEVICE_ID,
    seed: int = NOISE_SEED,
) -> tuple[PublicAnalogSpec, HiddenAnalogConfig]:
    """Typed analog device configs derived entirely from a hidden instance.

    Single source of the instance -> analog-qtype-config mapping, shared by the
    committed device-YAML generators and the feasibility-gate :class:`Oracle`
    (so both the gate and the production device path execute one engine).
    """
    public_dict = dict(instance.public_spec)
    public_dict.update(
        device_id=device_id,
        qtype="blackbox_analog_dynamics",
        num_terms=N_TERMS,
        terms=instance.hamiltonian_dictionary["terms"],
    )
    public = PublicAnalogSpec.model_validate(public_dict)

    terms = [
        {"pauli": pauli_string(term), "coefficient": float(coeff)}
        for term, coeff in zip(TERM_ORDER, instance.omega_star, strict=True)
        if coeff != 0.0
    ]
    stale = instance.stale_notebook
    hidden = HiddenAnalogConfig.model_validate(
        {
            "device_id": device_id,
            "qtype": "blackbox_analog_dynamics",
            "seed": seed,
            "hamiltonian_terms": terms,
            "stale_lab_notebook": {
                "coefficients_rad_per_us": dict(stale["coefficients_rad_per_us"]),
                "note": stale["note"],
                "reliability": stale["reliability"],
            },
        }
    )
    return public, hidden


def build_public_device_spec() -> dict:
    """Public device spec for ``get_device_spec`` (validated as PublicAnalogSpec)."""
    public, _ = analog_configs_from_instance(build_instance())
    return public.model_dump()


def build_hidden_device_config() -> dict:
    """Hidden device config (validated as HiddenAnalogConfig). Hidden truth."""
    _, hidden = analog_configs_from_instance(build_instance())
    return hidden.model_dump()


def _yaml_bytes(payload: dict) -> bytes:
    yaml = YAML()
    yaml.default_flow_style = False
    yaml.width = 4096
    buf = io.BytesIO()
    yaml.dump(payload, buf)
    return buf.getvalue()


def write_device_configs(devices_dir: Path) -> dict[str, Path]:
    """Write the public + hidden device YAMLs; return the written paths."""
    devices_dir = Path(devices_dir)
    devices_dir.mkdir(parents=True, exist_ok=True)
    public_path = devices_dir / f"{DEVICE_ID}.public.yaml"
    hidden_path = devices_dir / f"{DEVICE_ID}.hidden.example.yaml"
    public_path.write_bytes(_yaml_bytes(build_public_device_spec()))
    hidden_path.write_bytes(_yaml_bytes(build_hidden_device_config()))
    return {"public": public_path, "hidden": hidden_path}
