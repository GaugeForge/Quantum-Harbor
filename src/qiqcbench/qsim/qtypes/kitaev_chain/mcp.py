"""MCP tool registration for the Kitaev-chain qtype.

Owns the qtype-local MCP surface: the instructions string and a
``register_mcp_tools`` callable that attaches the three charge-readout tools when
the ``kitaev_charge_readout`` capability is active. Always-on tools
(list_devices, get_device_spec, get_lab_notebook, get_job_result,
submit_final_answer) are registered by the server.

The instructions are neutral (two-world litmus): they describe the available
probes and their raw outputs, never the hidden regime, the defect, or the
expected scaling.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from qiqcbench.qsim.actions import kitaev_chain as kitaev_actions
from qiqcbench.qsim.actions.base import ActionContext

if TYPE_CHECKING:  # pragma: no cover - import-only typing aid
    from mcp.server.fastmcp import FastMCP


INSTRUCTIONS = (
    "Quantum simulator MCP server for a semiconductor Kitaev chain (a 1D array "
    "of quantum dots coupled through superconductor hybrid segments), operated "
    "through programmable gates and charge readout. Available probes (each "
    "returns a job_id; poll get_job_result):\n"
    "- run_charge_stability(bond, mu_ld_values, mu_rd_values, shots): base-band "
    "pulse a 2-dot readout window onto one bond and scan the two dot detunings. "
    "Returns raw per-shot parity assignments from two channels -- a global "
    "quantum-capacitance channel and a local charge sensor -- packed as base64 "
    "of np.packbits over a [n_grid][shots] array (grid index = i_ld*len(rd)+i_rd). "
    "Form the Pearson correlation between the two channels to study parity.\n"
    "- run_protection_sweep(mu_common_values, shots): apply a common-mode "
    "detuning to all dots and read out parity; returns raw per-shot parity bits "
    "per detuning ([n_mu][shots]). The parity polarization P_M = <1 - 2*bit> "
    "tracks the edge-mode energy splitting.\n"
    "- run_subchain_spectroscopy(site_a, site_b, shots): measure the bulk "
    "excitation gap (micro-eV) of the contiguous sub-chain spanning dots "
    "site_a..site_b.\n"
    "- run_majorana_pulse_batch(sequences, shots): when the Majorana-qubit gate "
    "capability is active, drive the fixed-parity logical qubit with arbitrary "
    "time-domain pulse sequences and read out levels 0, 1, or aggregate "
    "non-computational level 2 per shot.\n"
    "- run_majorana_clifford_rb(gate_x90_segments, gate_z90_segments, lengths, "
    "sequences_per_length, shots_per_sequence): privately sample uniform "
    "single-qubit Cliffords, compile them and the exact inverse through one "
    "canonical X90/Z90 compiler, and return raw level-resolved shots.\n"
    "Calibrations in the lab notebook may be stale; verify. Use "
    "submit_final_answer when ready."
)


def register_mcp_tools(mcp: FastMCP, state: Any) -> None:
    """Attach the charge-readout MCP tools for the given qsim state."""

    ctx = ActionContext(state=state, surface="mcp")

    if "kitaev_charge_readout" in state.active_capabilities:

        @mcp.tool()
        def run_charge_stability(
            device_id: str,
            bond: int,
            mu_ld_values: list[float],
            mu_rd_values: list[float],
            shots: int = 2048,
        ) -> dict:
            """Windowed 2-dot charge-stability scan on one bond. Returns a job_id.

            Reads out parity on the (dot `bond`, dot `bond`+1) window over the
            product grid of `mu_ld_values` x `mu_rd_values` (micro-eV detunings).
            Each grid point returns `shots` per-shot bits in each of two channels
            (global quantum capacitance, local charge sensor); see the server
            instructions for the packing.
            """
            return kitaev_actions.submit_charge_stability(
                ctx,
                device_id=device_id,
                bond=bond,
                mu_ld_values=mu_ld_values,
                mu_rd_values=mu_rd_values,
                shots=shots,
            )

        @mcp.tool()
        def run_protection_sweep(
            device_id: str,
            mu_common_values: list[float],
            shots: int = 8192,
        ) -> dict:
            """Common-mode detuning sweep with parity readout. Returns a job_id.

            Applies the same detuning to every dot for each value in
            `mu_common_values` (micro-eV) and returns `shots` per-shot parity
            bits per value.
            """
            return kitaev_actions.submit_protection_sweep(
                ctx, device_id=device_id, mu_common_values=mu_common_values, shots=shots
            )

        @mcp.tool()
        def run_subchain_spectroscopy(
            device_id: str,
            site_a: int,
            site_b: int,
            shots: int = 8192,
        ) -> dict:
            """Measure the bulk excitation gap of a sub-chain. Returns a job_id.

            Spectroscopy of the contiguous window dots `site_a`..`site_b`; the
            measured gap (micro-eV) has finite-shot measurement noise. The window
            must span at least one bond (`site_a` < `site_b`): a one-site window
            has no bulk excitation, so it has no bulk gap to measure.
            """
            return kitaev_actions.submit_subchain_spectroscopy(
                ctx, device_id=device_id, site_a=site_a, site_b=site_b, shots=shots
            )

    if "majorana_pulse_control" in state.active_capabilities:

        @mcp.tool()
        def run_majorana_pulse_batch(
            device_id: str,
            sequences: list[list[dict]],
            shots: int = 512,
        ) -> dict:
            """Run a batch of time-domain pulse sequences on the logical Majorana qubit.

            Returns a job_id. Each entry of `sequences` is a list of drive segments
            applied in order from the prepared logical |0>; a segment is a dict with
            keys `amp` (0..1), `phase_rad` (rotation axis in the x-y plane),
            `detuning_ueV`, `duration_ns`, and optional `drag` (a constant
            quadrature ratio, not a derivative waveform).
            With zero detuning and `drag=0`, positive amplitude at `phase_rad=0`
            implements U=exp(-i theta X/2). With zero amplitude, positive
            detuning implements U=exp(-i theta Z/2), for theta > 0.
            The realized Rabi rate is `amp * drive_rate`, where `drive_rate` is a hidden
            device parameter and the lab-notebook estimate may be stale. The qubit sits
            below a bulk gap, so fast/strong pulses leak; charge noise + poisoning add a
            time-dependent floor.

            Returns raw per-shot outcomes (`levels_b64`: 0 and 1 are the
            computational states; 2 is outside that manifold).
            """
            return kitaev_actions.submit_pulse_batch(
                ctx,
                device_id=device_id,
                sequences=sequences,
                shots=shots,
            )

        @mcp.tool()
        def run_majorana_clifford_rb(
            device_id: str,
            gate_x90_segments: list[dict],
            gate_z90_segments: list[dict],
            lengths: list[int],
            sequences_per_length: int = 16,
            shots_per_sequence: int = 400,
        ) -> dict:
            """Run uniform 24-element Clifford RB with private qsim randomization.

            qsim compiles every sampled Clifford and its exact inverse through one
            canonical compiler over your submitted X90/Z90 pulse primitives. The
            named primitives are sign-sensitive positive rotations:
            X90=exp(-i*pi*X/4) and Z90=exp(-i*pi*Z/4).
            The short-sequence design must include at least three distinct lengths in
            [1,8]. Poll the returned job_id for raw level outcomes.
            """

            return kitaev_actions.submit_clifford_rb(
                ctx,
                device_id=device_id,
                gate_x90_segments=gate_x90_segments,
                gate_z90_segments=gate_z90_segments,
                lengths=lengths,
                sequences_per_length=sequences_per_length,
                shots_per_sequence=shots_per_sequence,
            )


__all__ = ["INSTRUCTIONS", "register_mcp_tools"]
