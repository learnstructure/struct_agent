"""
Dynamics tools.

Provides time-history and transient dynamic analysis operations using ``structdyn``
bridged with ``fem2d`` and ``fem3d`` structural models.

Features:
- Earthquake ground motion time-history analysis (Newmark-Beta / Central Difference)
- Dynamic force enveloping across all time steps into ElementResult
- Direct compatibility with downstream ACI 318 reinforced concrete design
- Dynamic result inspection and peak response summarization
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
import numpy as np

from struct_core import (
    AnalysisResult,
    BeamSectionForce,
    ElementResult,
    ModalResult,
    ModeShape,
    NodeDisplacement,
    NodeResult,
    TimeHistoryResult,
    TimeStepResult,
)
from ..validation.analysis import validate_analysis
from .analysis import _detect_solver
from .base import AgentError, ToolResult


def run_time_history(
    project: Any,
    *,
    analysis_case_id: str,
    ground_motion: Any = None,
    earthquake: str = "elcentro",
    direction: str = "x",
    dt: Optional[float] = None,
    scale_factor: Optional[float] = None,
    damping_ratio: float = 0.05,
    n_modes_damping: int = 3,
    method: str = "newmark_beta",
    solver: str = "auto",
    envelope_to_design: bool = True,
) -> ToolResult:
    """
    Run dynamic earthquake time-history analysis using structdyn on a 2D or 3D model.

    Parameters
    ----------
    project
        A ``struct_core.Project`` containing the structural model.
    analysis_case_id
        A unique identifier to store the dynamic results on ``project.analysis_results``.
    ground_motion
        Optional ``GroundMotion`` instance, 1D array of accelerations, or None to use preset.
    earthquake
        Preset earthquake record when ``ground_motion`` is None (default: ``"elcentro"``).
    direction
        Excitation direction (``"x"``, ``"y"``, or ``"z"``).
    dt
        Time step in seconds (defaults to record dt, e.g. 0.02 s for El Centro).
    scale_factor
        Acceleration multiplier to convert record to model units. If None, automatically
        detected from ``project.metadata.units.length`` (9.80665 for m, 386.09 for in, 32.174 for ft).
    damping_ratio
        Modal damping ratio zeta (default: 0.05 for 5% damping).
    n_modes_damping
        Number of modes to apply Rayleigh/modal damping to (default: 3).
    method
        Integration method: ``"newmark_beta"`` (default) or ``"central_difference"``.
    solver
        ``"auto"`` (detect 2D vs 3D), ``"fem2d"``, or ``"fem3d"``.
    envelope_to_design
        If True, calculates peak dynamic force envelopes (axial P, shear V, moment M)
        across all time steps for each member, populating ``ElementResult`` so downstream
        ``run_design`` (ACI 318) can immediately verify/design the members.
    """
    if project is None or getattr(project, "model", None) is None:
        return ToolResult.failure(
            "run_time_history",
            AgentError(
                error_type="missing_input",
                message="No Project supplied to run_time_history.",
                possible_causes=["Project was never created/loaded"],
            ),
        )

    # 1. Resolve Ground Motion
    try:
        from structdyn import GroundMotion, elcentro_chopra
    except ImportError as exc:
        return ToolResult.failure(
            "run_time_history",
            AgentError(
                error_type="dependency_error",
                message=f"structdyn is required for dynamic analysis: {exc}",
            ),
        )

    if ground_motion is not None:
        if isinstance(ground_motion, GroundMotion):
            gm = ground_motion
        elif isinstance(ground_motion, (list, np.ndarray)):
            dt_val = dt or 0.02
            sf = scale_factor if scale_factor is not None else 1.0
            gm = GroundMotion.from_arrays(np.asarray(ground_motion), dt=dt_val, scale_factor=sf)
        else:
            return ToolResult.failure(
                "run_time_history",
                AgentError(
                    error_type="invalid_input",
                    message=f"Unsupported ground_motion type: {type(ground_motion)}",
                ),
            )
    else:
        # Load preset earthquake
        if earthquake.lower() in ("elcentro", "el_centro", "el-centro"):
            elc = elcentro_chopra()
            dt_val = dt or 0.02
            if scale_factor is not None:
                sf = float(scale_factor)
            else:
                length_unit = getattr(getattr(getattr(project, "metadata", None), "units", None), "length", "m")
                length_str = str(length_unit).lower()
                if "in" in length_str:
                    sf = 386.09
                elif "ft" in length_str:
                    sf = 32.174
                else:
                    sf = 9.80665  # SI m/s^2
            gm = GroundMotion.from_arrays(elc["acc (g)"], dt=dt_val, scale_factor=sf)
        else:
            return ToolResult.failure(
                "run_time_history",
                AgentError(
                    error_type="unsupported_earthquake",
                    message=f"Unknown earthquake preset {earthquake!r}. Use 'elcentro' or provide custom ground_motion.",
                ),
            )

    # 2. Resolve Solver & Build Structure
    resolved_solver = solver.lower().strip()
    if resolved_solver == "auto":
        resolved_solver = _detect_solver(project)

    try:
        if resolved_solver == "fem2d":
            from fem2d.adapters import model_from_core as fem2d_model_from_core
            structure = fem2d_model_from_core(project)
        else:
            from fem3d.adapters import model_from_core as fem3d_model_from_core
            structure = fem3d_model_from_core(project)
    except Exception as exc:
        return ToolResult.failure(
            "run_time_history",
            AgentError(
                error_type="adapter_failure",
                message=f"Failed to build {resolved_solver} model from project: {exc}",
            ),
        )

    structure.number_dofs()
    structure.apply_boundary_conditions()
    if hasattr(structure, "assemble_mass_matrix"):
        structure.assemble_mass_matrix()

    from structdyn import MDF
    mdf = MDF.from_fem(structure, zeta=damping_ratio, n_modes=n_modes_damping)

    if np.max(np.abs(mdf.M)) < 1e-12:
        return ToolResult.failure(
            "run_time_history",
            AgentError(
                error_type="zero_mass",
                message=f"{resolved_solver} model has zero mass matrix. Assign material density (rho) or nodal lumped mass.",
                possible_causes=[
                    "Material schema lacks rho",
                    "Node schemas lack lumped mass",
                ],
            ),
        )

    # 3. Time-History Integration
    try:
        resp = mdf.find_response_ground_motion(
            gm,
            direction=direction,
            method=method,
        )
    except Exception as exc:
        return ToolResult.failure(
            "run_time_history",
            AgentError(
                error_type="solver_failure",
                message=f"Time-history integration failed: {exc}",
            ),
        )

    # 4. Extract Dynamic Response Histories
    time_series = resp["time"].values if "time" in resp else np.arange(len(resp)) * gm.dt
    n_t = len(time_series)
    ndof = mdf.ndof

    # u_free shape: (n_t, ndof)
    u_free = np.column_stack([resp[f"u{i+1}"].values for i in range(ndof)])

    # Map back to full global DOF space: (n_t, neq)
    u_global_all = np.zeros((n_t, structure.neq))
    u_global_all[:, structure.free_dofs] = u_free

    # 5. Extract Peak Nodal Displacements & Roof Response
    node_results: Dict[Any, NodeResult] = {}
    max_lateral_disp = 0.0
    roof_node_id = None
    max_height = -float("inf")

    for nid, node in structure.nodes.items():
        dofs = getattr(node, "dofs", [])
        u_node = u_global_all[:, dofs]
        peak_u = np.max(np.abs(u_node), axis=0)

        # Height tracking
        h_coord = float(getattr(node, "z", getattr(node, "y", 0.0)))
        if h_coord > max_height:
            max_height = h_coord
            roof_node_id = nid

        if len(dofs) == 6:
            nd = NodeDisplacement(
                ux=float(peak_u[0]),
                uy=float(peak_u[1]),
                uz=float(peak_u[2]),
                rx=float(peak_u[3]),
                ry=float(peak_u[4]),
                rz=float(peak_u[5]),
            )
            lat_disp = float(peak_u[0]) if direction.lower() == "x" else float(peak_u[1])
        elif len(dofs) == 3:
            nd = NodeDisplacement(
                ux=float(peak_u[0]),
                uy=float(peak_u[1]),
                rz=float(peak_u[2]),
            )
            lat_disp = float(peak_u[0]) if direction.lower() == "x" else float(peak_u[1])
        else:
            nd = NodeDisplacement()
            lat_disp = 0.0

        max_lateral_disp = max(max_lateral_disp, lat_disp)
        node_results[nid] = NodeResult(node_id=nid, displacement=nd)

    # 6. Element Force Envelopes for Downstream Code Design
    element_results: Dict[Any, ElementResult] = {}
    max_structure_moment = 0.0
    max_structure_shear = 0.0
    max_structure_axial = 0.0

    if envelope_to_design:
        for eid, el in structure.elements.items():
            dofs = el.node_i.dofs + el.node_j.dofs
            u_elem_g = u_global_all[:, dofs].T  # (n_elem_dofs, n_t)
            T = el.transformation_matrix()
            k = el.local_stiffness()
            u_elem_local = T @ u_elem_g
            f_local = k @ u_elem_local  # (n_elem_dofs, n_t)
            abs_f = np.abs(f_local)

            if len(dofs) == 6:  # 2D Frame Element
                p_max = float(np.max(abs_f[[0, 3], :]))
                v_max = float(np.max(abs_f[[1, 4], :]))
                m_max = float(np.max(abs_f[[2, 5], :]))

                station_i = BeamSectionForce(
                    station=0.0,
                    P=float(np.max(abs_f[0, :])),
                    V2=float(np.max(abs_f[1, :])),
                    M3=float(np.max(abs_f[2, :])),
                )
                station_j = BeamSectionForce(
                    station=1.0,
                    P=float(np.max(abs_f[3, :])),
                    V2=float(np.max(abs_f[4, :])),
                    M3=float(np.max(abs_f[5, :])),
                )
            else:  # 3D Frame Element (12 DOFs)
                p_max = float(np.max(abs_f[[0, 6], :]))
                v_max = float(np.max(abs_f[[1, 2, 7, 8], :]))
                m_max = float(np.max(abs_f[[4, 5, 10, 11], :]))

                station_i = BeamSectionForce(
                    station=0.0,
                    P=float(np.max(abs_f[0, :])),
                    V2=float(np.max(abs_f[1, :])),
                    V3=float(np.max(abs_f[2, :])),
                    T=float(np.max(abs_f[3, :])),
                    M2=float(np.max(abs_f[4, :])),
                    M3=float(np.max(abs_f[5, :])),
                )
                station_j = BeamSectionForce(
                    station=1.0,
                    P=float(np.max(abs_f[6, :])),
                    V2=float(np.max(abs_f[7, :])),
                    V3=float(np.max(abs_f[8, :])),
                    T=float(np.max(abs_f[9, :])),
                    M2=float(np.max(abs_f[10, :])),
                    M3=float(np.max(abs_f[11, :])),
                )

            max_structure_moment = max(max_structure_moment, m_max)
            max_structure_shear = max(max_structure_shear, v_max)
            max_structure_axial = max(max_structure_axial, p_max)

            element_results[str(eid)] = ElementResult(
                element_id=eid,
                forces=[station_i, station_j],
                max_axial=p_max,
                max_moment=m_max,
                max_shear=v_max,
            )

    # 7. Modal Eigenvalues & TimeHistoryResult schema
    omega, phi = mdf.modal.modal_analysis(n_modes=min(3, mdf.ndof))
    freqs = omega / (2.0 * np.pi)
    periods = np.where(freqs > 1e-10, 1.0 / freqs, 0.0)
    modes = [
        ModeShape(
            mode_number=i + 1,
            period=max(float(periods[i]), 1e-6),
            frequency=max(float(freqs[i]), 1e-6),
            circular_frequency=max(float(omega[i]), 1e-6),
        )
        for i in range(len(omega))
    ]
    modal_result = ModalResult(modes=modes)

    # Subsample response series for schema storage (up to 100 snapshots)
    sample_indices = np.linspace(0, n_t - 1, min(100, n_t), dtype=int)
    th_steps: List[TimeStepResult] = []
    for idx in sample_indices:
        t_val = float(time_series[idx])
        step_disps = {}
        for nid, node in structure.nodes.items():
            dofs = getattr(node, "dofs", [])
            u_t = u_global_all[idx, dofs]
            if len(dofs) == 6:
                step_disps[nid] = NodeDisplacement(
                    ux=float(u_t[0]), uy=float(u_t[1]), uz=float(u_t[2]),
                    rx=float(u_t[3]), ry=float(u_t[4]), rz=float(u_t[5]),
                )
            elif len(dofs) == 3:
                step_disps[nid] = NodeDisplacement(
                    ux=float(u_t[0]), uy=float(u_t[1]), rz=float(u_t[2]),
                )
        th_steps.append(TimeStepResult(time=t_val, displacements=step_disps))

    th_result = TimeHistoryResult(dt=float(gm.dt), time_steps=th_steps)

    # 8. Assemble AnalysisResult
    analysis_result = AnalysisResult(
        analysis_case_id=analysis_case_id,
        element_results=element_results,
        node_results=node_results,
        modal_result=modal_result,
        time_history_result=th_result,
    )

    project.analysis_results[analysis_case_id] = analysis_result
    if "dynamic_envelope" not in project.analysis_results:
        project.analysis_results["dynamic_envelope"] = analysis_result

    validation = validate_analysis(analysis_result)
    if not validation.ok:
        return ToolResult.failure(
            "run_time_history",
            AgentError(
                error_type="empty_result",
                message="Time-history completed but produced invalid results.",
                possible_causes=validation.error.possible_causes if validation.error else [],
            ),
        )

    out_data = {
        "analysis_case_id": analysis_case_id,
        "solver_used": resolved_solver,
        "earthquake": earthquake,
        "direction": direction,
        "duration": float(time_series[-1]),
        "n_steps": n_t,
        "pga": float(np.max(np.abs(gm.acceleration))),
        "peak_lateral_displacement": max_lateral_disp,
        "roof_node": roof_node_id,
        "fundamental_period": float(periods[0]) if len(periods) > 0 else 0.0,
        "max_axial_envelope": max_structure_axial,
        "max_shear_envelope": max_structure_shear,
        "max_moment_envelope": max_structure_moment,
        "n_element_results": len(element_results),
    }

    return ToolResult.success(
        "run_time_history",
        data=out_data,
        next_suggested_stage="ANALYSIS_COMPLETE",
    )


def inspect_dynamic_results(project: Any, analysis_case_id: str) -> ToolResult:
    """Return a structured summary of dynamic time-history results."""
    if project is None or not getattr(project, "analysis_results", None):
        return ToolResult.failure(
            "inspect_dynamic_results",
            AgentError(
                error_type="missing_result",
                message="No analysis results attached to the project.",
                possible_causes=["run_time_history was never called"],
            ),
        )

    result = project.analysis_results.get(analysis_case_id)
    if result is None:
        return ToolResult.failure(
            "inspect_dynamic_results",
            AgentError(
                error_type="unknown_case",
                message=f"Analysis case {analysis_case_id!r} not found.",
                possible_causes=[f"Available cases: {list(project.analysis_results.keys())}"],
            ),
        )

    th = getattr(result, "time_history_result", None)
    summary: Dict[str, Any] = {
        "analysis_case_id": analysis_case_id,
        "has_time_history": th is not None,
    }
    if th is not None:
        summary["dt"] = float(th.dt)
        summary["n_recorded_steps"] = len(th.time_steps)
        if th.time_steps:
            summary["duration"] = float(th.time_steps[-1].time)

    if getattr(result, "modal_result", None) and getattr(result.modal_result, "modes", None):
        summary["fundamental_period"] = float(result.modal_result.modes[0].period)
        summary["periods"] = [float(m.period) for m in result.modal_result.modes]

    # Peak displacements
    peak_ux = {nid: float(getattr(nr.displacement, "ux", 0.0)) for nid, nr in result.node_results.items()}
    peak_uy = {nid: float(getattr(nr.displacement, "uy", 0.0)) for nid, nr in result.node_results.items()}
    summary["max_displacement_x"] = max(peak_ux.values(), default=0.0)
    summary["max_displacement_y"] = max(peak_uy.values(), default=0.0)

    # Element force envelopes
    elements_summary = {}
    for eid, er in result.element_results.items():
        elements_summary[str(eid)] = {
            "max_axial": float(getattr(er, "max_axial", 0.0) or 0.0),
            "max_shear": float(getattr(er, "max_shear", 0.0) or 0.0),
            "max_moment": float(getattr(er, "max_moment", 0.0) or 0.0),
        }
    summary["elements"] = elements_summary

    return ToolResult.success("inspect_dynamic_results", data=summary)
