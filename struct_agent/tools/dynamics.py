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
    ground_motion: Any = "elcentro",
    component: Optional[str] = None,
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
        Ground motion to analyze (default: ``"elcentro"``). Can be:
        - Preset string name (e.g. ``"elcentro"``, ``"northridge"``, ``"loma_prieta"``, ``"san_fernando"``)
        - Composite string with component (e.g. ``"northridge:hor2"``)
        - A ``GroundMotion`` instance
        - A 1D list or numpy array of acceleration values
    component
        Component to load when ``ground_motion`` is a string (``"hor1"``, ``"hor2"``, ``"up"``).
        If None, automatically inferred from ``direction``:
        - direction="x" -> "hor1"
        - direction="y" -> "hor2"
        - direction="z" -> "up"
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

    # 1. Resolve Scale Factor
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

    # 2. Resolve Ground Motion
    try:
        from structdyn import GroundMotion
    except ImportError as exc:
        return ToolResult.failure(
            "run_time_history",
            AgentError(
                error_type="dependency_error",
                message=f"structdyn is required for dynamic analysis: {exc}",
            ),
        )

    if isinstance(ground_motion, GroundMotion):
        gm = ground_motion
    elif isinstance(ground_motion, (list, np.ndarray)):
        dt_val = dt or 0.02
        gm = GroundMotion.from_arrays(np.asarray(ground_motion), dt=dt_val, scale_factor=sf)
    elif isinstance(ground_motion, str):
        direction_comp_map = {"x": "hor1", "y": "hor2", "z": "up"}
        comp = component or direction_comp_map.get(str(direction).lower(), "hor1")
        try:
            gm = GroundMotion.from_event(ground_motion, component=comp, scale_factor=sf)
            if dt is not None:
                gm.dt = float(dt)
        except Exception as exc:
            return ToolResult.failure(
                "run_time_history",
                AgentError(
                    error_type="unsupported_ground_motion",
                    message=f"Failed to load ground motion record {ground_motion!r}: {exc}",
                    possible_causes=[
                        f"Available events: {list(GroundMotion.available_events().keys())}",
                        "Or pass a custom GroundMotion instance or array in ground_motion.",
                    ],
                ),
            )
    else:
        return ToolResult.failure(
            "run_time_history",
            AgentError(
                error_type="invalid_input",
                message=f"Unsupported ground_motion type: {type(ground_motion)}",
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
        "ground_motion": getattr(gm, "name", str(ground_motion)),
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


def run_response_spectrum(
    project: Any,
    *,
    analysis_case_id: str,
    sds: Optional[float] = None,
    sd1: Optional[float] = None,
    tl: float = 8.0,
    spectrum_curve: Optional[Any] = None,
    ground_motion: Any = None,
    component: Optional[str] = None,
    Sa: Optional[Union[List[float], np.ndarray]] = None,
    Sd: Optional[Union[List[float], np.ndarray]] = None,
    direction: str = "x",
    combination: str = "CQC",
    n_modes: Optional[int] = None,
    damping_ratio: float = 0.05,
    scale_factor: Optional[float] = None,
    solver: str = "auto",
    envelope_to_design: bool = True,
) -> ToolResult:
    """
    Run linear dynamic Response Spectrum Analysis (RSA) using structdyn bridged with fem2d/fem3d.

    Parameters
    ----------
    project
        A struct_core.Project containing the structural model.
    analysis_case_id
        A unique identifier to store the RSA results on project.analysis_results.
    sds
        ASCE 7 short-period design spectral acceleration SDS (in g).
    sd1
        ASCE 7 1-second design spectral acceleration SD1 (in g).
    tl
        ASCE 7 long-period transition period (seconds). Defaults to 8.0.
    spectrum_curve
        User-supplied continuous response spectrum curve Sa(T).
    ground_motion
        Optional ground motion to generate response spectrum dynamically. Can be:
        - Preset string name (e.g. ``"elcentro"``, ``"northridge"``, ``"loma_prieta"``, ``"san_fernando"``)
        - Composite string with component (e.g. ``"northridge:hor2"``)
        - A ``GroundMotion`` instance
        - A 1D list or numpy array of acceleration values
    component
        Component to load when ``ground_motion`` is a string (``"hor1"``, ``"hor2"``, ``"up"``).
        If None, automatically inferred from ``direction``:
        - direction="x" -> "hor1"
        - direction="y" -> "hor2"
        - direction="z" -> "up"
    Sa
        Explicit list of modal spectral accelerations.
    Sd
        Explicit list of modal spectral displacements.
    direction
        Excitation direction ('x', 'y', or 'z'). Defaults to 'x'.
    combination
        Modal combination rule: 'CQC' (default), 'SRSS', or 'ABS'.
    n_modes
        Number of modes to consider.
    damping_ratio
        Modal damping ratio (default: 0.05).
    scale_factor
        Acceleration multiplier to convert g to model units.
        If None, automatically detected from project units (9.80665 for m, 386.09 for in, 32.174 for ft).
    solver
        'auto', 'fem2d', or 'fem3d'.
    envelope_to_design
        If True, writes station forces into ElementResult so downstream run_design (ACI 318)
        can size/check reinforcement directly.
    """
    if project is None or getattr(project, "model", None) is None:
        return ToolResult.failure(
            "run_response_spectrum",
            AgentError(
                error_type="missing_input",
                message="No Project supplied to run_response_spectrum.",
                possible_causes=["Project was never created/loaded"],
            ),
        )

    # 1. Resolve scale factor
    if scale_factor is not None:
        sf = float(scale_factor)
    else:
        unit_len = "m"
        if getattr(project, "metadata", None) and getattr(project.metadata, "units", None):
            unit_len = str(getattr(project.metadata.units, "length", "m")).lower()
        elif getattr(project, "units", None):
            unit_len = str(getattr(project.units, "length", "m")).lower()

        if unit_len in ("in", "inch", "inches"):
            sf = 386.09
        elif unit_len in ("ft", "foot", "feet"):
            sf = 32.174
        elif unit_len in ("mm", "millimeter"):
            sf = 9806.65
        else:
            sf = 9.80665  # SI m/s^2

    # 2. Resolve ground motion or default spectrum if nothing supplied
    resolved_gm = None
    if (
        sds is None
        and sd1 is None
        and spectrum_curve is None
        and Sa is None
        and Sd is None
        and ground_motion is None
    ):
        from structdyn import GroundMotion
        direction_comp_map = {"x": "hor1", "y": "hor2", "z": "up"}
        comp = component or direction_comp_map.get(str(direction).lower(), "hor1")
        try:
            resolved_gm = GroundMotion.from_event("elcentro", component=comp, scale_factor=sf)
        except Exception:
            sds = 1.0
            sd1 = 0.6
    elif ground_motion is not None:
        from structdyn import GroundMotion
        if isinstance(ground_motion, GroundMotion):
            resolved_gm = ground_motion
        elif isinstance(ground_motion, (list, np.ndarray)):
            resolved_gm = GroundMotion.from_arrays(np.asarray(ground_motion), dt=0.02, scale_factor=sf)
        elif isinstance(ground_motion, str):
            direction_comp_map = {"x": "hor1", "y": "hor2", "z": "up"}
            comp = component or direction_comp_map.get(str(direction).lower(), "hor1")
            try:
                resolved_gm = GroundMotion.from_event(ground_motion, component=comp, scale_factor=sf)
            except Exception as exc:
                return ToolResult.failure(
                    "run_response_spectrum",
                    AgentError(
                        error_type="unsupported_ground_motion",
                        message=f"Failed to load ground motion record {ground_motion!r}: {exc}",
                        possible_causes=[
                            f"Available events: {list(GroundMotion.available_events().keys())}",
                        ],
                    ),
                )

    # 3. Resolve solver and build structure
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
            "run_response_spectrum",
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

    mdf = MDF.from_fem(structure, zeta=damping_ratio)
    if np.max(np.abs(mdf.M)) < 1e-12:
        return ToolResult.failure(
            "run_response_spectrum",
            AgentError(
                error_type="zero_mass",
                message=f"{resolved_solver} model has zero mass matrix. Assign material density (rho) or nodal lumped mass.",
                possible_causes=[
                    "rho was left at 0.0 on materials",
                    "No story or nodal masses were assigned",
                ],
            ),
        )

    # 4. Execute Response Spectrum Analysis
    try:
        rsa_res = structure.response_spectrum_analysis(
            spectrum_curve=spectrum_curve,
            sds=sds,
            sd1=sd1,
            tl=tl,
            gm=resolved_gm,
            Sa=Sa,
            Sd=Sd,
            n_modes=n_modes,
            ji=damping_ratio,
            direction=direction,
            scale_factor=sf,
            combination=combination,
        )
    except Exception as exc:
        return ToolResult.failure(
            "run_response_spectrum",
            AgentError(
                error_type="rsa_execution_failed",
                message=f"Response spectrum analysis failed: {exc}",
            ),
        )

    # 5. Extract Nodal Displacements
    node_results: Dict[Any, NodeResult] = {}
    max_lateral_disp = 0.0

    for nid, node in structure.nodes.items():
        dofs = getattr(node, "dofs", [])
        if len(dofs) == 6:
            d = structure.disp[dofs]
            nd = NodeDisplacement(
                ux=float(d[0]), uy=float(d[1]), uz=float(d[2]),
                rx=float(d[3]), ry=float(d[4]), rz=float(d[5]),
            )
            lat_disp = float(np.hypot(d[0], d[1]))
        elif len(dofs) == 3:
            d = structure.disp[dofs]
            nd = NodeDisplacement(
                ux=float(d[0]), uy=float(d[1]), rz=float(d[2]),
            )
            lat_disp = float(abs(d[0]))
        else:
            nd = NodeDisplacement()
            lat_disp = 0.0

        max_lateral_disp = max(max_lateral_disp, lat_disp)
        node_results[str(nid)] = NodeResult(node_id=nid, displacement=nd)

    # 6. Extract Element Force Envelopes for Downstream Code Design
    element_results: Dict[Any, ElementResult] = {}
    comb_forces = rsa_res["combined_element_forces"]

    max_structure_moment = 0.0
    max_structure_shear = 0.0
    max_structure_axial = 0.0

    for eid, el in structure.elements.items():
        f_el = comb_forces.get(eid)
        if f_el is None:
            f_el = comb_forces.get(int(eid) if str(eid).isdigit() else eid)

        if f_el is not None:
            if len(f_el) == 6:  # 2D frame
                p_max = float(max(abs(f_el[0]), abs(f_el[3])))
                v_max = float(max(abs(f_el[1]), abs(f_el[4])))
                m_max = float(max(abs(f_el[2]), abs(f_el[5])))

                station_i = BeamSectionForce(
                    station=0.0,
                    P=float(abs(f_el[0])),
                    V2=float(abs(f_el[1])),
                    M3=float(abs(f_el[2])),
                )
                station_j = BeamSectionForce(
                    station=1.0,
                    P=float(abs(f_el[3])),
                    V2=float(abs(f_el[4])),
                    M3=float(abs(f_el[5])),
                )
            else:  # 3D frame (12 DOFs)
                p_max = float(max(abs(f_el[0]), abs(f_el[6])))
                v_max = float(max(abs(f_el[1]), abs(f_el[2]), abs(f_el[7]), abs(f_el[8])))
                m_max = float(max(abs(f_el[4]), abs(f_el[5]), abs(f_el[10]), abs(f_el[11])))

                station_i = BeamSectionForce(
                    station=0.0,
                    P=float(abs(f_el[0])),
                    V2=float(abs(f_el[1])),
                    V3=float(abs(f_el[2])),
                    T=float(abs(f_el[3])),
                    M2=float(abs(f_el[4])),
                    M3=float(abs(f_el[5])),
                )
                station_j = BeamSectionForce(
                    station=1.0,
                    P=float(abs(f_el[6])),
                    V2=float(abs(f_el[7])),
                    V3=float(abs(f_el[8])),
                    T=float(abs(f_el[9])),
                    M2=float(abs(f_el[10])),
                    M3=float(abs(f_el[11])),
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

    # 7. Modal Eigenvalues & ModalResult schema
    omega = rsa_res["omega"]
    periods = rsa_res["periods"]
    freqs = omega / (2.0 * np.pi)
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

    # 8. Assemble AnalysisResult
    analysis_result = AnalysisResult(
        analysis_case_id=analysis_case_id,
        element_results=element_results,
        node_results=node_results,
        modal_result=modal_result,
    )

    project.analysis_results[analysis_case_id] = analysis_result
    if envelope_to_design:
        project.analysis_results["dynamic_envelope"] = analysis_result

    validation = validate_analysis(analysis_result)
    if not validation.ok:
        return ToolResult.failure(
            "run_response_spectrum",
            AgentError(
                error_type="empty_result",
                message="Response spectrum completed but produced invalid results.",
                possible_causes=validation.error.possible_causes if validation.error else [],
            ),
        )

    combined_base_shear = float(rsa_res["combined_base_shear"][0])
    mmpr = rsa_res["mass_participation_ratio"]

    out_data = {
        "analysis_case_id": analysis_case_id,
        "solver_used": resolved_solver,
        "direction": direction,
        "combination_method": combination,
        "combined_base_shear": combined_base_shear,
        "fundamental_period": float(periods[0]) if len(periods) > 0 else 0.0,
        "cumulative_mass_participation": float(np.sum(mmpr)),
        "peak_lateral_displacement": max_lateral_disp,
        "max_axial_envelope": max_structure_axial,
        "max_shear_envelope": max_structure_shear,
        "max_moment_envelope": max_structure_moment,
        "n_element_results": len(element_results),
    }

    return ToolResult.success(
        "run_response_spectrum",
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
