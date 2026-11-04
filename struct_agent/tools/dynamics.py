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

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from struct_core import (
    AnalysisResult,
    BeamSectionForce,
    ElementResult,
    ModalResult,
    ModeShape,
    NodeDisplacement,
    NodeReaction,
    NodeResult,
    TimeHistoryResult,
    TimeStepResult,
)
from ..validation.analysis import validate_analysis
from .analysis import _detect_solver, modal_participation
from .base import AgentError, ToolResult
from .model_state import stamps_results


def ensure_seismic_mass_source(project: Any, mass_source: Optional[Any]) -> None:
    """Attach an ASCE 7-22 §12.7.2 mass source when neither the call nor the model defines one.

    Dead load 100 %, storage live 25 %, general floor live 0 %. Member self-mass comes from
    element density, so self-weight loads inside dead cases are not counted twice.
    """
    if mass_source is not None or getattr(project.model, "mass_source", None) is not None:
        return
    if not getattr(project.model, "load_cases", None):
        return
    from struct_load import create_asce7_mass_source

    project.model.mass_source = create_asce7_mass_source(project.model)


def _resolve_ground_motion(
    ground_motion: Any,
    *,
    component: Optional[str],
    direction: str,
    scale_factor: float,
    dt: Optional[float] = None,
):
    """Build a structdyn GroundMotion from any supported ``ground_motion`` input.

    Accepts a ``GroundMotion`` instance, an acceleration array in g (``dt`` defaults to 0.02 s),
    a record file path (PEER ``.AT2`` or two-column ``.csv`` of time [s] / acceleration [g]),
    or a built-in event name such as ``"elcentro"`` or ``"loma_prieta:hor2"``.
    """
    from structdyn import GroundMotion

    if isinstance(ground_motion, GroundMotion):
        return ground_motion
    if isinstance(ground_motion, (list, np.ndarray)):
        if dt is None:
            raise ValueError("dt is required for array ground_motion input")
        return GroundMotion.from_arrays(
            np.asarray(ground_motion, dtype=float), dt=dt, scale_factor=scale_factor
        )
    if isinstance(ground_motion, (str, Path)):
        text = str(ground_motion)
        if Path(text).suffix.lower() in (".at2", ".csv"):
            return GroundMotion.from_file(text, scale_factor=scale_factor)
        comp = component or {"x": "hor1", "y": "hor2", "z": "up"}.get(str(direction).lower(), "hor1")
        return GroundMotion.from_event(text, component=comp, scale_factor=scale_factor)
    raise TypeError(f"Unsupported ground_motion type: {type(ground_motion).__name__}")


_STATION_COMPONENTS = {6: ("P", "V2", "M3"), 12: ("P", "V2", "V3", "T", "M2", "M3")}


def _envelope_element_result(eid: Any, f_local: np.ndarray, *, signed: bool) -> ElementResult:
    """Envelope of member internal forces at STANDARD_STATIONS from local end forces.

    Dynamic responses carry no member loads, so internal-force diagrams are linear between the
    ends: at station s the action is (1 - s)(-f_i) + s f_j. ``f_local`` has shape (n_dofs,) or
    (n_dofs, n_t). For signed time histories the diagram is evaluated per step and enveloped; for
    sign-less spectral magnitudes the end magnitudes are interpolated, which bounds the CQC/SRSS
    combination of linear modal diagrams from above.

    The envelope holds magnitudes, so the sense of the axial force is unknown: ``combination_forces``
    carries the peak profile once as compression and once as tension, and design checks both.
    """
    from struct_core.analysis_result import STANDARD_STATIONS

    f = np.asarray(f_local, dtype=float)
    f = f.reshape(f.shape[0], -1)
    n = f.shape[0] // 2
    names = _STATION_COMPONENTS[f.shape[0]]
    stations = []
    for s in STANDARD_STATIONS:
        if signed:
            values = np.max(np.abs((1.0 - s) * (-f[:n]) + s * f[n:]), axis=1)
        else:
            values = np.max((1.0 - s) * np.abs(f[:n]) + s * np.abs(f[n:]), axis=1)
        stations.append(BeamSectionForce(station=s, **{c: float(v) for c, v in zip(names, values)}))
    compression = [st.model_copy(update={"P": -st.P}) for st in stations]
    return ElementResult(
        element_id=eid,
        forces=stations,
        combination_forces={"peak_tension": stations, "peak_compression": compression},
        max_axial=max(st.P for st in stations),
        max_moment=max(max(st.M2, st.M3) for st in stations),
        max_shear=max(max(st.V2, st.V3) for st in stations),
    )


def _rsa_element_results(structure: Any, forces: Dict[Any, List[float]]) -> Dict[str, ElementResult]:
    """Convert sign-less CQC/SRSS element end forces into ElementResults at STANDARD_STATIONS."""
    element_results: Dict[str, ElementResult] = {}
    for eid in structure.elements:
        f_el = forces.get(eid)
        if f_el is None:
            f_el = forces.get(int(eid) if str(eid).isdigit() else eid)
        if f_el is None:
            continue
        element_results[str(eid)] = _envelope_element_result(eid, np.asarray(f_el, dtype=float), signed=False)
    return element_results


def _rsa_support_reactions(
    model: Any,
    structure: Any,
    rsa_out: Dict[str, Any],
    combination: str,
    factor: float = 1.0,
) -> Dict[Any, NodeResult]:
    """Sign-less CQC/SRSS support reactions from modal nodal equilibrium.

    For each mode m, the modal nodal force vector at each supported node is the sum of the modal global member
    end forces K_e,global u_e,m of all elements framing into that node (restricted to restrained or spring-supported
    DOFs), and the modal reaction components across modes are combined by CQC/SRSS/ABS using the modal frequencies
    and damping ratios.
    """
    supports_by_str = {str(s.node_id): s for s in model.supports}
    if not supports_by_str or not structure.nodes:
        return {}
    rsa = rsa_out["rsa"]
    modal_u = np.asarray(rsa_out["modal_displacements"], dtype=float)
    n_modes = modal_u.shape[1]
    full_u = np.zeros((structure.neq, n_modes), dtype=float)
    full_u[structure.free_dofs, :] = modal_u

    sample_node = next(iter(structure.nodes.values()))
    n_dof = len(getattr(sample_node, "dofs", ()))
    dof_names = ("ux", "uy", "uz", "rx", "ry", "rz") if n_dof == 6 else ("ux", "uy", "rz")
    masks: Dict[Any, np.ndarray] = {}
    modal_reactions: Dict[Any, np.ndarray] = {}
    for s in model.supports:
        mask = np.array(
            [bool(getattr(s, d, False)) or float(getattr(s, f"k{d}", 0.0) or 0.0) > 0.0 for d in dof_names],
            dtype=bool,
        )
        masks[s.node_id] = mask
        modal_reactions[s.node_id] = np.zeros((n_dof, n_modes), dtype=float)

    for el in structure.elements.values():
        ni_str, nj_str = str(el.node_i.id), str(el.node_j.id)
        if ni_str not in supports_by_str and nj_str not in supports_by_str:
            continue
        dofs = list(el.node_i.dofs) + list(el.node_j.dofs)
        f_g = np.asarray(el.global_stiffness(), dtype=float) @ full_u[dofs, :]
        if ni_str in supports_by_str:
            nid = supports_by_str[ni_str].node_id
            modal_reactions[nid] += f_g[:n_dof, :]
        if nj_str in supports_by_str:
            nid = supports_by_str[nj_str].node_id
            modal_reactions[nid] += f_g[n_dof:, :]

    out: Dict[Any, NodeResult] = {}
    for nid, r_modal in modal_reactions.items():
        g = np.where(masks[nid], rsa.combine_modal_responses(r_modal, method=combination) * factor, 0.0)
        reaction = (NodeReaction(fx=float(g[0]), fy=float(g[1]), fz=float(g[2]),
                                 mx=float(g[3]), my=float(g[4]), mz=float(g[5]))
                    if n_dof == 6 else NodeReaction(fx=float(g[0]), fy=float(g[1]), mz=float(g[2])))
        out[nid] = NodeResult(node_id=nid, reaction=reaction)
    return out


def _rsa_design_envelope(
    project: Any,
    analysis_case_id: str,
    directional: Dict[str, AnalysisResult],
    *,
    sds: Optional[float],
    solver: str,
    include_shear_deformation: bool,
    cracked_sections: bool = False,
    second_order: bool = False,
) -> Tuple[Optional[str], Dict[str, AnalysisResult], Any]:
    """Envelope ASCE 7-22 LRFD combinations of the gravity cases with sign-less RSA responses.

    The RSA results act as the seismic cases E_X / E_Y of ``generate_asce7_combinations`` (with
    Ev = 0.2 SDS D when ``sds`` is known and 1.0 E_X + 0.3 E_Y orthogonal combinations for
    bi-directional runs). Each combination takes the governing branch of G +/- E per force component.
    With ``second_order`` the spectral (sway) responses are magnified by the story delta_s from an
    auxiliary lateral pattern on the cracked model, and column moments by the member delta
    (ACI 318-25 §6.6.4).

    Returns (design envelope case ID or None without a dead load case, gravity case results,
    SecondOrderReport or None).
    """
    from struct_core import LoadCaseFactor
    from struct_core.analysis_result import envelope_results, superpose_results
    from struct_load import generate_asce7_combinations

    from .analysis import run_analysis
    from .frame_stability import (
        SecondOrderReport,
        aci_cracked_stiffness_modifiers,
        magnify_member_curvature,
        magnify_sway_cases,
        pattern_flexibility,
    )

    dead_ids = []
    live_id = None
    for lc in project.model.load_cases:
        cid = lc.id
        l_type = (getattr(lc, "load_type", None) or "").lower()
        cid_str = str(cid).lower()
        if l_type in ("dead", "sdl", "superimposed_dead", "self_weight", "gravity") or any(tag in cid_str for tag in ("dead", "sdl", "gravity")):
            dead_ids.append(cid)
        elif l_type in ("live", "occupancy") or ("live" in cid_str and "roof" not in cid_str):
            if live_id is None:
                live_id = cid
    if not dead_ids:
        return None, {}, None

    gravity: Dict[str, AnalysisResult] = {}
    for cid in dead_ids + ([live_id] if live_id else []):
        res = run_analysis(
            project, analysis_case_id=cid, load_case_id=cid, solver=solver,
            include_shear_deformation=include_shear_deformation, cracked_sections=cracked_sections,
        )
        if not res.ok:
            raise RuntimeError(f"gravity case {cid!r} failed: {res.error.message if res.error else res}")
        gravity[cid] = project.analysis_results[cid]

    basis = getattr(project, "seismic_basis", None)
    rho_val = float(getattr(basis, "redundancy_factor", 1.0) or 1.0)
    e_ids = [r.analysis_case_id for r in directional.values()]
    case_results: Dict[str, AnalysisResult] = {**gravity, **{r.analysis_case_id: r for r in directional.values()}}
    combos = generate_asce7_combinations(
        dead_case_id=dead_ids[0],
        live_case_id=live_id,
        seismic_case_id=e_ids[0],
        seismic_y_case_id=e_ids[1] if len(e_ids) > 1 else None,
        orthogonal_combinations=len(e_ids) > 1,
        sds=sds,
        rho=rho_val,
    )
    for comb in combos:  # every dead-type case takes the dead-load factor
        f_dead = next((f.factor for f in comb.factors if f.load_case_id == dead_ids[0]), 0.0)
        comb.factors.extend(LoadCaseFactor(load_case_id=cid, factor=f_dead) for cid in dead_ids[1:])

    def _superpose_all(results: Dict[str, AnalysisResult]) -> List[AnalysisResult]:
        out = []
        for comb in combos:
            factors = {f.load_case_id: float(f.factor) for f in comb.factors}
            out.append(superpose_results(results, factors, output_case_id=comb.id, spectral_case_ids=e_ids))
        return out

    comb_results = _superpose_all(case_results)
    report = None
    if second_order:
        report = SecondOrderReport()
        directions = {r.analysis_case_id: d for d, r in directional.items()}
        modifiers = aci_cracked_stiffness_modifiers(project.model)
        flexibility = pattern_flexibility(
            project,
            lambda case: _solve_static_case(project, case, solver, modifiers, include_shear_deformation),
            sorted(set(directions.values())),
        )
        magnified = magnify_sway_cases(
            project, case_results, combos, directions, flexibility, report, sign_less_cases=e_ids
        )
        first_by_id = {r.analysis_case_id: r for r in comb_results}
        comb_results = magnify_member_curvature(
            project, _superpose_all(magnified), first_by_id, case_results, combos, dead_ids, report
        )
    design = envelope_results(comb_results, output_case_id=f"{analysis_case_id}_design")
    design.stiffness_basis = "cracked" if cracked_sections else "gross"
    design.code_checks = list(report.checks) if report else []
    for res in gravity.values():
        res.stiffness_basis = design.stiffness_basis
    project.analysis_results[design.analysis_case_id] = design
    return design.analysis_case_id, gravity, report


def _solve_static_case(
    project: Any, load_case: Any, solver: str, modifiers: Optional[Dict[Any, float]], include_shear_deformation: bool
) -> AnalysisResult:
    """Linear static analysis of an auxiliary load case on a copy of the project model."""
    trial = project.model_copy(deep=False)
    trial.model = project.model.model_copy(update={
        "load_cases": [load_case],
        "load_combinations": [],
        "mass_source": None,
        "mass_sources": [],
    })
    if solver == "fem2d":
        from fem2d.adapters import model_from_core, result_to_core

        struct = model_from_core(
            trial, load_case_id=load_case.id, include_self_weight=False,
            include_shear_deformation=include_shear_deformation, flexural_stiffness_modifiers=modifiers,
        )
        struct.solve()
        return result_to_core(struct, analysis_case_id=load_case.id)
    from fem3d.adapters import model_from_core, result_to_core

    struct = model_from_core(
        trial, load_case_id=load_case.id, include_self_weight=False,
        include_shear_deformation=include_shear_deformation, flexural_stiffness_modifiers=modifiers,
    )
    struct.solve()
    res = result_to_core(struct)
    res.analysis_case_id = load_case.id
    return res


def _elf_base_shear_scaling(
    project: Any,
    rsa_by_direction: Dict[str, Dict[str, Any]],
    base_shear_by_direction: Dict[str, float],
    *,
    sds: float,
    sd1: float,
    tl: Optional[float],
    r_factor: float,
    importance_factor: float,
    system_type: str,
    s1: Optional[float],
    vertical_axis: str,
    site_class: str = "D",
    risk_category: str = "II",
) -> Dict[str, Any]:
    """Force scale factors per ASCE 7-22 §12.9.1.4.1 for each analysed direction.

    V = Cs W (Section 12.8) with the period T of the mode that dominates the direction, capped at
    Cu Ta; where the modal base shear Vt < V, the factor is V / Vt (otherwise 1.0).
    """
    from struct_core import parse_units
    from struct_load import (
        SeismicParameters,
        approximate_fundamental_period,
        calculate_mass_source,
        calculate_seismic_response_coefficient,
        get_cu_coefficient,
    )

    units = getattr(project, "units", None) or getattr(getattr(project, "metadata", None), "units", None)
    units = parse_units(units or "kN, m")
    weight = calculate_mass_source(project).total_seismic_weight
    elevations = [float(getattr(n, vertical_axis)) for n in project.model.nodes]
    hn = max(elevations) - min(elevations)
    ta = approximate_fundamental_period(hn, system_type=system_type, units=units)
    cu = get_cu_coefficient(sd1)
    params = SeismicParameters(
        sds=sds, sd1=sd1, site_class=site_class, risk_category=risk_category, tl=tl,
        r_factor=r_factor, importance_factor=importance_factor,
        s1=s1, system_type=system_type,
    )

    per_direction: Dict[str, Any] = {}
    factors: Dict[str, float] = {}
    for d_dir, res in rsa_by_direction.items():
        mmpr = np.asarray(res["mass_participation_ratio"], dtype=float)
        t_modal = float(np.asarray(res["periods"], dtype=float)[int(np.argmax(mmpr))])
        t_used = min(t_modal, cu * ta)
        cs = calculate_seismic_response_coefficient(params, t_used)
        v_elf = cs * weight
        vt = base_shear_by_direction[d_dir]
        factor = v_elf / vt if 0.0 < vt < v_elf else 1.0
        factors[d_dir] = factor
        per_direction[d_dir] = {
            "modal_period": t_modal, "period_used": t_used, "cs": cs,
            "elf_base_shear": v_elf, "modal_base_shear": vt, "scale_factor": factor,
        }
    return {
        "applied": True,
        "reason": "ASCE 7-22 Section 12.9.1.4.1: forces scaled by V/Vt where Vt < V.",
        "seismic_weight": weight, "ta": ta, "cu": cu,
        "directions": per_direction, "factors": factors,
    }


def _ground_motion_failure(tool_name: str, ground_motion: Any, exc: Exception) -> ToolResult:
    from structdyn import GroundMotion

    return ToolResult.failure(
        tool_name,
        AgentError(
            error_type="unsupported_ground_motion",
            message=f"Failed to load ground motion {ground_motion!r}: {exc}",
            possible_causes=[
                f"Built-in events: {list(GroundMotion.available_events().keys())}",
                "Or pass a record file path (.AT2 / .csv), a GroundMotion instance, or an acceleration array in g.",
            ],
        ),
    )


@stamps_results(with_combinations=False)
def run_time_history(
    project: Any,
    *,
    analysis_case_id: str,
    ground_motion: Any,
    component: Optional[str] = None,
    direction: str = "x",
    dt: Optional[float] = None,
    record_scale: float = 1.0,
    scale_factor: Optional[float] = None,
    damping_ratio: float = 0.05,
    n_modes_damping: int = 3,
    method: str = "newmark_beta",
    solver: str = "auto",
    mass_source: Optional[Any] = None,
    include_shear_deformation: bool = False,
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
        Ground motion to analyze (required). Can be:
        - Record file path: PEER NGA ``.AT2`` or two-column ``.csv`` (time [s], acceleration [g])
        - Preset string name (e.g. ``"elcentro"``, ``"loma_prieta"``, ``"san_fernando"``)
        - Composite string with component (e.g. ``"loma_prieta:hor2"``)
        - A ``GroundMotion`` instance
        - A 1D list or numpy array of acceleration values in g (requires ``dt``)
    component
        Component to load when ``ground_motion`` is a string (``"hor1"``, ``"hor2"``, ``"up"``).
        If None, automatically inferred from ``direction``:
        - direction="x" -> "hor1"
        - direction="y" -> "hor2"
        - direction="z" -> "up"
    direction
        Excitation direction (``"x"``, ``"y"``, or ``"z"``).
    dt
        Time step in seconds for array input. Records carry their own dt.
    record_scale
        Dimensionless amplitude multiplier (default: 1.0; g to model-unit conversion is automatic).
    scale_factor
        Deprecated: use record_scale instead.
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

    if scale_factor is not None:
        return ToolResult.failure(
            "run_time_history",
            AgentError(
                error_type="invalid_input",
                message="scale_factor was replaced by record_scale (dimensionless); the g to model-unit conversion is automatic.",
                possible_causes=["Pass record_scale instead of scale_factor"],
            ),
        )

    applied_defaults = []

    # 1. Resolve Scale Factor
    u = getattr(project, "units", None) or getattr(getattr(project, "metadata", None), "units", None)
    if u is None:
        raise ValueError("Units unknown: pass units= or use a Project")
    from struct_core.units import parse_units
    parsed_u = parse_units(u) if isinstance(u, (str, dict)) else u
    sf = parsed_u.gravity_acceleration * float(record_scale)

    # 2. Resolve Ground Motion
    try:
        gm = _resolve_ground_motion(
            ground_motion, component=component, direction=direction, scale_factor=sf, dt=dt
        )
    except Exception as exc:
        return _ground_motion_failure("run_time_history", ground_motion, exc)

    # 2. Resolve Solver & Build Structure
    ensure_seismic_mass_source(project, mass_source)

    resolved_solver = solver.lower().strip()
    if resolved_solver == "auto":
        resolved_solver = _detect_solver(project)

    try:
        if resolved_solver == "fem2d":
            from fem2d.adapters import model_from_core as fem2d_model_from_core
            structure = fem2d_model_from_core(
                project,
                include_shear_deformation=include_shear_deformation,
                mass_source=mass_source,
            )
        else:
            from fem3d.adapters import model_from_core as fem3d_model_from_core
            structure = fem3d_model_from_core(
                project,
                include_shear_deformation=include_shear_deformation,
                mass_source=mass_source,
            )
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

    from structdyn import MDF
    # Lumped mass, consistent with run_analysis (modal) and run_response_spectrum
    mdf = MDF.from_fem(structure, zeta=damping_ratio, n_modes=n_modes_damping, lumped=True)

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

    # Base shear history from global equilibrium of elastic restoring forces: Vb(t) = -iota^T K_ff u(t)
    comp_idx = {"x": 0, "y": 1, "z": 2}.get(str(direction).lower(), 0)
    free_index = {int(g): i for i, g in enumerate(structure.free_dofs)}
    iota = np.zeros(ndof)
    for node in structure.nodes.values():
        dofs = getattr(node, "dofs", [])
        if comp_idx < len(dofs) and int(dofs[comp_idx]) in free_index:
            iota[free_index[int(dofs[comp_idx])]] = 1.0
    base_shear_history = u_free @ (np.asarray(mdf.K) @ iota)
    peak_base_shear = float(np.max(np.abs(base_shear_history)))

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
            er = _envelope_element_result(eid, f_local, signed=True)
            max_structure_moment = max(max_structure_moment, er.max_moment)
            max_structure_shear = max(max_structure_shear, er.max_shear)
            max_structure_axial = max(max_structure_axial, er.max_axial)
            element_results[str(eid)] = er

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
        result_type="time_history",
        carries_factored_actions=False,
    )

    project.analysis_results[analysis_case_id] = analysis_result
    if "dynamic_envelope" not in project.analysis_results:
        project.analysis_results["dynamic_envelope"] = analysis_result
    project.active_analysis_case = analysis_case_id

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
        "peak_base_shear": peak_base_shear,
        "roof_node": roof_node_id,
        "fundamental_period": float(periods[0]) if len(periods) > 0 else 0.0,
        "max_axial_envelope": max_structure_axial,
        "max_shear_envelope": max_structure_shear,
        "max_moment_envelope": max_structure_moment,
        "n_element_results": len(element_results),
        **({"defaults_applied": [d.model_dump() for d in applied_defaults]} if applied_defaults else {}),
    }

    return ToolResult.success(
        "run_time_history",
        data=out_data,
        next_suggested_stage="ANALYSIS_COMPLETE",
    )


@stamps_results(with_combinations=False)
def run_response_spectrum(
    project: Any,
    *,
    analysis_case_id: str,
    params: Optional[Any] = None,
    sds: Optional[float] = None,
    sd1: Optional[float] = None,
    tl: Optional[float] = None,
    site_class: Optional[str] = None,
    spectrum_curve: Optional[Any] = None,
    ground_motion: Any = None,
    component: Optional[str] = None,
    Sa: Optional[Union[List[float], np.ndarray]] = None,
    Sd: Optional[Union[List[float], np.ndarray]] = None,
    direction: str = "x",
    combination: str = "CQC",
    r_factor: Optional[float] = None,
    importance_factor: Optional[float] = None,
    system_type: Optional[str] = None,
    s1: Optional[float] = None,
    n_modes: Optional[int] = None,
    damping_ratio: float = 0.05,
    spectrum_scale: float = 1.0,
    scale_factor: Optional[float] = None,
    solver: str = "auto",
    mass_source: Optional[Any] = None,
    include_shear_deformation: bool = False,
    envelope_to_design: bool = True,
    risk_category: Optional[str] = None,
    cracked_sections: bool = False,
    second_order: bool = False,
) -> ToolResult:
    """
    Run linear dynamic Response Spectrum Analysis (RSA) using structdyn bridged with fem2d/fem3d.

    ``cracked_sections`` uses the ACI 318-25 Table 6.6.3.1.1 stiffness (also for the periods, ASCE
    7-22 §12.7.3); ``second_order`` adds ACI 318-25 §6.6.4 moment magnification to the design
    envelope (requires ``cracked_sections``). When ``r_factor`` and ``system_type`` are given the
    seismic design basis is recorded on the project, and with ``cracked_sections`` the story drift
    (§12.12.1, from CQC/SRSS-combined modal story drifts) and stability coefficient (§12.8.7, with
    the combined modal story shears) are checked.

    With an ASCE 7 design spectrum (``sds``/``sd1`` or ``params``) and ``r_factor``, the modal forces
    are reduced by R/Ie and then scaled per ASCE 7-22 §12.9.1.4.1: where the modal base shear Vt in a
    direction is less than the ELF base shear V (Section 12.8, T capped at Cu*Ta), the forces in that
    direction are multiplied by V/Vt. ``system_type`` is then required for Ta (Table 12.8-2).

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
        ASCE 7 long-period transition period TL (seconds); required only when T1 >= 4.0 s.
    r_factor
        ASCE 7 seismic response modification coefficient R (e.g. 5.0 for Intermediate Moment Frames,
        8.0 for Special Moment Frames). Elastic spectrum forces are divided by R / Ie.
    importance_factor
        Seismic importance factor Ie (Section 11.5.1); by default that of ``risk_category`` (Table 1.5-2).
    system_type
        Seismic force-resisting system for the approximate period Ta (Table 12.8-2), e.g. 'rc_smf',
        'rc_imf', 'rc_omf', 'steel_smf', 'steel_ebf', 'steel_brbf', 'all_other'. Required for the
        §12.9.1.4 scaling of design-spectrum RSA.
    s1
        Mapped MCE_R spectral acceleration S1 (g) for the Cs lower bound of Eq. (12.8-7).
    direction
        Excitation direction: 'both' analyses X and Y separately and combines them in the design envelope
        with the 100%-30% rule (ASCE 7-22 §12.5.3); 'x', 'y', or 'z' gives a one-direction envelope. Defaults to 'x'.
    spectrum_curve
        User-supplied continuous response spectrum curve Sa(T).
    ground_motion
        Optional ground motion to generate response spectrum dynamically. Can be:
        - Record file path: PEER NGA ``.AT2`` or two-column ``.csv`` (time [s], acceleration [g])
        - Preset string name (e.g. ``"elcentro"``, ``"loma_prieta"``, ``"san_fernando"``)
        - Composite string with component (e.g. ``"loma_prieta:hor2"``)
        - A ``GroundMotion`` instance
        - A 1D list or numpy array of acceleration values in g (dt = 0.02 s)
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
    combination
        Modal combination rule: 'CQC' (default), 'SRSS', or 'ABS'.
    n_modes
        Number of modes to consider.
    damping_ratio
        Modal damping ratio (default: 0.05).
    spectrum_scale
        Dimensionless multiplier applied to the response spectrum (default: 1.0).
        Acceleration unit conversions from g to model units are automatic.
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
    if second_order and not cracked_sections:
        return ToolResult.failure(
            "run_response_spectrum",
            AgentError(
                error_type="invalid_input",
                message="second_order=True requires cracked_sections=True (ACI 318-25 §6.6.4.4.3 and Table 6.6.3.1.1).",
            ),
        )
    # Design-level forces (r_factor) belong to a seismic force-resisting system, which sets Ta, Cd, the drift
    # limits, and the ACI 318 Chapter 18 frame type
    if r_factor is not None and system_type is None:
        return ToolResult.failure(
            "run_response_spectrum",
            AgentError(
                error_type="missing_input",
                message="system_type is required with r_factor (e.g. 'rc_smf', 'rc_imf'): it sets Ta for the "
                        "Section 12.9.1.4 scaling, Cd for the drift checks, and the ACI 318 Chapter 18 frame type.",
            ),
        )
    if system_type is not None:
        from struct_load import normalize_system_type, seismic_system_coefficients

        try:
            system_type = normalize_system_type(system_type)
        except ValueError as exc:
            return ToolResult.failure("run_response_spectrum", AgentError(error_type="invalid_input", message=str(exc)))
        if r_factor is None:
            r_factor, _, _ = seismic_system_coefficients(system_type)

    # 1. Validate spectrum inputs and record documented defaults
    from .. import defaults

    applied_defaults = []
    if (
        params is None
        and sds is None
        and sd1 is None
        and spectrum_curve is None
        and Sa is None
        and Sd is None
        and ground_motion is None
    ):
        return ToolResult.failure(
            "run_response_spectrum",
            AgentError(
                error_type="missing_input",
                message="Response spectrum analysis requires design spectral accelerations (sds and sd1), "
                        "a spectrum_curve, Sa/Sd arrays, or a ground_motion.",
                possible_causes=["Pass sds and sd1 (in g) for an ASCE 7 design spectrum."],
            ),
        )
    # One spectrum per analysis: the design spectrum would otherwise replace the record's without notice
    # (qwen B06 passed the Loma Prieta record with sds/sd1 it chose, and got a design-spectrum RSA)
    if ground_motion is not None and (params is not None or sds is not None or sd1 is not None):
        return ToolResult.failure(
            "run_response_spectrum",
            AgentError(
                error_type="invalid_input",
                message="Both a ground_motion and an ASCE 7 design spectrum (sds/sd1) were given; an analysis "
                        "uses one spectrum.",
                possible_causes=["Pass ground_motion alone for the record's response spectrum, or sds and sd1 "
                                 "alone for the ASCE 7 design spectrum."],
            ),
        )
    if params is None and (sds is None) != (sd1 is None):
        return ToolResult.failure(
            "run_response_spectrum",
            AgentError(
                error_type="missing_input",
                message="Both sds and sd1 are required for an ASCE 7 design spectrum.",
                possible_causes=["Provide both sds and sd1 (in g)."],
            ),
        )
    is_design_spectrum = params is not None or (sds is not None and sd1 is not None)
    if risk_category is None:
        risk_category = defaults.RISK_CATEGORY.value
        if is_design_spectrum or r_factor is not None or system_type is not None:
            applied_defaults.append(defaults.RISK_CATEGORY.record())
    if site_class is None:
        site_class = defaults.SITE_CLASS.value
        if params is None and sds is not None and sd1 is not None:
            applied_defaults.append(defaults.SITE_CLASS.record())
    if importance_factor is None:
        from struct_load import seismic_importance_factor
        importance_factor = seismic_importance_factor(risk_category)

    # 2. Resolve scale factor (accounting for R factor reduction if supplied)
    if scale_factor is not None:
        return ToolResult.failure(
            "run_response_spectrum",
            AgentError(
                error_type="invalid_input",
                message="scale_factor was replaced by spectrum_scale (dimensionless); the g to model-unit conversion is automatic.",
                possible_causes=["Pass spectrum_scale instead of scale_factor"],
            ),
        )

    u = getattr(project, "units", None) or getattr(getattr(project, "metadata", None), "units", None)
    if u is None:
        raise ValueError("Units unknown: pass units= or use a Project")
    from struct_core.units import parse_units
    parsed_u = parse_units(u) if isinstance(u, (str, dict)) else u
    sf = parsed_u.gravity_acceleration * float(spectrum_scale)

    if r_factor is not None and float(r_factor) > 0:
        sf = sf * float(importance_factor) / float(r_factor)

    resolved_gm = None
    if ground_motion is not None:
        try:
            resolved_gm = _resolve_ground_motion(
                ground_motion, component=component, direction=direction, scale_factor=sf
            )
        except Exception as exc:
            return _ground_motion_failure("run_response_spectrum", ground_motion, exc)

    # 3. Resolve solver and build structure
    ensure_seismic_mass_source(project, mass_source)

    resolved_solver = solver.lower().strip()
    if resolved_solver == "auto":
        resolved_solver = _detect_solver(project)

    from .frame_stability import aci_cracked_stiffness_modifiers

    modifiers = aci_cracked_stiffness_modifiers(project.model) if cracked_sections else None
    try:
        if resolved_solver == "fem2d":
            from fem2d.adapters import model_from_core as fem2d_model_from_core

            structure = fem2d_model_from_core(
                project,
                include_shear_deformation=include_shear_deformation,
                mass_source=mass_source,
                flexural_stiffness_modifiers=modifiers,
            )
        else:
            from fem3d.adapters import model_from_core as fem3d_model_from_core

            structure = fem3d_model_from_core(
                project,
                include_shear_deformation=include_shear_deformation,
                mass_source=mass_source,
                flexural_stiffness_modifiers=modifiers,
            )
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

    from structdyn import MDF

    mdf = MDF.from_fem(structure, zeta=damping_ratio, lumped=True)
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
        rsa_params = params
        if rsa_params is None and sds is not None and sd1 is not None:
            from struct_load import SeismicParameters
            rsa_params = SeismicParameters(
                sds=float(sds),
                sd1=float(sd1),
                site_class=site_class,
                risk_category=risk_category,
                system_type=system_type or "rc_smf",
                r_factor=float(r_factor) if r_factor is not None else None,
                importance_factor=float(importance_factor) if importance_factor is not None else None,
                tl=tl,
                damping_ratio=float(damping_ratio) if damping_ratio is not None else 0.05,
            )

        dir_clean = str(direction or "x").lower().strip()
        if dir_clean in ("both", "xy", "bidirectional"):
            dirs_to_run = ["x", "y"]
        else:
            dirs_to_run = [dir_clean]

        rsa_res = None
        rsa_res_by_dir: Dict[str, Dict[str, Any]] = {}
        story_response: Dict[str, Dict[int, Tuple[float, float, float]]] = {}
        dir_forces: Dict[str, Dict[Any, List[float]]] = {}
        base_shear_by_direction: Dict[str, float] = {}
        node_disps: Dict[Any, List[float]] = {}

        for d_dir in dirs_to_run:
            sub_res = structure.response_spectrum_analysis(
                params=rsa_params,
                spectrum_curve=spectrum_curve,
                gm=resolved_gm,
                Sa=Sa,
                Sd=Sd,
                n_modes=n_modes,
                ji=damping_ratio,
                direction=d_dir,
                scale_factor=sf,
                combination=combination,
            )
            rsa_res = sub_res
            rsa_res_by_dir[d_dir] = sub_res
            base_shear_by_direction[d_dir] = float(sub_res["combined_base_shear"][0])
            if cracked_sections and d_dir in ("x", "y"):
                story_response[d_dir] = _rsa_story_response(project, structure, sub_res, combination, d_dir)
            dir_forces[d_dir] = {
                eid: [float(abs(val)) for val in f_el]
                for eid, f_el in sub_res.get("combined_element_forces", {}).items()
            }

            for nid, node in structure.nodes.items():
                dofs = getattr(node, "dofs", [])
                if len(dofs) in (3, 6):
                    d_vec = structure.disp[dofs]
                    if nid not in node_disps:
                        node_disps[nid] = [float(d_vec[k]) for k in range(len(dofs))]
                    else:
                        node_disps[nid] = [float(np.hypot(node_disps[nid][k], d_vec[k])) for k in range(len(dofs))]
    except Exception as exc:
        return ToolResult.failure(
            "run_response_spectrum",
            AgentError(
                error_type="rsa_execution_failed",
                message=f"Response spectrum analysis failed: {exc}",
            ),
        )

    # 4b. ASCE 7-22 §12.9.1.4.1: scale design-spectrum forces up to the ELF base shear
    elf_scaling: Dict[str, Any] = {
        "applied": False,
        "reason": "Not a design-spectrum RSA with an R factor (Section 12.9.1.4 does not apply).",
    }
    if rsa_params is not None and r_factor is not None:
        try:
            elf_scaling = _elf_base_shear_scaling(
                project, rsa_res_by_dir, base_shear_by_direction,
                sds=float(rsa_params.sds), sd1=float(rsa_params.sd1), tl=rsa_params.tl,
                r_factor=float(r_factor), importance_factor=float(importance_factor),
                system_type=system_type, s1=s1, vertical_axis="y" if resolved_solver == "fem2d" else "z",
                site_class=getattr(rsa_params, "site_class", site_class),
                risk_category=getattr(rsa_params, "risk_category", risk_category),
            )
        except Exception as exc:
            return ToolResult.failure(
                "run_response_spectrum",
                AgentError(error_type="elf_scaling_failed", message=f"Section 12.9.1.4 scaling failed: {exc}"),
            )
        for d_dir, factor in elf_scaling["factors"].items():
            dir_forces[d_dir] = {eid: [v * factor for v in f_el] for eid, f_el in dir_forces[d_dir].items()}
            base_shear_by_direction[d_dir] *= factor

    seismic_basis_summary: Dict[str, Any] = {"recorded": False, "reason": "r_factor and system_type are required."}
    if r_factor is not None and system_type is not None:
        from .seismic_basis import record_seismic_basis

        seismic_basis_summary = record_seismic_basis(
            project,
            system_type=system_type,
            r_factor=float(r_factor),
            importance_factor=float(importance_factor),
            risk_category=risk_category,
            sds=float(rsa_params.sds) if rsa_params is not None else None,
            sd1=float(rsa_params.sd1) if rsa_params is not None else None,
            s1=s1,
            procedure="rsa",
            lateral_case_ids=[f"{analysis_case_id}_E{d.upper()}" for d in dirs_to_run],
            directions=list(dirs_to_run),
        )

    # 5. Extract Nodal Displacements
    node_results: Dict[Any, NodeResult] = {}
    max_lateral_disp = 0.0

    for nid, node in structure.nodes.items():
        dofs = getattr(node, "dofs", [])
        d = node_disps.get(nid)
        if d is not None and len(dofs) == 6:
            nd = NodeDisplacement(
                ux=float(d[0]), uy=float(d[1]), uz=float(d[2]),
                rx=float(d[3]), ry=float(d[4]), rz=float(d[5]),
            )
            lat_disp = float(np.hypot(d[0], d[1]))
        elif d is not None and len(dofs) == 3:
            nd = NodeDisplacement(
                ux=float(d[0]), uy=float(d[1]), rz=float(d[2]),
            )
            lat_disp = float(abs(d[0]))
        else:
            nd = NodeDisplacement()
            lat_disp = 0.0

        max_lateral_disp = max(max_lateral_disp, lat_disp)
        node_results[str(nid)] = NodeResult(node_id=nid, displacement=nd)

    # 6. Extract Element Force Envelopes (all directions) for inspection
    comb_forces: Dict[Any, List[float]] = {}
    for forces in dir_forces.values():
        for eid, f_el in forces.items():
            prev = comb_forces.get(eid)
            comb_forces[eid] = f_el if prev is None else [max(a, b) for a, b in zip(prev, f_el)]
    element_results = _rsa_element_results(structure, comb_forces)
    max_structure_axial = max((er.max_axial for er in element_results.values()), default=0.0)
    max_structure_shear = max((er.max_shear for er in element_results.values()), default=0.0)
    max_structure_moment = max((er.max_moment for er in element_results.values()), default=0.0)

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
        stiffness_basis="cracked" if cracked_sections else "gross",
    )

    project.analysis_results[analysis_case_id] = analysis_result
    project.active_analysis_case = analysis_case_id

    # 9. Seismic design envelope: ASCE 7 combinations of gravity cases with the sign-less RSA response
    # The sign-less response of each direction (the seismic cases E_X / E_Y of the recorded basis), with support
    # reactions for the service-level footing check
    directional = {
        d: AnalysisResult(
            analysis_case_id=f"{analysis_case_id}_E{d.upper()}",
            element_results=_rsa_element_results(structure, forces),
            node_results=_rsa_support_reactions(
                project.model,
                structure,
                rsa_res_by_dir[d],
                combination,
                factor=float(elf_scaling.get("factors", {}).get(d, 1.0)),
            ),
        )
        for d, forces in dir_forces.items()
    }
    for res in directional.values():
        project.analysis_results[res.analysis_case_id] = res

    design_case_id = None
    gravity_results: Dict[str, AnalysisResult] = {}
    report = None
    if envelope_to_design:
        try:
            design_case_id, gravity_results, report = _rsa_design_envelope(
                project,
                analysis_case_id,
                directional,
                sds=float(rsa_params.sds) if rsa_params is not None else None,
                solver=resolved_solver,
                include_shear_deformation=include_shear_deformation,
                cracked_sections=cracked_sections,
                second_order=second_order,
            )
        except Exception as exc:
            return ToolResult.failure(
                "run_response_spectrum",
                AgentError(
                    error_type="design_envelope_failed",
                    message=f"Failed to combine RSA response with gravity load cases: {exc}",
                ),
            )
        if design_case_id is not None:
            project.active_analysis_case = design_case_id
        project.analysis_results["dynamic_envelope"] = project.analysis_results[design_case_id or analysis_case_id]

    # 10. ASCE 7-22 story drift and stability (cracked stiffness, recorded seismic basis)
    if cracked_sections and seismic_basis_summary.get("recorded") and story_response:
        if not gravity_results:
            return ToolResult.failure(
                "run_response_spectrum",
                AgentError(
                    error_type="missing_input",
                    message="Story drift / stability checks need the dead (and live) load cases for Px (ASCE 7-22 §12.8.7).",
                ),
            )
        try:
            drift_checks = _rsa_drift_checks(
                project, story_response, gravity_results, rsa_res_by_dir,
                r_factor=float(r_factor), importance_factor=float(importance_factor), s1=s1,
            )
        except Exception as exc:
            return ToolResult.failure(
                "run_response_spectrum",
                AgentError(error_type="drift_check_failure", message=f"Story drift / stability evaluation failed: {exc}"),
            )
        analysis_result.code_checks = drift_checks
        if design_case_id is not None:
            project.analysis_results[design_case_id].code_checks.extend(drift_checks)

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

    combined_base_shear = max(base_shear_by_direction.values())
    mmpr = rsa_res["mass_participation_ratio"]

    out_data = {
        "analysis_case_id": analysis_case_id,
        "design_case_id": design_case_id,
        "solver_used": resolved_solver,
        "direction": direction,
        "combination_method": combination,
        "combined_base_shear": combined_base_shear,
        "base_shear_by_direction": base_shear_by_direction,
        "elf_scaling": elf_scaling,
        "seismic_basis": seismic_basis_summary,
        "stiffness_basis": "cracked" if cracked_sections else "gross",
        "fundamental_period": float(periods[0]) if len(periods) > 0 else 0.0,
        "cumulative_mass_participation": float(np.sum(mmpr)),
        "peak_lateral_displacement": max_lateral_disp,
        "max_axial_envelope": max_structure_axial,
        "max_shear_envelope": max_structure_shear,
        "max_moment_envelope": max_structure_moment,
        "n_element_results": len(element_results),
        "defaults_applied": [d.model_dump() for d in applied_defaults],
    }
    if report is not None:
        out_data["second_order"] = report.as_dict()["stories"]
    frame_checks = project.analysis_results[design_case_id].code_checks if design_case_id else analysis_result.code_checks
    if frame_checks:
        worst = max(frame_checks, key=lambda c: c.dcr)
        out_data["frame_checks"] = {
            "n_checks": len(frame_checks),
            "all_passed": all(c.passed for c in frame_checks),
            "governing": {"clause": worst.clause, "description": worst.description, "dcr": round(worst.dcr, 4)},
        }

    return ToolResult.success(
        "run_response_spectrum",
        data=out_data,
        next_suggested_stage="ANALYSIS_COMPLETE",
    )


def _rsa_story_response(
    project: Any, structure: Any, rsa_out: Dict[str, Any], combination: str, direction: str
) -> Dict[int, Tuple[float, float, float]]:
    """
    Per story: (largest combined column-line drift, combined mean story drift, combined story shear)
    of one spectral direction. Modal story drifts and modal story shears (from the modal
    equivalent static forces) are combined with the modal combination rule of the analysis.
    """
    from struct_core import frame_stories, node_elevation

    from .frame_stability import _column_ends

    rsa = rsa_out["rsa"]
    modal_u = np.asarray(rsa_out["modal_displacements"], dtype=float)  # (n_free, n_modes)
    modal_f = np.asarray(rsa.modal_equivalent_forces(), dtype=float)
    free_index = {int(g): k for k, g in enumerate(structure.free_dofs)}
    comp = 0 if direction == "x" else 1
    model = project.model
    nodes = {n.id: n for n in model.nodes}

    def modal_component(values: np.ndarray, nid: Any) -> np.ndarray:
        node = structure.nodes.get(nid)
        if node is None:
            node = structure.nodes[str(nid)]
        k = free_index.get(int(node.dofs[comp]))
        return np.zeros(values.shape[1]) if k is None else values[k, :]

    out: Dict[int, Tuple[float, float, float]] = {}
    for story in frame_stories(model):
        line_drifts = []
        for eid in story.column_ids:
            bottom, top = _column_ends(model, eid)
            line_drifts.append(modal_component(modal_u, top) - modal_component(modal_u, bottom))
        combined = [float(rsa.combine_modal_responses(d, method=combination)) for d in line_drifts]
        mean_modal = np.mean(line_drifts, axis=0)
        cut = 0.5 * (story.bottom + story.top)
        shear_modal = np.zeros(modal_f.shape[1])
        for nid, node in nodes.items():
            if node_elevation(model, node) >= cut - 1e-9:
                shear_modal += modal_component(modal_f, nid)
        out[story.index] = (
            max(combined),
            float(rsa.combine_modal_responses(mean_modal, method=combination)),
            float(rsa.combine_modal_responses(shear_modal, method=combination)),
        )
    return out


def _rsa_drift_checks(
    project: Any,
    story_response: Dict[str, Dict[int, Tuple[float, float, float]]],
    gravity_results: Dict[str, AnalysisResult],
    rsa_by_direction: Dict[str, Dict[str, Any]],
    *,
    r_factor: float,
    importance_factor: float,
    s1: Optional[float],
) -> List[Any]:
    """
    ASCE 7-22 drift (§12.12.1) and stability (§12.8.7) checks from the modal story response.
    Drifts are scaled by Cs W / Vt per §12.9.1.4.2 where S1 >= 0.6 and Vt < Cs W (Cs of Eq. 12.8-7).
    """
    from struct_load import calculate_mass_source

    from .analysis import _dead_and_live_ids
    from .frame_stability import drift_and_stability_checks, gravity_story_load

    drifts, shears = {}, {}
    weight = calculate_mass_source(project).total_seismic_weight
    for direction, stories in story_response.items():
        factor = 1.0
        if s1 is not None and s1 >= 0.6:
            vt = float(rsa_by_direction[direction]["combined_base_shear"][0])
            cs_min = 0.5 * s1 / (r_factor / importance_factor)
            if 0.0 < vt < cs_min * weight:
                factor = cs_min * weight / vt
        for s_idx, (drift_max, drift_mean, shear) in stories.items():
            drifts[(s_idx, direction)] = (drift_max * factor, drift_mean)
            shears[(s_idx, direction)] = shear
    dead_ids, live_ids = _dead_and_live_ids(project.model)
    px = gravity_story_load(project, gravity_results, dead_ids, [c for c in live_ids if c in gravity_results])
    return drift_and_stability_checks(project, drifts, shears, px)


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
        summary.update(modal_participation(project, result.modal_result.modes))

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
