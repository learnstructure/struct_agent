"""
Analysis tools.

These wrap ``fem2d`` through the existing struct_core adapter. The agent
does not assemble matrices, apply boundary conditions, or solve systems
itself; the fem2d library owns all of that.
"""

from __future__ import annotations

from typing import Any, Optional

from ..validation.analysis import validate_analysis
from .base import AgentError, ToolResult
from .model_state import stamps_results


def _detect_solver(project: Any, analysis_type: str = "static") -> str:
    """Detect whether a model requires fem2d or fem3d."""
    model = getattr(project, "model", None)
    if model is None:
        return "fem2d"

    dim = getattr(model, "dimension", None)
    if dim:
        if str(dim).upper() == "3D":
            return "fem3d"
        if str(dim).upper() == "2D":
            return "fem2d"

    # Check node 3D coordinates (Z != 0)
    for n in getattr(model, "nodes", []):
        if abs(getattr(n, "z", 0.0)) > 1e-4:
            return "fem3d"

    # Check 3D boundary conditions (uz, rx, ry)
    for s in getattr(model, "supports", []):
        if any(bool(getattr(s, dof, False)) for dof in ("uz", "rx", "ry")):
            return "fem3d"

    v_axis = getattr(model, "vertical_axis", None)
    if v_axis and str(v_axis).upper() == "Z":
        return "fem3d"

    return "fem2d"


@stamps_results(with_combinations=True)
def run_analysis(
    project: Any,
    *,
    analysis_case_id: str,
    analysis_type: str = "static",
    solver: str = "auto",
    load_case_id: Optional[str] = None,
    num_modes: int = 6,
    include_self_weight: bool = True,
    include_shear_deformation: Optional[bool] = None,
    mass_source: Optional[Any] = None,
    cracked_sections: bool = False,
    second_order: bool = False,
) -> ToolResult:
    """
    Run finite element analysis through the struct_core adapters (fem2d or fem3d).

    Parameters
    ----------
    project
        A ``struct_core.Project`` containing the model.
    analysis_case_id
        A unique identifier used to store the resulting ``AnalysisResult``
        on the project.
    analysis_type
        ``"static"`` (linear static) or ``"modal"`` (eigenvalue modal analysis).
    solver
        ``"auto"`` (detect based on geometry & analysis type), ``"fem2d"``, or ``"fem3d"``.
    load_case_id
        Which load case or combination from the model to apply.
        Defaults to ``"envelope"`` if combinations exist, else the first load case.
    num_modes
        Number of modes to compute when ``analysis_type="modal"`` (default: 6).
    include_self_weight
        Whether to automatically compute and include element self-weight for any
        load case where ``include_self_weight=True`` (default: True).
    include_shear_deformation
        Whether to consider Timoshenko shear deformation in frame/beam elements.
    mass_source
        Optional MassSource definition or ID to use for dynamic mass matrix assembly.
    cracked_sections
        Use the ACI 318-25 Table 6.6.3.1.1 member stiffness (0.70 Ig columns, 0.35 Ig beams), as
        required for factored-load analysis of concrete frames and for seismic drift (ASCE 7-22
        §12.7.3). With a seismic design basis recorded by ``apply_lateral_loads`` (ELF), the story
        drift (§12.12.1) and stability coefficient (§12.8.7) checks are added to the envelope.
    second_order
        Include slenderness effects by moment magnification (ACI 318-25 §6.6.4): sway moments of the
        lateral cases are magnified by the story delta_s and column moments by the member delta.
        Requires ``cracked_sections=True``. The limit checks are stored on the envelope.
    """
    if project is None or getattr(project, "model", None) is None:
        return ToolResult.failure(
            "run_analysis",
            AgentError(
                error_type="missing_input",
                message="No Project supplied to run_analysis.",
                possible_causes=["Project was never created/loaded"],
            ),
        )

    # Resolve solver
    resolved_solver = solver.lower().strip()
    if resolved_solver == "auto":
        resolved_solver = _detect_solver(project, analysis_type=analysis_type)

    if resolved_solver not in ("fem2d", "fem3d"):
        return ToolResult.failure(
            "run_analysis",
            AgentError(
                error_type="unsupported_solver",
                message=f"Solver {solver!r} is not recognized. Must be 'auto', 'fem2d', or 'fem3d'.",
                possible_causes=["Typo in solver name"],
            ),
        )
    if second_order and not cracked_sections:
        return ToolResult.failure(
            "run_analysis",
            AgentError(
                error_type="invalid_input",
                message="second_order=True requires cracked_sections=True (ACI 318-25 §6.6.4.4.3 and Table 6.6.3.1.1).",
            ),
        )

    from .frame_stability import aci_cracked_stiffness_modifiers

    modifiers = aci_cracked_stiffness_modifiers(project.model) if cracked_sections else None
    stiffness_basis = "cracked" if cracked_sections else "gross"
    second_order_report = None
    stability_checks: list = []

    # -------------------------------------------------------------
    # Static Analysis with Linear Superposition & Enveloping
    # -------------------------------------------------------------
    if analysis_type == "static":
        from struct_core.analysis_result import envelope_results, superpose_results

        def _solve_single_case(proj: Any, solver_name: str, target_lc_id: Optional[str]) -> Any:
            if solver_name == "fem2d":
                from fem2d.adapters import model_from_core, result_to_core
                struct = model_from_core(
                    proj,
                    load_case_id=target_lc_id,
                    include_self_weight=include_self_weight,
                    include_shear_deformation=include_shear_deformation,
                    flexural_stiffness_modifiers=modifiers,
                )
                struct.solve()
                res = result_to_core(struct, analysis_case_id=target_lc_id or "LCASE1")
            else:
                from fem3d.adapters import model_from_core as fem3d_model_from_core
                from fem3d.adapters import result_to_core as fem3d_result_to_core
                struct = fem3d_model_from_core(
                    proj,
                    load_case_id=target_lc_id,
                    include_self_weight=include_self_weight,
                    include_shear_deformation=include_shear_deformation,
                    flexural_stiffness_modifiers=modifiers,
                )
                struct.solve()
                res = fem3d_result_to_core(struct)
                res.analysis_case_id = target_lc_id or "LCASE1"
            res.stiffness_basis = stiffness_basis
            return res

        load_cases = list(getattr(project.model, "load_cases", []))
        load_combinations = list(getattr(project.model, "load_combinations", []))
        comb_map = {c.id: c for c in load_combinations}
        if not load_cases:
            return ToolResult.failure(
                "run_analysis",
                AgentError(
                    error_type="missing_input",
                    message="The model has no load cases, so a static analysis has nothing to solve. Apply loads "
                            "(apply_gravity_loads, apply_lateral_loads) and generate load combinations first; "
                            "creating a model again starts it without loads.",
                ),
            )
        is_single_comb = load_case_id in comb_map

        if is_single_comb:
            comb = comb_map[load_case_id]
            factors = {f.load_case_id: float(f.factor) for f in getattr(comb, "factors", [])}
            case_results = {}
            for lc_id in factors:
                try:
                    res_lc = _solve_single_case(project, resolved_solver, lc_id)
                    case_results[lc_id] = res_lc
                    project.analysis_results[lc_id] = res_lc
                except Exception as exc:
                    return ToolResult.failure(
                        "run_analysis",
                        AgentError(
                            error_type="solver_failure",
                            message=f"{resolved_solver} solver failed on case {lc_id!r}: {exc}",
                            possible_causes=["Zero-stiffness mechanism or unstable support"],
                        ),
                    )
            comb_res = superpose_results(case_results, factors, output_case_id=comb.id)
            comb_res.stiffness_basis = stiffness_basis
            comb_res.result_type = "combination"
            is_strength = getattr(comb, "kind", "strength") == "strength" and getattr(comb, "method", "LRFD") == "LRFD"
            comb_res.carries_factored_actions = is_strength
            project.analysis_results[comb.id] = comb_res
            project.analysis_results[analysis_case_id] = comb_res
            analysis_result = comb_res

        elif (
            load_case_id
            and str(load_case_id).upper() not in ("ALL", "TOTAL", "ENVELOPE")
        ):
            try:
                analysis_result = _solve_single_case(project, resolved_solver, load_case_id)
                analysis_result.result_type = "case"
                analysis_result.carries_factored_actions = False
                project.analysis_results[load_case_id] = analysis_result
                project.analysis_results[analysis_case_id] = analysis_result
            except Exception as exc:
                return ToolResult.failure(
                    "run_analysis",
                    AgentError(
                        error_type="solver_failure",
                        message=f"{resolved_solver} solver failed on case {load_case_id!r}: {exc}",
                        possible_causes=["Zero-stiffness mechanism or invalid loading"],
                    ),
                )
        else:
            # 1. Solve each primary load case independently
            case_results = {}
            if load_cases:
                for lc in load_cases:
                    try:
                        res_lc = _solve_single_case(project, resolved_solver, lc.id)
                        case_results[lc.id] = res_lc
                        project.analysis_results[lc.id] = res_lc
                    except Exception as exc:
                        return ToolResult.failure(
                            "run_analysis",
                            AgentError(
                                error_type="solver_failure",
                                message=f"{resolved_solver} solver failed on case {lc.id!r}: {exc}",
                                possible_causes=["Zero-stiffness mechanism or unstable support"],
                            ),
                        )
            else:
                try:
                    res_def = _solve_single_case(project, resolved_solver, "LCASE1")
                    case_results["LCASE1"] = res_def
                    project.analysis_results["LCASE1"] = res_def
                except Exception as exc:
                    return ToolResult.failure(
                        "run_analysis",
                        AgentError(
                            error_type="solver_failure",
                            message=f"{resolved_solver} solver failed on default load case: {exc}",
                            possible_causes=["Zero-stiffness mechanism or unstable support"],
                        ),
                    )

            # 2. Linear Superposition for Combinations
            if load_combinations:
                def _superpose_all(results: dict) -> list:
                    out = []
                    for comb in load_combinations:
                        factors = {f.load_case_id: float(f.factor) for f in getattr(comb, "factors", [])}
                        comb_res = superpose_results(results, factors, output_case_id=comb.id)
                        comb_res.stiffness_basis = stiffness_basis
                        comb_res.result_type = "combination"
                        is_strength = getattr(comb, "kind", "strength") == "strength" and getattr(comb, "method", "LRFD") == "LRFD"
                        comb_res.carries_factored_actions = is_strength
                        out.append(comb_res)
                    return out

                comb_results = _superpose_all(case_results)
                if second_order:
                    try:
                        comb_results, second_order_report = _apply_second_order(
                            project, case_results, load_combinations, comb_results, _superpose_all
                        )
                    except Exception as exc:
                        return ToolResult.failure(
                            "run_analysis",
                            AgentError(
                                error_type="second_order_failure",
                                message=f"Slenderness (ACI 318-25 §6.6.4) evaluation failed: {exc}",
                            ),
                        )
                for comb_res in comb_results:
                    project.analysis_results[comb_res.analysis_case_id] = comb_res
                if cracked_sections and getattr(project, "seismic_basis", None) is not None \
                        and project.seismic_basis.procedure == "elf":
                    try:
                        stability_checks = _elf_drift_checks(project, case_results)
                    except Exception as exc:
                        return ToolResult.failure(
                            "run_analysis",
                            AgentError(
                                error_type="drift_check_failure",
                                message=f"Story drift / stability (ASCE 7-22 §12.12, §12.8.7) evaluation failed: {exc}",
                            ),
                        )

                # 3. Compile Governing Envelope across combinations
                envelope_res = envelope_results(comb_results, output_case_id=analysis_case_id)
                envelope_res.result_type = "envelope"
                envelope_res.carries_factored_actions = True
                envelope_res.stiffness_basis = stiffness_basis
                envelope_res.code_checks = (second_order_report.checks if second_order_report else []) + stability_checks
                project.analysis_results[analysis_case_id] = envelope_res
                project.analysis_results["envelope"] = envelope_res  # the latest static envelope
                analysis_result = envelope_res
            else:
                # No combinations defined: compute governing envelope across the solved cases
                if len(case_results) == 1:
                    raw_res = list(case_results.values())[0].model_copy(deep=True)
                    raw_res.analysis_case_id = analysis_case_id
                    raw_res.result_type = "case"
                else:
                    raw_res = envelope_results(list(case_results.values()), output_case_id=analysis_case_id)
                    raw_res.result_type = "envelope"
                raw_res.stiffness_basis = stiffness_basis
                raw_res.carries_factored_actions = False
                project.analysis_results[analysis_case_id] = raw_res
                analysis_result = raw_res

        project.active_analysis_case = analysis_case_id

    elif analysis_type == "modal":
        try:
            import numpy as np
            from struct_core import AnalysisResult, ModalResult, ModeShape, NodeDisplacement
            from structdyn import MDF
        except ImportError as exc:
            return ToolResult.failure(
                "run_analysis",
                AgentError(error_type="dependency_error", message=f"struct_core/structdyn import error: {exc}"),
            )

        from .dynamics import ensure_seismic_mass_source

        ensure_seismic_mass_source(project, mass_source)
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

            structure.number_dofs()
            structure.apply_boundary_conditions()
            # Lumped mass: the convention shared by all dynamic tools and commercial CAE (SAP2000/ETABS)
            mdf = MDF.from_fem(structure, lumped=True)
            if np.max(np.abs(mdf.M)) < 1e-12:
                return ToolResult.failure(
                    "run_analysis",
                    AgentError(
                        error_type="solver_failure",
                        message=f"{resolved_solver} mass matrix is zero. Assign material density (rho) or nodal lumped mass.",
                        possible_causes=[
                            "Mass matrix is singular (no mass or density assigned to elements/nodes)",
                            "Unconstrained rigid-body mechanism",
                        ],
                    ),
                )

            n_calc = min(num_modes, mdf.ndof)
            omega, phi = mdf.modal.modal_analysis(n_modes=n_calc)
        except Exception as exc:
            return ToolResult.failure(
                "run_analysis",
                AgentError(
                    error_type="solver_failure",
                    message=f"{resolved_solver} modal analysis failed: {exc}",
                    possible_causes=[
                        "Mass matrix is singular (no mass or density assigned to elements/nodes)",
                        "Unconstrained rigid-body mechanism",
                    ],
                ),
            )

        # Build ModeShape schemas & compute modal mass participation ratios
        freqs = omega / (2.0 * np.pi)
        periods = np.where(freqs > 1e-10, 1.0 / freqs, 0.0)

        r_x = np.zeros(structure.neq)
        r_y = np.zeros(structure.neq)
        for nid, node in structure.nodes.items():
            dofs = getattr(node, "dofs", [])
            if len(dofs) >= 2:
                r_x[dofs[0]] = 1.0
                r_y[dofs[1]] = 1.0

        rx_free = r_x[structure.free_dofs]
        ry_free = r_y[structure.free_dofs]
        M_mat = mdf.M

        total_mx = float(rx_free.T @ M_mat @ rx_free) if len(rx_free) > 0 else 0.0
        total_my = float(ry_free.T @ M_mat @ ry_free) if len(ry_free) > 0 else 0.0

        mode_shapes = []
        for i in range(len(omega)):
            nodal_disps = {}
            part_x = 0.0
            part_y = 0.0
            if phi is not None and phi.shape[1] > i:
                phi_i = phi[:, i]
                Mn = float(phi_i.T @ M_mat @ phi_i)
                if Mn > 1e-12:
                    Lx = float(phi_i.T @ M_mat @ rx_free)
                    Ly = float(phi_i.T @ M_mat @ ry_free)
                    if total_mx > 1e-12:
                        part_x = min(1.0, max(0.0, (Lx**2) / (Mn * total_mx)))
                    if total_my > 1e-12:
                        part_y = min(1.0, max(0.0, (Ly**2) / (Mn * total_my)))

                full_mode = np.zeros(structure.neq)
                full_mode[structure.free_dofs] = phi[:, i]
                max_amp = np.max(np.abs(full_mode))
                if max_amp > 1e-12:
                    full_mode /= max_amp

                for nid, node in structure.nodes.items():
                    dofs = getattr(node, "dofs", [])
                    if len(dofs) == 6:
                        nodal_disps[nid] = NodeDisplacement(
                            ux=float(full_mode[dofs[0]]),
                            uy=float(full_mode[dofs[1]]),
                            uz=float(full_mode[dofs[2]]),
                            rx=float(full_mode[dofs[3]]),
                            ry=float(full_mode[dofs[4]]),
                            rz=float(full_mode[dofs[5]]),
                        )
                    elif len(dofs) == 3:
                        nodal_disps[nid] = NodeDisplacement(
                            ux=float(full_mode[dofs[0]]),
                            uy=float(full_mode[dofs[1]]),
                            rz=float(full_mode[dofs[2]]),
                        )

            mode_shapes.append(
                ModeShape(
                    mode_number=i + 1,
                    period=max(float(periods[i]), 1e-6),
                    frequency=max(float(freqs[i]), 1e-6),
                    circular_frequency=max(float(omega[i]), 1e-6),
                    mass_participation_x=round(part_x, 4),
                    mass_participation_y=round(part_y, 4),
                    displacements=nodal_disps,
                )
            )

        analysis_result = AnalysisResult(
            analysis_case_id=analysis_case_id,
            modal_result=ModalResult(modes=mode_shapes),
            stiffness_basis=stiffness_basis,
        )
    elif analysis_type in ("response_spectrum", "rsa"):
        from .dynamics import run_response_spectrum
        return run_response_spectrum(
            project,
            analysis_case_id=analysis_case_id,
            solver=resolved_solver,
            n_modes=num_modes,
        )
    else:
        return ToolResult.failure(
            "run_analysis",
            AgentError(
                error_type="unsupported_analysis_type",
                message=f"Analysis type {analysis_type!r} is not supported. Use 'static', 'modal', or 'response_spectrum'.",
                possible_causes=["Unsupported analysis type"],
            ),
        )

    project.analysis_results[analysis_case_id] = analysis_result

    validation = validate_analysis(analysis_result)
    if not validation.ok:
        return ToolResult.failure(
            "run_analysis",
            AgentError(
                error_type="empty_result",
                message="Analysis completed but produced no usable results.",
                possible_causes=validation.error.possible_causes if validation.error else [],
            ),
            data={"analysis_case_id": analysis_case_id},
        )

    out_data: dict[str, Any] = {
        "analysis_case_id": analysis_case_id,
        "solver_used": resolved_solver,
        "analysis_type": analysis_type,
        "n_element_results": len(analysis_result.element_results),
        "stiffness_basis": stiffness_basis,
    }
    if second_order_report is not None:
        out_data["second_order"] = second_order_report.as_dict()["stories"]
    if analysis_result.code_checks:
        worst = max(analysis_result.code_checks, key=lambda c: c.dcr)
        out_data["frame_checks"] = {
            "n_checks": len(analysis_result.code_checks),
            "all_passed": all(c.passed for c in analysis_result.code_checks),
            "governing": {"clause": worst.clause, "description": worst.description, "dcr": round(worst.dcr, 4)},
        }
    if analysis_result.modal_result and analysis_result.modal_result.modes:
        out_data["n_modes"] = len(analysis_result.modal_result.modes)
        out_data["fundamental_period"] = round(analysis_result.modal_result.modes[0].period, 4)
        out_data["periods"] = [round(m.period, 4) for m in analysis_result.modal_result.modes]
        out_data.update(modal_participation(project, analysis_result.modal_result.modes))

    if getattr(analysis_result, "node_reactions", None):
        out_data["reactions"] = _reaction_summary(project, analysis_result, details=False)

    return ToolResult.success(
        "run_analysis",
        data=out_data,
        next_suggested_stage="ANALYSIS_COMPLETE",
    )


def modal_participation(project: Any, modes: list) -> dict[str, Any]:
    """Modal mass participation ratios per mode and cumulative, in X (and in Y for a 3D model, where Y is the
    other horizontal direction; in 2D, Y is vertical and is left out)."""
    is_3d = any(abs(getattr(n, "z", 0.0) or 0.0) > 1e-9 for n in getattr(project.model, "nodes", []))
    out: dict[str, Any] = {}
    for axis in ("x", "y") if is_3d else ("x",):
        ratios = [float(getattr(m, f"mass_participation_{axis}", 0.0) or 0.0) for m in modes]
        out[f"mass_participation_{axis}"] = [round(r, 4) for r in ratios]
        out[f"cumulative_mass_participation_{axis}"] = [round(sum(ratios[: i + 1]), 4) for i in range(len(ratios))]
    return out


def _dead_and_live_ids(model: Any) -> tuple:
    """Dead-type case IDs and live case IDs using canonical load classification."""
    from struct_core import classify_load_case
    dead = []
    live = []
    for lc in model.load_cases:
        cat = classify_load_case(lc, allow_unknown=True)
        if cat in ("dead", "super_dead"):
            dead.append(lc.id)
        elif cat in ("live", "storage_live"):
            live.append(lc.id)
    return dead, live


def _apply_second_order(project: Any, case_results: dict, combinations: list, first_order: list, superpose_all) -> tuple:
    """ACI 318-25 §6.6.4 moment magnification of the combination results (static analysis)."""
    from .frame_stability import (
        SecondOrderReport,
        classify_lateral_cases,
        magnify_member_curvature,
        magnify_sway_cases,
        static_flexibility,
    )

    report = SecondOrderReport()
    lateral = classify_lateral_cases(project.model, case_results)
    magnified_cases = case_results
    if lateral:
        flexibility = static_flexibility(project, case_results, lateral)
        magnified_cases = magnify_sway_cases(project, case_results, combinations, lateral, flexibility, report)
    second = superpose_all(magnified_cases) if lateral else first_order
    dead_ids, _ = _dead_and_live_ids(project.model)
    first_by_id = {r.analysis_case_id: r for r in first_order}
    combos = magnify_member_curvature(project, second, first_by_id, case_results, combinations, dead_ids, report)
    return combos, report


def _elf_drift_checks(project: Any, case_results: dict) -> list:
    """ASCE 7-22 story drift and stability checks from the ELF lateral cases of the seismic basis."""
    from struct_core import frame_stories

    from .frame_stability import drift_and_stability_checks, gravity_story_load, story_drifts, story_shear

    basis = project.seismic_basis
    model = project.model
    cases = {lc.id: lc for lc in model.load_cases}
    drifts, shears = {}, {}
    for cid, direction in zip(basis.lateral_case_ids, basis.directions):
        if cid not in case_results:
            raise ValueError(f"Seismic lateral case {cid!r} was not analysed.")
        for story in frame_stories(model):
            d = story_drifts(model, case_results[cid], story, direction)
            drifts[(story.index, direction)] = (max(d, key=abs), sum(d) / len(d))
            shears[(story.index, direction)] = story_shear(model, cases[cid], story, direction)
    dead_ids, live_ids = _dead_and_live_ids(model)
    px = gravity_story_load(project, case_results, dead_ids, live_ids)
    return drift_and_stability_checks(project, drifts, shears, px)


def _reaction_summary(project: Any, result: Any, details: bool) -> dict:
    """Support reactions of a result in the model force unit: equilibrium totals of a load case or combination,
    and (``details``) the largest vertical reaction at a single support.

    An envelope keeps each support's reaction from its governing combination, so the sum over the supports is no
    equilibrium total (nemotron reported the envelope sum as a maximum support reaction and as a base shear).
    """
    from struct_core import parse_units

    reacs = result.node_reactions
    fz_sum = sum(r.fz for r in reacs)
    vertical = "fz" if abs(fz_sum) > 1e-4 or any(abs(r.fz) > 1e-4 for r in reacs) else "fy"
    out: dict = {"force_unit": str(parse_units(getattr(project.metadata, "units", None) or "kN, m").force)}
    n_combos = max((len(er.combination_forces or {}) for er in result.element_results.values()), default=0)
    if n_combos > 1:
        out["note"] = (f"Envelope of {n_combos} load combinations: each support's reaction comes from its governing "
                       "combination, so sums over the supports are not totals. For total reactions and base shear, "
                       "inspect a load case or combination by its id.")
    else:
        out["total_vertical_reaction"] = round(abs(sum(getattr(r, vertical) for r in reacs)), 3)
        out["base_shear_x"] = round(abs(sum(r.fx for r in reacs)), 3)
        if vertical == "fz":
            out["base_shear_y"] = round(abs(sum(r.fy for r in reacs)), 3)
    if details and reacs:
        top = max(reacs, key=lambda r: abs(getattr(r, vertical)))
        out["max_support_vertical_reaction"] = {"node_id": top.node_id, "value": round(abs(getattr(top, vertical)), 3)}
    return out


def inspect_analysis_results(project: Any, analysis_case_id: str) -> ToolResult:
    """Return a structured summary of an analysis case for the agent."""
    if project is None or not getattr(project, "analysis_results", None):
        return ToolResult.failure(
            "inspect_analysis_results",
            AgentError(
                error_type="missing_result",
                message="No analysis results attached to the project.",
                possible_causes=["run_analysis was never called"],
            ),
        )

    result = project.analysis_results.get(analysis_case_id)
    if result is None:
        available = list(project.analysis_results.keys())
        return ToolResult.failure(
            "inspect_analysis_results",
            AgentError(
                error_type="unknown_case",
                message=f"Analysis case {analysis_case_id!r} not found.",
                possible_causes=[f"Available cases: {available}"],
            ),
        )

    summary: dict[str, Any] = {"analysis_case_id": analysis_case_id, "elements": {}}
    for elem_id, elem_res in result.element_results.items():
        forces = getattr(elem_res, "forces", None) or getattr(elem_res, "section_forces", []) or []
        summary["elements"][str(elem_id)] = {
            "n_stations": len(forces),
            "max_axial": float(getattr(elem_res, "max_axial", 0.0) or 0.0),
            "max_moment": float(getattr(elem_res, "max_moment", 0.0) or 0.0),
            "max_shear": float(getattr(elem_res, "max_shear", 0.0) or 0.0),
        }

    if getattr(result, "modal_result", None) and getattr(result.modal_result, "modes", None):
        summary["modal"] = {
            "n_modes": len(result.modal_result.modes),
            "periods": [round(m.period, 4) for m in result.modal_result.modes],
            "frequencies": [round(m.frequency, 4) for m in result.modal_result.modes],
            **modal_participation(project, result.modal_result.modes),
        }

    if getattr(result, "node_reactions", None):
        summary["reactions_summary"] = _reaction_summary(project, result, details=True)

    if getattr(result, "time_history_result", None):
        th = result.time_history_result
        summary["time_history"] = {
            "dt": float(th.dt),
            "n_steps": len(th.time_steps),
            "duration": float(th.time_steps[-1].time) if th.time_steps else 0.0,
        }

    return ToolResult.success("inspect_analysis_results", data=summary)