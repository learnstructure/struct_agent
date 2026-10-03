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


def _detect_solver(project: Any, analysis_type: str = "static") -> str:
    """Detect whether a model requires fem2d or fem3d."""
    model = getattr(project, "model", None)
    if model is None:
        return "fem2d"

    # Check node 3D coordinates (Z != 0)
    for n in getattr(model, "nodes", []):
        if abs(getattr(n, "z", 0.0)) > 1e-4:
            return "fem3d"

    # Check 3D boundary conditions (uz, rx, ry)
    for s in getattr(model, "supports", []):
        if any(bool(getattr(s, dof, False)) for dof in ("uz", "rx", "ry")):
            return "fem3d"

    return "fem2d"


def run_analysis(
    project: Any,
    *,
    analysis_case_id: str,
    analysis_type: str = "static",
    solver: str = "auto",
    load_case_id: Optional[str] = None,
    num_modes: int = 6,
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

    # -------------------------------------------------------------
    # Static Analysis with Linear Superposition & Enveloping
    # -------------------------------------------------------------
    if analysis_type == "static":
        from struct_core.analysis_result import envelope_results, superpose_results

        def _solve_single_case(proj: Any, solver_name: str, target_lc_id: Optional[str]) -> Any:
            if solver_name == "fem2d":
                from fem2d.adapters import model_from_core, result_to_core
                struct = model_from_core(proj, load_case_id=target_lc_id)
                struct.solve()
                return result_to_core(struct, analysis_case_id=target_lc_id or "LCASE1")
            else:
                from fem3d.adapters import model_from_core as fem3d_model_from_core
                from fem3d.adapters import result_to_core as fem3d_result_to_core
                struct = fem3d_model_from_core(proj, load_case_id=target_lc_id)
                struct.solve()
                res = fem3d_result_to_core(struct)
                res.analysis_case_id = target_lc_id or "LCASE1"
                return res

        load_cases = list(getattr(project.model, "load_cases", []))
        load_combinations = list(getattr(project.model, "load_combinations", []))
        comb_map = {c.id: c for c in load_combinations}
        is_single_comb = load_case_id in comb_map

        if (
            load_case_id
            and not is_single_comb
            and str(load_case_id).upper() not in ("ALL", "TOTAL", "ENVELOPE")
        ):
            try:
                analysis_result = _solve_single_case(project, resolved_solver, load_case_id)
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
                comb_results = []
                for comb in load_combinations:
                    factors = {f.load_case_id: float(f.factor) for f in getattr(comb, "factors", [])}
                    comb_res = superpose_results(case_results, factors, output_case_id=comb.id)
                    project.analysis_results[comb.id] = comb_res
                    comb_results.append(comb_res)

                # 3. Compile Governing Envelope across combinations
                envelope_res = envelope_results(comb_results, output_case_id=analysis_case_id)
                project.analysis_results[analysis_case_id] = envelope_res
                if "envelope" not in project.analysis_results:
                    project.analysis_results["envelope"] = envelope_res
                analysis_result = envelope_res
            else:
                # No combinations defined:
                if len(case_results) > 1:
                    total_factors = {cid: 1.0 for cid in case_results}
                    total_res = superpose_results(case_results, total_factors, output_case_id="Total")
                    project.analysis_results["Total"] = total_res
                    project.analysis_results[analysis_case_id] = total_res
                    analysis_result = total_res
                else:
                    first_res = next(iter(case_results.values()))
                    project.analysis_results[analysis_case_id] = first_res
                    analysis_result = first_res


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

        try:
            if resolved_solver == "fem2d":
                from fem2d.adapters import model_from_core as fem2d_model_from_core
                structure = fem2d_model_from_core(project)
            else:
                from fem3d.adapters import model_from_core as fem3d_model_from_core
                structure = fem3d_model_from_core(project)

            structure.number_dofs()
            structure.apply_boundary_conditions()
            if hasattr(structure, "assemble_mass_matrix"):
                structure.assemble_mass_matrix()

            mdf = MDF.from_fem(structure)
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

        # Build ModeShape schemas
        freqs = omega / (2.0 * np.pi)
        periods = np.where(freqs > 1e-10, 1.0 / freqs, 0.0)
        mode_shapes = []
        for i in range(len(omega)):
            nodal_disps = {}
            if phi is not None and phi.shape[1] > i:
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
                    displacements=nodal_disps,
                )
            )

        analysis_result = AnalysisResult(
            analysis_case_id=analysis_case_id,
            modal_result=ModalResult(modes=mode_shapes),
        )
    else:
        return ToolResult.failure(
            "run_analysis",
            AgentError(
                error_type="unsupported_analysis_type",
                message=f"Analysis type {analysis_type!r} is not supported. Use 'static' or 'modal'.",
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
    }
    if analysis_result.modal_result and analysis_result.modal_result.modes:
        out_data["n_modes"] = len(analysis_result.modal_result.modes)
        out_data["fundamental_period"] = analysis_result.modal_result.modes[0].period

    return ToolResult.success(
        "run_analysis",
        data=out_data,
        next_suggested_stage="ANALYSIS_COMPLETE",
    )


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
            "periods": [m.period for m in result.modal_result.modes],
            "frequencies": [m.frequency for m in result.modal_result.modes],
        }

    if getattr(result, "time_history_result", None):
        th = result.time_history_result
        summary["time_history"] = {
            "dt": float(th.dt),
            "n_steps": len(th.time_steps),
            "duration": float(th.time_steps[-1].time) if th.time_steps else 0.0,
        }

    return ToolResult.success("inspect_analysis_results", data=summary)