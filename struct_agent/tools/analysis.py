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
    if analysis_type == "modal":
        return "fem3d"
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
    # fem2d branch
    # -------------------------------------------------------------
    if resolved_solver == "fem2d":
        if analysis_type != "static":
            return ToolResult.failure(
                "run_analysis",
                AgentError(
                    error_type="unsupported_analysis_type",
                    message=f"fem2d only supports analysis_type='static', got {analysis_type!r}.",
                    possible_causes=["Use solver='fem3d' for modal analysis"],
                ),
            )

        try:
            from fem2d.adapters import model_from_core, result_to_core, solve_combinations_and_envelope
        except Exception as exc:  # pragma: no cover - environment guard
            return ToolResult.failure(
                "run_analysis",
                AgentError(
                    error_type="dependency_error",
                    message=f"fem2d adapters are unavailable: {exc}",
                    possible_causes=["fem2d is not installed in the active environment"],
                ),
            )

        use_envelope = (
            (isinstance(load_case_id, str) and load_case_id.lower() == "envelope")
            or (load_case_id is None and bool(getattr(project.model, "load_combinations", [])))
        )

        if use_envelope:
            try:
                _, analysis_result = solve_combinations_and_envelope(
                    project, envelope_case_id=analysis_case_id
                )
            except Exception as exc:
                return ToolResult.failure(
                    "run_analysis",
                    AgentError(
                        error_type="solver_failure",
                        message=f"fem2d combination/envelope solver failed: {exc}",
                        possible_causes=["Zero-stiffness mechanism or invalid combination"],
                    ),
                )
        else:
            try:
                structure = model_from_core(project, load_case_id=load_case_id)
            except Exception as exc:
                return ToolResult.failure(
                    "run_analysis",
                    AgentError(
                        error_type="model_conversion_error",
                        message=f"struct_core -> fem2d conversion failed: {exc}",
                        possible_causes=[
                            "Section/material schema is not recognized by fem2d",
                            "Element type is not supported by the fem2d adapter",
                        ],
                    ),
                )

            try:
                structure.solve()
            except Exception as exc:
                return ToolResult.failure(
                    "run_analysis",
                    AgentError(
                        error_type="solver_failure",
                        message=f"fem2d solver failed: {exc}",
                        possible_causes=[
                            "Insufficient boundary conditions",
                            "Disconnected structural component",
                            "Structural mechanism (zero-stiffness DOF)",
                        ],
                    ),
                )

            try:
                analysis_result = result_to_core(structure, analysis_case_id=analysis_case_id)
            except Exception as exc:
                return ToolResult.failure(
                    "run_analysis",
                    AgentError(
                        error_type="result_conversion_error",
                        message=f"fem2d -> struct_core conversion failed: {exc}",
                        possible_causes=[
                            "Adapter mismatch with the installed fem2d version",
                        ],
                    ),
                )

    # -------------------------------------------------------------
    # fem3d branch
    # -------------------------------------------------------------
    else:
        try:
            from fem3d.adapters import model_from_core as fem3d_model_from_core
            from fem3d.adapters import result_to_core as fem3d_result_to_core
        except Exception as exc:  # pragma: no cover - environment guard
            return ToolResult.failure(
                "run_analysis",
                AgentError(
                    error_type="dependency_error",
                    message=f"fem3d adapters are unavailable: {exc}",
                    possible_causes=["fem3d is not installed in the active environment"],
                ),
            )

        if analysis_type == "static":
            use_envelope = (
                (isinstance(load_case_id, str) and load_case_id.lower() == "envelope")
                or (load_case_id is None and bool(getattr(project.model, "load_combinations", [])))
            )

            if use_envelope and getattr(project.model, "load_combinations", None):
                # Solve each combination and create envelope
                try:
                    comb_results = []
                    for comb in project.model.load_combinations:
                        s3d = fem3d_model_from_core(project, load_combination_id=comb.id)
                        s3d.solve()
                        res = fem3d_result_to_core(s3d)
                        comb_results.append(res)

                    if not comb_results:
                        return ToolResult.failure(
                            "run_analysis",
                            AgentError(
                                error_type="empty_result",
                                message="No combinations could be evaluated in fem3d.",
                                possible_causes=["Empty load combinations list"],
                            ),
                        )

                    # Envelope element results
                    first_res = comb_results[0]
                    enveloped_elements = {}
                    for eid in first_res.element_results:
                        all_res_e = [cr.element_results[eid] for cr in comb_results if eid in cr.element_results]
                        if not all_res_e:
                            continue
                        max_ax = max(getattr(r, "max_axial", 0.0) or 0.0 for r in all_res_e)
                        max_mom = max(getattr(r, "max_moment", 0.0) or 0.0 for r in all_res_e)
                        max_sh = max(getattr(r, "max_shear", 0.0) or 0.0 for r in all_res_e)
                        rep = all_res_e[0].model_copy(deep=True)
                        rep.max_axial = max_ax
                        rep.max_moment = max_mom
                        rep.max_shear = max_sh
                        enveloped_elements[eid] = rep

                    analysis_result = first_res.model_copy(deep=True)
                    analysis_result.analysis_case_id = analysis_case_id
                    analysis_result.element_results = enveloped_elements
                except Exception as exc:
                    return ToolResult.failure(
                        "run_analysis",
                        AgentError(
                            error_type="solver_failure",
                            message=f"fem3d combination envelope failed: {exc}",
                            possible_causes=["Zero-stiffness mechanism or invalid combination"],
                        ),
                    )
            else:
                try:
                    structure = fem3d_model_from_core(project, load_case_id=load_case_id)
                except Exception as exc:
                    return ToolResult.failure(
                        "run_analysis",
                        AgentError(
                            error_type="model_conversion_error",
                            message=f"struct_core -> fem3d conversion failed: {exc}",
                            possible_causes=[
                                "Section/material schema not recognized by fem3d",
                                "Element type is not supported by fem3d adapter",
                            ],
                        ),
                    )

                try:
                    structure.solve()
                except Exception as exc:
                    return ToolResult.failure(
                        "run_analysis",
                        AgentError(
                            error_type="solver_failure",
                            message=f"fem3d solver failed: {exc}",
                            possible_causes=[
                                "Insufficient boundary conditions in 3D (need 6 DOFs constrained or stable geometry)",
                                "Zero-stiffness rotational mechanism",
                            ],
                        ),
                    )

                try:
                    analysis_result = fem3d_result_to_core(structure)
                    analysis_result.analysis_case_id = analysis_case_id
                except Exception as exc:
                    return ToolResult.failure(
                        "run_analysis",
                        AgentError(
                            error_type="result_conversion_error",
                            message=f"fem3d -> struct_core conversion failed: {exc}",
                            possible_causes=["Result conversion failed"],
                        ),
                    )

        elif analysis_type == "modal":
            try:
                from struct_core import AnalysisResult, ModalResult, ModeShape
            except ImportError as exc:
                return ToolResult.failure(
                    "run_analysis",
                    AgentError(error_type="dependency_error", message=f"struct_core import error: {exc}"),
                )

            try:
                structure = fem3d_model_from_core(project)
                modal_data = structure.modal_analysis(num_modes=num_modes)
            except Exception as exc:
                return ToolResult.failure(
                    "run_analysis",
                    AgentError(
                        error_type="solver_failure",
                        message=f"fem3d modal analysis failed: {exc}",
                        possible_causes=[
                            "Mass matrix is singular (no mass or density assigned to elements/nodes)",
                            "Unconstrained rigid-body mechanism",
                        ],
                    ),
                )

            # Build ModeShape schemas
            freqs = modal_data.get("frequencies", [])
            periods = modal_data.get("periods", [])
            omegas = modal_data.get("omega", [])
            mode_shapes = []
            for i, (freq, per, om) in enumerate(zip(freqs, periods, omegas)):
                mode_shapes.append(
                    ModeShape(
                        mode_number=i + 1,
                        period=float(per),
                        frequency=float(freq),
                        circular_frequency=float(om),
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

    return ToolResult.success("inspect_analysis_results", data=summary)