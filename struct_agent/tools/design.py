"""
Design tools.

These wrap ``aci318`` through its struct_core adapter. The agent does not
perform ACI 318 calculations itself.
"""

from __future__ import annotations

from typing import Any, Optional

from .. import defaults
from ..validation.design import validate_design
from .base import AgentError, ToolResult
from .model_state import is_current
from .units_input import check_realistic, unit_label


def _carries_factored_actions(project: Any, result: Any) -> bool:
    """An envelope of load combinations (an ASCE 7-22 combination envelope or a response spectrum design
    envelope; each records the forces of its combinations), or the solution of one load combination."""
    if getattr(result, "result_type", None) == "time_history":
        return False
    if hasattr(result, "carries_factored_actions") and result.carries_factored_actions is not None:
        return bool(result.carries_factored_actions)
    combinations = {c.id: c for c in getattr(project.model, "load_combinations", None) or []}
    if result.analysis_case_id in combinations:
        c = combinations[result.analysis_case_id]
        if getattr(c, "kind", None) == "service" or getattr(c, "method", None) == "ASD":
            return False
        return True
    return any(
        er.combination_forces for er in result.element_results.values()
    ) and bool(combinations)


def _service_level_combinations(result: Any, project: Optional[Any] = None) -> list:
    """ASD (service-level) combinations in a result: ACI 318 strength design needs factored combinations."""
    ids = {result.analysis_case_id} | {k for er in result.element_results.values() for k in (er.combination_forces or {})}
    comb_map = {}
    if project and getattr(project, "model", None) and getattr(project.model, "load_combinations", None):
        comb_map = {c.id: c for c in project.model.load_combinations}
    asd = []
    for i in ids:
        if str(i).upper().startswith("ASD_") or str(i).upper().startswith("SERVICE"):
            asd.append(i)
        elif i in comb_map:
            c = comb_map[i]
            if getattr(c, "kind", None) == "service" or getattr(c, "method", None) == "ASD":
                asd.append(i)
    return sorted(asd)


def _service_results(project: Any) -> list:
    """Service-level results for the footing bearing check (ACI 318-25 R13.2.6.1): the ASCE 7-22 §2.4 (ASD)
    combinations superposed from the stored load-case results. A response-spectrum direction (sign-less, with its
    support reactions) enters as +-0.7|E| on each reaction component independently: one result adds it to every
    component, a second subtracts it from the vertical reaction (the least axial load with the largest moment).
    Empty when a load case of the combinations has no current stored result."""
    from struct_core import LoadCase, vertical_axis
    from struct_core.analysis_result.superposition import superpose_results
    from struct_load import LoadGenerator

    model = project.model.model_copy(deep=True)
    model.load_combinations = []
    # Keep only load cases that carry loads and have a current stored analysis result
    model.load_cases = [
        lc for lc in model.load_cases
        if (getattr(lc, "element_loads", None) or getattr(lc, "point_loads", None)
            or getattr(lc, "include_self_weight", False))
        and lc.id in project.analysis_results
        and is_current(project, project.analysis_results[lc.id])
    ]
    sources = {lc.id: project.analysis_results.get(lc.id) for lc in model.load_cases}
    spectral = {}
    basis = getattr(project, "seismic_basis", None)
    rsa_cases: list[tuple[str, str]] = []
    if basis is not None and basis.procedure == "rsa":
        rsa_cases = list(zip(basis.lateral_case_ids, basis.directions))
    else:
        for cid, res_obj in project.analysis_results.items():
            for d_tag in ("x", "y"):
                if str(cid).endswith(f"_E{d_tag.upper()}") and is_current(project, res_obj):
                    rsa_cases.append((cid, d_tag))
    if rsa_cases:
        from struct_core import classify_load_case
        # The spectrum is the seismic effect: equivalent-lateral-force cases kept in the model do not enter
        model.load_cases = [lc for lc in model.load_cases
                            if classify_load_case(lc, allow_unknown=True) != "seismic"]
        for case_id, direction in rsa_cases:
            placeholder = f"EQ_{direction.upper()}_RSA"
            model.load_cases.append(LoadCase(id=placeholder))
            sources[placeholder] = project.analysis_results.get(case_id)
            spectral[placeholder] = sources[placeholder]
    try:
        combinations = LoadGenerator(units=project.metadata.units).generate_combinations(
            model, design_method="ASD", sds=getattr(basis, "sds", None),
            rho=getattr(basis, "redundancy_factor", 1.0) or 1.0)
    except ValueError:
        return []  # no dead load case
    results = []
    for combination in combinations:
        factors = {f.load_case_id: f.factor for f in combination.factors}
        if any(sources.get(c) is None or not is_current(project, sources[c]) for c in factors):
            return []
        spectral_ids = [c for c in factors if c in spectral]
        if any(factors[c] < 0 for c in spectral_ids):
            continue
        sup = superpose_results(case_results=sources, factors=factors, output_case_id=combination.id,
                                spectral_case_ids=spectral_ids)
        results.append(sup)
        if spectral_ids:
            vert = "fz" if getattr(getattr(project, "model", None), "dimension", "") == "3D" or any(abs(n.z) > 1e-9 for n in project.model.nodes) else "fy"
            results.append(_least_axial(sup, sources, factors, spectral_ids, vert))
    return results


def _least_axial(combined: Any, sources: dict, factors: dict, spectral_ids: list, vertical: str) -> Any:
    """``combined`` (gravity +|E| on every component) with the vertical reaction G - |0.7 E| instead."""
    out = combined.model_copy(deep=True)
    out.analysis_case_id = f"{combined.analysis_case_id}_least_axial"
    for nid, nr in out.node_results.items():
        if nr.reaction is None:
            continue
        static = sum(f * getattr(_reaction(sources[c], nid), vertical, 0.0)
                     for c, f in factors.items() if c not in spectral_ids and _reaction(sources[c], nid) is not None)
        peak = sum(abs(f * getattr(_reaction(sources[c], nid), vertical, 0.0))
                   for c, f in factors.items() if c in spectral_ids and _reaction(sources[c], nid) is not None)
        setattr(nr.reaction, vertical, static - peak if static >= 0.0 else static + peak)
    return out


def _reaction(result: Any, node_id: Any) -> Any:
    nr = result.node_results.get(node_id) or result.node_results.get(str(node_id))
    return getattr(nr, "reaction", None)


def run_design(
    project: Any,
    *,
    design_case_id: str,
    design_code: str = "ACI 318-25",
    analysis_case_id: Optional[str] = None,
    allowable_bearing_pressure: Optional[float] = None,
    footing_cover: Optional[float] = None,
) -> ToolResult:
    """
    Run ACI 318 design + detailing for every element in the project model.

    Members are designed with the materials and cover of their sections. Isolated footings under the supported
    columns are sized for ``allowable_bearing_pressure`` (the allowable soil bearing pressure under service loads,
    strictly in the model pressure unit; default 150 kPa, marked preliminary) with ``footing_cover`` (model length
    unit; default 3 in) from the service-level reactions (ASCE 7-22 §2.4 combinations) and designed for strength from
    the factored ones. The defaults used are reported and recorded on the design result.

    The result is stored on the project under ``design_case_id`` and
    :func:`validate_design` is run immediately so the agent never sees
    an unvalidated design.
    """
    if project is None or getattr(project, "model", None) is None:
        return ToolResult.failure(
            "run_design",
            AgentError(
                error_type="missing_input",
                message="No Project supplied to run_design.",
            ),
        )

    # Engineering gate: don't design without analysis.
    if not project.analysis_results:
        return ToolResult.failure(
            "run_design",
            AgentError(
                error_type="workflow_error",
                message="Cannot run design before a valid analysis exists.",
                possible_causes=["run_analysis must succeed before run_design"],
            ),
            next_suggested_stage="MODEL_VALIDATED",
        )

    try:
        from aci318.adapters import design_from_core  # local import
    except Exception as exc:  # pragma: no cover - environment guard
        return ToolResult.failure(
            "run_design",
            AgentError(
                error_type="dependency_error",
                message=f"aci318 adapters are unavailable: {exc}",
                possible_causes=["aci318 is not installed in the active environment"],
            ),
        )

    # The analysis (the active one when none is named) must exist and carry factored design actions
    case_id = analysis_case_id or getattr(project, "active_analysis_case", None)
    if case_id not in project.analysis_results:
        return ToolResult.failure("run_design", AgentError(
            error_type="invalid_input",
            message=f"Analysis case '{case_id}' does not exist; available: {sorted(project.analysis_results)}."))
    res = project.analysis_results[case_id]
    # A raw response-spectrum result is the sign-less seismic response alone; its gravity combination is
    # stored as '<id>_design'
    envelope = f"{case_id}_design"
    is_raw_rsa = res.modal_result is not None and bool(res.element_results) and res.time_history_result is None
    if envelope in project.analysis_results or is_raw_rsa:
        hint = (f"design '{envelope}', which combines it with the gravity loads (ASCE 7-22 load combinations)"
                if envelope in project.analysis_results else
                "run run_response_spectrum with envelope_to_design=True after the gravity loads are applied")
        return ToolResult.failure("run_design", AgentError(
            error_type="invalid_input",
            message=f"'{case_id}' is a seismic-only response spectrum result, not a set of design actions; {hint}."))
    if not is_current(project, res):
        return ToolResult.failure("run_design", AgentError(
            error_type="invalid_input",
            message=f"'{case_id}' was computed before the latest change of the model (sections, loads, load "
                    "combinations or seismic basis), so its demands do not describe the current model; run the "
                    "analysis again and design the new result."))
    if not _carries_factored_actions(project, res):
        return ToolResult.failure("run_design", AgentError(
            error_type="invalid_input",
            message=f"'{case_id}' holds unfactored load-case results, not the factored load combinations "
                    "(ASCE 7-22 §2.3); run generate_load_combinations, then run_analysis, and design that analysis."))
    asd = _service_level_combinations(res, project)
    if asd:
        return ToolResult.failure("run_design", AgentError(
            error_type="invalid_input",
            message=f"'{case_id}' includes allowable-stress (ASD) combinations ({', '.join(asd[:3])}"
                    f"{', ...' if len(asd) > 3 else ''}); ACI 318 strength design needs the strength-level (LRFD) "
                    "combinations (ACI 318-25 §5.3, ASCE 7-22 §2.3). Run generate_load_combinations with "
                    "standard='ACI318_LRFD', then run_analysis, and design that analysis."))

    # Foundation inputs: the given values, else the defaults (recorded)
    units = project.metadata.units
    applied = []
    try:
        check_realistic("allowable_bearing_pressure", allowable_bearing_pressure, units)
        check_realistic("footing_cover", footing_cover, units)
    except ValueError as exc:
        return ToolResult.failure("run_design", AgentError(error_type="invalid_input", message=str(exc)))
    if allowable_bearing_pressure is None:
        allowable_bearing_pressure = defaults.ALLOWABLE_BEARING_PRESSURE.in_units(units)
        applied.append(defaults.ALLOWABLE_BEARING_PRESSURE.record(model_units=units))
    if footing_cover is None:
        footing_cover = defaults.FOOTING_COVER.in_units(units)
        applied.append(defaults.FOOTING_COVER.record(model_units=units))
    service = _service_results(project) if project.model.supports else []

    try:
        project.active_analysis_case = case_id
        design_res, detail_res = design_from_core(
            project,
            analysis_result=res,
            design_code=design_code,
            allowable_bearing_pressure=allowable_bearing_pressure if service else None,
            footing_cover=footing_cover if service else None,
            service_analysis_results=service or None,
        )
    except Exception as exc:
        return ToolResult.failure(
            "run_design",
            AgentError(
                error_type="design_failure",
                message=f"aci318 design_from_core raised: {exc}",
                possible_causes=[
                    "A section lacks its concrete (f'c), rebar materials or cover: set them with modify_project",
                    "Analysis result has no element internal forces",
                    "Section schema is missing required dimensions",
                ],
            ),
        )

    from .model_state import model_fingerprint
    fp = model_fingerprint(project)
    design_res.analysis_case_id = case_id
    design_res.model_fingerprint = fp
    detail_res.analysis_case_id = case_id
    detail_res.model_fingerprint = fp

    if design_res.inputs is not None:
        model_defaults = list(getattr(project, "defaults_applied", None) or [])
        foundation_defaults = applied if service else []
        seen_defaults = set()
        merged_defaults = []
        for d in model_defaults + foundation_defaults:
            key = (d.name, d.applies_to)
            if key not in seen_defaults:
                seen_defaults.add(key)
                merged_defaults.append(d)
        design_res.inputs.defaults_applied = merged_defaults
    project.design_results[design_case_id] = design_res
    project.detailing_results[design_case_id] = detail_res
    foundations = (
        {"designed": True, "allowable_bearing_pressure": allowable_bearing_pressure,
         "pressure_unit": unit_label(units, "pressure"), "footing_cover": footing_cover,
         "defaults_applied": [d.model_dump() for d in applied]}
        if service else
        {"designed": False,
         "reason": "No service-level reactions: run_analysis keeps the load-case results the bearing check needs."}
    )

    validation = validate_design(design_res, detail_res)
    if not validation.ok:
        return ToolResult.failure(
            "run_design",
            AgentError(
                error_type="design_failure",
                message="Design completed but produced no element results.",
                possible_causes=validation.error.possible_causes if validation.error else [],
            ),
            data={"design_case_id": design_case_id},
        )

    return ToolResult.success(
        "run_design",
        data={
            "design_case_id": design_case_id,
            "design_code": design_code,
            "analysis_case_id": case_id,
            **governing_frame_summary(design_res),
            "n_element_results": len(design_res.element_results),
            "n_detailing_results": len(detail_res.element_results),
            "foundations": foundations,
        },
        next_suggested_stage="DESIGN_COMPLETE",
    )


def governing_frame_summary(design_res: Any) -> dict[str, Any]:
    """Governing demand-capacity summary of the superstructure (beams and columns) and footings."""
    frame = {
        eid: r for eid, r in design_res.element_results.items()
        if getattr(r, "member_type", None) in ("beam", "column")
    }
    footings = [
        float(r.governing_dcr or 0.0) for r in design_res.element_results.values()
        if getattr(r, "member_type", None) == "footing"
    ]
    summary: dict[str, Any] = {
        "governing_dcr": None,
        "passed": False,
        "governing_member_id": None,
        "governing_member_type": None,
        "governing_limit_state": None,
        "dcr_flexure": None,
        "dcr_shear": None,
        "foundation_governing_dcr": max(footings) if footings else None,
    }
    if not frame and not footings:
        return summary

    all_elem_passed = all(getattr(r, "passed", True) for r in design_res.element_results.values())
    system_checks = list(getattr(design_res, "system_checks", None) or [])
    all_sys_passed = all(getattr(c, "passed", True) for c in system_checks)
    footing_ok = (max(footings) <= 1.0) if footings else True

    if not frame:
        max_ftg = max(footings)
        summary.update(
            governing_dcr=max_ftg,
            passed=footing_ok and all_elem_passed and all_sys_passed,
            governing_member_type="footing",
            governing_limit_state="Foundation bearing / flexure",
        )
        return summary

    gov_id, gov = max(frame.items(), key=lambda item: float(item[1].governing_dcr or 0.0))
    dcr_flex = float(gov.flexure.dcr) if gov.flexure is not None else 0.0
    dcr_shear = float(gov.shear.dcr) if gov.shear is not None else 0.0
    if gov.member_type == "column":
        states = [
            (dcr_flex, "Column axial-flexure (P-M) interaction (ACI 318 §22.4)"),
            (dcr_shear, "Column shear (ACI 318 §22.5)"),
        ]
    else:
        states = [(dcr_flex, "Beam flexure (ACI 318 §22.3)"), (dcr_shear, "Beam shear (ACI 318 §22.5)")]
    # Additional code checks (e.g. beam ductility §9.3.3.1, torsion §22.7) carry their own clause
    states += [(float(c.dcr), f"{c.description} ({c.clause})") for c in (gov.code_checks or [])]
    state = max(states, key=lambda item: item[0])[1]
    governing = float(gov.governing_dcr or 0.0)
    member_id, member_type = str(gov_id), gov.member_type
    # Frame-level checks (joints, slenderness limits, story drift, stability) may govern
    if system_checks:
        worst = max(system_checks, key=lambda c: c.dcr)
        if worst.dcr > governing:
            governing = float(worst.dcr)
            state = f"{worst.description} ({worst.clause})"
            member_id, member_type = None, "frame"
    d_units = getattr(design_res, "units", None)
    summary.update(
        governing_dcr=governing,
        passed=governing <= 1.0 and footing_ok and all_elem_passed and all_sys_passed,
        governing_member_id=member_id,
        governing_member_type=member_type,
        governing_limit_state=state,
        dcr_flexure=dcr_flex,
        dcr_shear=dcr_shear,
        n_system_checks=len(system_checks),
        units=d_units.short_label if hasattr(d_units, "short_label") else str(d_units) if d_units else None,
    )
    return summary


def inspect_design_results(project: Any, design_case_id: str) -> ToolResult:
    """Return a structured summary of a design case for the agent."""
    if project is None or not getattr(project, "design_results", None):
        return ToolResult.failure(
            "inspect_design_results",
            AgentError(
                error_type="missing_result",
                message="No design results attached to the project.",
                possible_causes=["run_design was never called"],
            ),
        )

    design_res = project.design_results.get(design_case_id)
    if design_res is None:
        available = list(project.design_results.keys())
        return ToolResult.failure(
            "inspect_design_results",
            AgentError(
                error_type="unknown_case",
                message=f"Design case {design_case_id!r} not found.",
                possible_causes=[f"Available cases: {available}"],
            ),
        )

    elements: dict[str, Any] = {}
    for elem_id, elem_des in design_res.element_results.items():
        elements[str(elem_id)] = {
            "member_type": getattr(elem_des, "member_type", None),
            "governing_dcr": float(getattr(elem_des, "governing_dcr", 0.0) or 0.0),
            "passed": bool(getattr(elem_des, "passed", False)),
        }

    return ToolResult.success(
        "inspect_design_results",
        data={
            "design_case_id": design_case_id,
            "design_code": getattr(design_res, "design_code", None),
            **governing_frame_summary(design_res),
            "elements": elements,
        },
    )


def inspect_detailing_results(
    project: Any,
    design_case_id: Optional[str] = None,
) -> ToolResult:
    """Return a structured summary of detailing results and rebar schedules for the agent."""
    if project is None or not getattr(project, "detailing_results", None):
        return ToolResult.failure(
            "inspect_detailing_results",
            AgentError(
                error_type="missing_result",
                message="No detailing results attached to the project.",
                possible_causes=["run_design was never called"],
            ),
        )

    target_case_id = design_case_id
    if target_case_id is None:
        target_case_id = next(iter(project.detailing_results.keys()))

    detail_res = project.detailing_results.get(target_case_id)
    if detail_res is None:
        available = list(project.detailing_results.keys())
        return ToolResult.failure(
            "inspect_detailing_results",
            AgentError(
                error_type="unknown_case",
                message=f"Detailing case {target_case_id!r} not found.",
                possible_causes=[f"Available cases: {available}"],
            ),
        )

    agg_schedule = getattr(detail_res, "aggregate_schedule", None)
    schedule_summary = []
    total_weight = 0.0
    if agg_schedule is not None:
        total_weight = float(getattr(agg_schedule, "total_weight", 0.0) or 0.0)
        for item in getattr(agg_schedule, "items", []):
            schedule_summary.append({
                "mark": getattr(item, "mark", ""),
                "bar_size": getattr(item, "bar_size", 0),
                "quantity": getattr(item, "quantity", 0),
                "cut_length": float(getattr(item, "cut_length", 0.0) or 0.0),
                "total_length": float(getattr(item, "total_length", 0.0) or 0.0),
                "weight": float(getattr(item, "weight", 0.0) or 0.0),
                "shape_code": getattr(item, "shape_code", ""),
                "location": getattr(item, "location", ""),
            })

    elements: dict[str, Any] = {}
    for elem_id, elem_det in getattr(detail_res, "element_results", {}).items():
        elem_items = []
        elem_sched = getattr(elem_det, "bar_schedule", None)
        if elem_sched is not None:
            for item in getattr(elem_sched, "items", []):
                elem_items.append({
                    "mark": getattr(item, "mark", ""),
                    "bar_size": getattr(item, "bar_size", 0),
                    "quantity": getattr(item, "quantity", 0),
                    "cut_length": float(getattr(item, "cut_length", 0.0) or 0.0),
                    "weight": float(getattr(item, "weight", 0.0) or 0.0),
                })
        elements[str(elem_id)] = {
            "member_type": getattr(elem_det, "member_type", None),
            "total_weight": float(getattr(elem_sched, "total_weight", 0.0) or 0.0) if elem_sched else 0.0,
            "items": elem_items,
        }

    det_units = getattr(detail_res, "units", None)
    bar_system = getattr(detail_res, "bar_system", "US")
    length_unit = getattr(det_units, "length", "in") if det_units else "in"
    weight_unit = getattr(det_units, "force", "lbf") if det_units else "lbf"

    return ToolResult.success(
        "inspect_detailing_results",
        data={
            "design_case_id": target_case_id,
            "units": det_units.short_label if hasattr(det_units, "short_label") else str(det_units) if det_units else None,
            "bar_system": bar_system,
            "length_unit": length_unit,
            "weight_unit": weight_unit,
            "total_weight": total_weight,
            "n_schedule_items": len(schedule_summary),
            "schedule": schedule_summary,
            "elements": elements,
        },
    )