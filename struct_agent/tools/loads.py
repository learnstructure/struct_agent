"""
Loads tools for struct_agent.

Wraps ``struct_load`` into validated, deterministic ToolResult calls.
The agent decides *what* to load (occupancy, criteria); the tools deterministically calculate *how*.
"""

from __future__ import annotations

from typing import Any, Literal, Optional, Union
from .. import defaults
from .base import AgentError, ToolResult


def apply_gravity_loads(
    project: Any,
    *,
    occupancy: str,
    superimposed_dead: float,
    tributary_width: Optional[float] = None,
    one_way_span: Optional[Literal["x", "y"]] = None,
    include_self_weight: bool = True,
    custom_live_load: Optional[float] = None,
    two_way: Optional[bool] = None,
    partition_weight: Optional[float] = None,
    pressure_unit: Optional[str] = None,
    length_unit: Optional[str] = None,
    allow_partial_floor: bool = False,
) -> ToolResult:
    """
    Apply code-compliant Dead and Live gravity loads to a Project's structural model.
    Automatically senses 2D (wy) vs 3D (wz) geometry and infers units from project metadata.
    In 3D, uses two-way floor panel yield-line distribution unless tributary_width is specified.

    Parameters
    ----------
    project : Project
        The project container holding the structural model.
    occupancy : str
        Building occupancy type per ASCE 7 / IBC (e.g. 'office', 'residential', 'classroom').
    tributary_width : float, optional
        Tributary framing width in the model length unit: required for one-way distribution (every 2D frame);
        in 3D, omitted means two-way floor panels.
    superimposed_dead : float
        Superimposed dead load pressure for finishes and MEP, in the model's pressure unit
        (kPa for 'kN, m'; ksf for 'kip, ft'; psf for 'lbf, ft'); 0 for none. Required.
    include_self_weight : bool
        Whether to calculate element self-weight from sections and density.
    custom_live_load : float, optional
        Custom live load pressure override.
    two_way : bool, optional
        Explicitly force two-way floor panel distribution in 3D frames.
    partition_weight : float, optional
        Partition weight for the effective seismic weight (ASCE 7-22 §12.7.2 item 2), model pressure
        unit. Defaults to the 10 psf (0.48 kPa) minimum for occupancies with a partition allowance;
        with ``custom_live_load`` it must be given to be included. Stored as the seismic-weight-only
        case 'PartitionWeight'.
    pressure_unit : str, optional
        Unit of superimposed_dead, custom_live_load, partition_weight (e.g. 'kPa', 'psf', 'ksf').
        If None, assumed in model pressure units.
    length_unit : str, optional
        Unit of tributary_width (e.g. 'm', 'ft', 'in'). If None, assumed in model length units.

    Pressures above 50 kPa (about 1,040 psf) are rejected: no floor load is that large, so such a value
    is a unit mistake (e.g. 105 psf entered in a ksf model).

    Returns
    -------
    ToolResult
        Success result with load cases information, or structured AgentError.
    """
    if project is None or getattr(project, "model", None) is None:
        return ToolResult.failure(
            "apply_gravity_loads",
            AgentError(
                error_type="missing_input",
                message="No valid Project or StructuralModel provided to apply_gravity_loads.",
            ),
        )

    try:
        from struct_core import conversion_factor, parse_units
        from struct_load import LoadGenerator
    except ImportError as exc:
        return ToolResult.failure(
            "apply_gravity_loads",
            AgentError(
                error_type="dependency_error",
                message=f"struct_load is unavailable: {exc}",
                possible_causes=["struct_load is not installed in the active conda environment"],
            ),
        )

    from .units_input import to_model_units, unit_label

    model_units = parse_units(getattr(project, "units", None) or getattr(project.metadata, "units", None) or defaults.UNITS.value)
    try:
        superimposed_dead = to_model_units(superimposed_dead, pressure_unit, "pressure", model_units)
        custom_live_load = to_model_units(custom_live_load, pressure_unit, "pressure", model_units)
        partition_weight = to_model_units(partition_weight, pressure_unit, "pressure", model_units)
        tributary_width = to_model_units(tributary_width, length_unit, "length", model_units)
    except ValueError as exc:
        return ToolResult.failure("apply_gravity_loads", AgentError(error_type="invalid_input", message=str(exc)))

    pressures = {"superimposed_dead": superimposed_dead, "custom_live_load": custom_live_load,
                 "partition_weight": partition_weight}
    to_kpa = conversion_factor(model_units, "kN, m", "pressure")
    for name, value in pressures.items():
        if value is not None and abs(value) * to_kpa > _MAX_FLOOR_PRESSURE_KPA:
            unit = unit_label(model_units, "pressure")
            return ToolResult.failure("apply_gravity_loads", AgentError(
                error_type="invalid_input",
                message=(f"{name} = {value:g} {unit} ({abs(value) * to_kpa:,.0f} kPa) is not a floor pressure. Values "
                         f"must strictly be given in the model pressure unit ({unit}).")))

    try:
        generator = LoadGenerator(units=model_units)
        dead_case, live_case = generator.apply_gravity_loads(
            project,
            occupancy=occupancy,
            tributary_width=tributary_width,
            one_way_span=one_way_span,
            superimposed_dead=superimposed_dead,
            include_self_weight=include_self_weight,
            custom_live_load=custom_live_load,
            two_way=two_way,
            partition_weight=partition_weight,
            allow_partial_floor=allow_partial_floor,
        )
    except Exception as exc:
        return ToolResult.failure(
            "apply_gravity_loads",
            AgentError(
                error_type="load_generation_failure",
                message=f"Failed to generate gravity loads: {exc}",
            ),
        )

    partition_case_id = next((lc.id for lc in project.model.load_cases if lc.id == "PartitionWeight"), None)
    totals = {
        "total_dead_load": _case_total(project, dead_case.id),
        "total_live_load": _case_total(project, live_case.id),
    }
    if partition_case_id is not None:
        totals["total_partition_weight"] = _case_total(project, partition_case_id)
    return ToolResult.success(
        "apply_gravity_loads",
        data={
            "occupancy": occupancy,
            "tributary_width": tributary_width,
            "dead_case_id": dead_case.id,
            "live_case_id": live_case.id,
            "n_dead_loads": len(dead_case.element_loads),
            "n_live_loads": len(live_case.element_loads),
            "partition_weight_case_id": partition_case_id,
            "loaded_floor_area": getattr(generator, "loaded_floor_area", {}),
            "unloaded_beams": getattr(generator, "unloaded_beams", []),
            # Resultant of each case (dead load includes the member self-weight entries), model force unit
            **totals,
            "force_unit": str(parse_units(getattr(project.metadata, "units", None) or "kN, m").force),
        },
        next_suggested_stage="LOADS_APPLIED",
    )


_MAX_FLOOR_PRESSURE_KPA = 50.0


def _case_total(project: Any, case_id: str) -> float:
    """Total vertical load of one load case (sum of its loads, in the model force unit)."""
    from struct_core import MassSource, MassSourceFactor
    from struct_load.engine.mass_source import calculate_mass_source

    source = MassSource(
        id=f"total_{case_id}", factors=[MassSourceFactor(load_case_id=case_id, factor=1.0)],
        include_element_mass=False, include_added_mass=False,
    )
    return round(calculate_mass_source(project, mass_source=source).total_seismic_weight, 3)


def apply_lateral_loads(
    project: Any,
    *,
    code: str = "ASCE7-22",
    params: Optional[Any] = None,
    sds: Optional[float] = None,
    sd1: Optional[float] = None,
    s1: Optional[float] = None,
    tl: Optional[float] = None,
    r_factor: Optional[float] = None,
    importance_factor: Optional[float] = None,
    redundancy_factor: float = 1.0,
    system_type: Optional[str] = None,
    site_class: Optional[str] = None,
    risk_category: Optional[str] = None,
    mass_source: Optional[Any] = None,
    distribution: str = "face",
    wind_speed: Optional[float] = None,
    exposure_category: Optional[str] = None,
    tributary_width: Optional[float] = None,
    direction: str = "both",
    storage_live_cases: Optional[list[str]] = None,
    partition_cases: Optional[list[str]] = None,
    snow_cases: Optional[list[str]] = None,
    flat_roof_snow_pressure: float = 0.0,
    length_unit: Optional[str] = None,
) -> ToolResult:
    """
    Calculate and apply ASCE 7 seismic (ELF) and/or wind (MWFRS) lateral forces to the structural model.
    Computes exact model-derived story seismic weights per ASCE 7-22 Section 12.7.2 and supports directional lateral loads.

    Seismic loads need the site's ``sds`` and ``sd1`` (or ``params``); R comes from ``system_type`` (ASCE 7-22
    Table 12.2-1) and Ie from ``risk_category`` (Table 1.5-2) unless given; ``risk_category`` defaults to II and
    ``site_class`` to D (reported); ``tl`` (the mapped TL) is needed only for periods above 4 s. Wind loads need
    ``wind_speed`` and the frame's ``tributary_width`` (model length unit).
    """
    if project is None or getattr(project, "model", None) is None:
        return ToolResult.failure(
            "apply_lateral_loads",
            AgentError(
                error_type="missing_input",
                message="No valid Project provided to apply_lateral_loads.",
            ),
        )

    try:
        from struct_load import LoadGenerator, SeismicParameters, normalize_system_type
    except ImportError as exc:
        return ToolResult.failure(
            "apply_lateral_loads",
            AgentError(
                error_type="dependency_error",
                message=f"struct_load is unavailable: {exc}",
            ),
        )
    applied = []
    if system_type is None:
        system_type = defaults.SYSTEM_TYPE.value
        if params is None and sds is not None:
            applied.append(defaults.SYSTEM_TYPE.record())

    if params is None:
        try:
            system_type = normalize_system_type(system_type)
        except ValueError as exc:
            return ToolResult.failure(
                "apply_lateral_loads", AgentError(error_type="invalid_input", message=str(exc))
            )

    results_data = {}
    if (sds is None) != (sd1 is None) and params is None:
        return ToolResult.failure("apply_lateral_loads", AgentError(
            error_type="invalid_input", message="Seismic loads need both sds and sd1 (the site's design accelerations)."))
    if params is None and sds is not None:
        if risk_category is None:
            risk_category = defaults.RISK_CATEGORY.value
            applied.append(defaults.RISK_CATEGORY.record())
        if site_class is None:
            site_class = defaults.SITE_CLASS.value
            applied.append(defaults.SITE_CLASS.record())
    if wind_speed is not None and wind_speed > 0:
        if tributary_width is None:
            return ToolResult.failure("apply_lateral_loads", AgentError(
                error_type="invalid_input", message="Wind loads need tributary_width (the loaded width of the frame)."))
        if exposure_category is None:
            exposure_category = defaults.EXPOSURE_CATEGORY.value
            applied.append(defaults.EXPOSURE_CATEGORY.record())
    elif exposure_category is None:
        exposure_category = "B"
    if params is None and sds is None and not (wind_speed is not None and wind_speed > 0):
        return ToolResult.failure("apply_lateral_loads", AgentError(
            error_type="invalid_input",
            message="No lateral load requested: give sds and sd1 (seismic ELF) or wind_speed (wind)."))

    from .units_input import to_model_units
    from struct_core import parse_units

    model_units = parse_units(getattr(project, "units", None) or getattr(project.metadata, "units", None) or defaults.UNITS.value)
    if tributary_width is not None and length_unit is not None:
        try:
            tributary_width = to_model_units(tributary_width, length_unit, "length", model_units)
        except ValueError as exc:
            return ToolResult.failure("apply_lateral_loads", AgentError(error_type="invalid_input", message=str(exc)))

    generator = LoadGenerator(units=model_units)

    try:
        # Apply seismic if params or (sds and sd1) provided
        if params is not None or (sds is not None and sd1 is not None):
            seismic_params = params
            if seismic_params is None:
                seismic_params = SeismicParameters(
                    sds=sds,
                    sd1=sd1,
                    s1=s1,
                    tl=tl,
                    r_factor=r_factor,
                    importance_factor=importance_factor,
                    redundancy_factor=redundancy_factor,
                    system_type=system_type,
                    site_class=site_class,
                    risk_category=risk_category,
                )
            # A modal or dynamic analysis attaches a default ASCE 7 mass source (ensure_seismic_mass_source), which
            # lumps the lower half of the ground-story columns at the base: an ELF call that names no mass source
            # then got a smaller W only because a modal analysis ran first (minimax B04: 715 kN instead of 751 kN).
            # The ELF weight does not depend on that order; a mass source given or imported still applies.
            default_ms = getattr(project.model, "mass_source", None)
            if mass_source is None and getattr(default_ms, "id", None) == "MS_ASCE7":
                project.model.mass_source = None
            try:
                eq_result, elf_res = generator.apply_lateral_seismic(
                    project,
                    params=seismic_params,
                    direction=direction,
                    mass_source=mass_source,
                    distribution=str(distribution or "face").lower().strip(),
                    storage_live_cases=storage_live_cases,
                    partition_cases=partition_cases,
                    snow_cases=snow_cases,
                    flat_roof_snow_pressure=flat_roof_snow_pressure,
                )
            finally:
                if project.model.mass_source is None:
                    project.model.mass_source = default_ms
            eq_cases = eq_result if isinstance(eq_result, list) else [eq_result]
            if isinstance(eq_result, list):
                results_data["seismic_case_ids"] = [c.id for c in eq_result]
            else:
                results_data["seismic_case_id"] = eq_result.id
            results_data["seismic_base_shear"] = elf_res.base_shear
            results_data["seismic_story_forces"] = elf_res.story_forces
            if not elf_res.s1_bound_evaluated:
                results_data["note"] = (
                    "S1 was not given, so the Cs lower bound 0.5 S1/(R/Ie) of ASCE 7-22 Eq. 12.8-7 "
                    "(required where S1 >= 0.6) was not evaluated."
                )

            from .seismic_basis import record_seismic_basis

            results_data["seismic_basis"] = record_seismic_basis(
                project,
                system_type=str(seismic_params.system_type),
                r_factor=float(seismic_params.r_factor),
                importance_factor=float(seismic_params.importance_factor),
                risk_category=str(seismic_params.risk_category),
                sds=float(seismic_params.sds),
                sd1=float(seismic_params.sd1),
                s1=seismic_params.s1,
                procedure="elf",
                lateral_case_ids=[c.id for c in eq_cases],
                directions=[str(c.seismic_params["direction"]).lower() for c in eq_cases],
                redundancy_factor=float(seismic_params.redundancy_factor),
            )
            # The story forces depend on the seismic weight: keep the inputs so a member resize can
            # regenerate them (not possible for a SeismicParameters object or a custom mass source)
            basis = project.seismic_basis
            if basis is not None and params is None and mass_source is None:
                inputs = {
                    "code": code, "sds": sds, "sd1": sd1, "s1": s1, "tl": tl, "r_factor": r_factor,
                    "importance_factor": importance_factor, "system_type": system_type, "site_class": site_class,
                    "risk_category": risk_category, "distribution": distribution, "direction": direction,
                    "storage_live_cases": storage_live_cases, "partition_cases": partition_cases,
                    "snow_cases": snow_cases, "flat_roof_snow_pressure": flat_roof_snow_pressure,
                }
                own = {str(c.seismic_params["direction"]).lower() for c in eq_cases}
                earlier = set(basis.directions) - own  # directions of earlier calls kept in the basis
                if not earlier:
                    basis.elf_inputs = inputs
                elif {k: v for k, v in basis.elf_inputs.items() if k != "direction"} == {k: v for k, v in inputs.items() if k != "direction"}:
                    # EQ_X and EQ_Y from separate calls: a resize regenerates both
                    basis.elf_inputs = {**inputs, "direction": "both" if set(basis.directions) == {"x", "y"}
                                        else direction}
                else:
                    basis.elf_inputs = {}  # the directions differ in other inputs: no single call regenerates them
            elif basis is not None:
                basis.elf_inputs = {}

        # Apply wind if wind_speed provided
        if wind_speed is not None and wind_speed > 0:
            w_result, wind_res = generator.apply_lateral_wind(
                project,
                basic_wind_speed=wind_speed,
                exposure_category=exposure_category,
                tributary_width=tributary_width,
                direction=direction,
            )
            if isinstance(w_result, list):
                results_data["wind_case_ids"] = [c.id for c in w_result]
            else:
                results_data["wind_case_id"] = w_result.id
            results_data["wind_base_shear"] = wind_res.total_base_shear
            results_data["wind_story_forces"] = wind_res.story_forces

    except Exception as exc:
        return ToolResult.failure(
            "apply_lateral_loads",
            AgentError(
                error_type="load_generation_failure",
                message=f"Failed to apply lateral loads: {exc}",
            ),
        )

    if applied:
        results_data["defaults_applied"] = [d.model_dump() for d in applied]
        if hasattr(project, "defaults_applied"):
            project.defaults_applied.extend(applied)
    return ToolResult.success(
        "apply_lateral_loads",
        data=results_data,
        next_suggested_stage="LOADS_APPLIED",
    )


def apply_roof_loads(
    project: Any,
    *,
    tributary_width: Optional[float] = None,
    one_way_span: Optional[Literal["x", "y"]] = None,
    two_way: Optional[bool] = None,
    ground_snow_load: Optional[float] = None,
    roof_live_load: Optional[float] = None,
    tributary_area: Optional[float] = None,
    exposure_factor_ce: float = 1.0,
    thermal_factor_ct: float = 1.0,
    risk_category: Optional[str] = None,
    pressure_unit: Optional[str] = None,
    length_unit: Optional[str] = None,
    area_unit: Optional[str] = None,
) -> ToolResult:
    """
    Calculate and apply ASCE 7-22 Snow and/or Roof Live loads to roof framing beams.

    ASCE 7-22 snow loads carry no importance factor; ``risk_category`` sets the minimum
    low-slope roof snow load (Table 7.3-4). For 2D frames, ``tributary_width`` is required;
    for 3D frames, specify ``one_way_span`` ('x' or 'y') or ``two_way=True``.
    Without ``roof_live_load`` the minimum Lr is reduced for ``tributary_area`` (model area unit) only when
    it is given (ASCE 7-22 Eq. 4.8-1).
    """
    if project is None or getattr(project, "model", None) is None:
        return ToolResult.failure(
            "apply_roof_loads",
            AgentError(
                error_type="missing_input",
                message="No valid Project provided to apply_roof_loads.",
            ),
        )

    try:
        from struct_load import LoadGenerator
    except ImportError as exc:
        return ToolResult.failure(
            "apply_roof_loads",
            AgentError(
                error_type="dependency_error",
                message=f"struct_load is unavailable: {exc}",
            ),
        )

    applied = []
    if risk_category is None:
        risk_category = defaults.RISK_CATEGORY.value
        applied.append(defaults.RISK_CATEGORY.record())

    from .units_input import to_model_units
    from struct_core import parse_units

    model_units = parse_units(getattr(project, "units", None) or getattr(project.metadata, "units", None) or defaults.UNITS.value)
    try:
        ground_snow_load = to_model_units(ground_snow_load, pressure_unit, "pressure", model_units)
        roof_live_load = to_model_units(roof_live_load, pressure_unit, "pressure", model_units)
        tributary_width = to_model_units(tributary_width, length_unit, "length", model_units)
        tributary_area = to_model_units(tributary_area, area_unit, "area", model_units)
    except ValueError as exc:
        return ToolResult.failure(
            "apply_roof_loads",
            AgentError(
                error_type="invalid_input",
                message=str(exc),
            ),
        )

    try:
        generator = LoadGenerator(units=model_units)
        created = generator.apply_roof_loads(
            project,
            ground_snow_load=ground_snow_load,
            roof_live_load=roof_live_load,
            tributary_width=tributary_width,
            one_way_span=one_way_span,
            two_way=two_way,
            tributary_area=tributary_area,
            exposure_factor_ce=exposure_factor_ce,
            thermal_factor_ct=thermal_factor_ct,
            risk_category=risk_category,
        )
    except Exception as exc:
        return ToolResult.failure(
            "apply_roof_loads",
            AgentError(
                error_type="load_generation_failure",
                message=f"Failed to apply roof loads: {exc}",
            ),
        )

    data = {
        "cases_created": list(created.keys()),
        "load_case_ids": [c.id for c in created.values()],
    }
    if applied:
        data["defaults_applied"] = [d.model_dump() for d in applied]
        if hasattr(project, "defaults_applied"):
            project.defaults_applied.extend(applied)

    return ToolResult.success(
        "apply_roof_loads",
        data=data,
        next_suggested_stage="LOADS_APPLIED",
    )


def distribute_area_loads(
    project: Any,
    *,
    target_case_id: Optional[str] = None,
    load_type_filter: Optional[str] = None,
    use_equivalent_uniform: bool = True,
    design_action: str = "moment",
) -> ToolResult:
    """
    Map surface pressures from AreaLoad schemas onto perimeter beam elements using tributary strips.
    """
    if project is None or getattr(project, "model", None) is None:
        return ToolResult.failure(
            "distribute_area_loads",
            AgentError(
                error_type="missing_input",
                message="No valid Project provided to distribute_area_loads.",
            ),
        )

    try:
        from struct_load import LoadGenerator
    except ImportError as exc:
        return ToolResult.failure(
            "distribute_area_loads",
            AgentError(
                error_type="dependency_error",
                message=f"struct_load is unavailable: {exc}",
            ),
        )

    try:
        generator = LoadGenerator()
        n_loads = generator.distribute_area_loads(
            project,
            target_case_id=target_case_id,
            load_type_filter=load_type_filter,
            use_equivalent_uniform=use_equivalent_uniform,
            design_action=design_action,
        )
    except Exception as exc:
        return ToolResult.failure(
            "distribute_area_loads",
            AgentError(
                error_type="load_generation_failure",
                message=f"Failed to distribute area loads: {exc}",
            ),
        )

    return ToolResult.success(
        "distribute_area_loads",
        data={
            "distributed_loads_created": n_loads,
            "target_case_id": target_case_id,
        },
        next_suggested_stage="LOADS_APPLIED",
    )


def generate_load_combinations(
    project: Any,
    *,
    standard: str = "ACI318_LRFD",
    orthogonal_seismic: bool = True,
    sds: Optional[float] = None,
    rho: float = 1.0,
    include_overstrength: bool = False,
    omega_0: Optional[float] = None,
    seismic_design_category: Optional[str] = None,
    reduce_live_in_seismic: bool = False,
) -> ToolResult:
    """
    Scan existing load cases in the project model and generate standard factored load combinations.
    Supports vertical seismic effect Ev = 0.2*SDS*D and overstrength combinations per ASCE 7-22 Section 12.4.
    """
    if project is None or getattr(project, "model", None) is None:
        return ToolResult.failure(
            "generate_load_combinations",
            AgentError(
                error_type="missing_input",
                message="No valid Project provided to generate_load_combinations.",
            ),
        )

    try:
        from struct_load import LoadGenerator
        from struct_core import parse_units
    except ImportError as exc:
        return ToolResult.failure(
            "generate_load_combinations",
            AgentError(
                error_type="dependency_error",
                message=f"struct_load is unavailable: {exc}",
            ),
        )

    # Strength design unless allowable-stress design is asked for: gpt-oss passed standard="ACI 318-25", which
    # used to fall through to ASD combinations (ACI 318 has no ASD)
    method = "ASD" if "ASD" in str(standard).upper() else "LRFD"
    try:
        model_units = parse_units(getattr(project, "units", None) or getattr(project.metadata, "units", None) or defaults.UNITS.value)
        generator = LoadGenerator(units=model_units)
        combs = generator.generate_combinations(
            project,
            design_method=method,
            orthogonal_seismic=orthogonal_seismic,
            sds=sds,
            rho=rho,
            include_overstrength=include_overstrength,
            omega_0=omega_0,
            seismic_design_category=seismic_design_category,
            reduce_live_in_seismic=reduce_live_in_seismic,
        )
    except Exception as exc:
        return ToolResult.failure(
            "generate_load_combinations",
            AgentError(
                error_type="combination_generation_failure",
                message=f"Failed to generate load combinations: {exc}",
            ),
        )

    return ToolResult.success(
        "generate_load_combinations",
        data={
            "standard": standard,
            "method": method,
            "n_combinations": len(combs),
            "combination_ids": [c.id for c in combs],
        },
        next_suggested_stage="MODEL_VALIDATED",
    )


def apply_self_weight(
    project: Any,
    *,
    target_case_id: Optional[str] = None,
    direction: Optional[str] = None,
) -> ToolResult:
    """
    Materialize member self-weight distributed loads for any load case where include_self_weight=True.
    """
    if project is None or getattr(project, "model", None) is None:
        return ToolResult.failure(
            "apply_self_weight",
            AgentError(
                error_type="missing_input",
                message="No valid Project provided to apply_self_weight.",
            ),
        )

    try:
        from struct_load import apply_self_weight as run_apply_self_weight
    except ImportError as exc:
        return ToolResult.failure(
            "apply_self_weight",
            AgentError(
                error_type="dependency_error",
                message=f"struct_load is unavailable: {exc}",
            ),
        )

    try:
        run_apply_self_weight(
            project,
            target_case_id=target_case_id,
            direction=direction,
        )
    except Exception as exc:
        return ToolResult.failure(
            "apply_self_weight",
            AgentError(
                error_type="load_generation_failure",
                message=f"Failed to apply self-weight: {exc}",
            ),
        )

    n_loads = sum(len(lc.element_loads) for lc in project.model.load_cases)
    return ToolResult.success(
        "apply_self_weight",
        data={
            "n_load_cases": len(project.model.load_cases),
            "total_element_loads": n_loads,
        },
        next_suggested_stage="LOADS_APPLIED",
    )


def define_mass_source(
    project: Any,
    *,
    name: str = "Default",
    factors: Optional[dict[str, float]] = None,
    include_element_mass: bool = True,
    include_added_mass: bool = True,
    lateral_only: bool = False,
    asce7_auto: bool = False,
) -> ToolResult:
    """
    Define a commercial-grade Mass Source (m = sum c_i * W_i / g) for dynamic/seismic analysis.

    Parameters
    ----------
    project : Any
        The project container holding the structural model.
    name : str
        Identifier for this mass source definition (default: "Default").
    factors : dict[str, float], optional
        Load case factors converting load to mass (e.g. {"DEAD": 1.0, "STORAGE_LIVE": 0.25, "PARTITION": 1.0}).
        Note: Per ASCE 7-22 Section 12.7.2, 25% live load factor applies strictly to storage areas;
        general floor live load (office, residential, school) is 0% in seismic weight.
    include_element_mass : bool
        Whether to include structural element self-masses (rho * A * L) (default: True).
    include_added_mass : bool
        Whether to include lumped nodal/element masses (default: True).
    lateral_only : bool
        Whether dynamic mass participates only in lateral translational DOFs.
    asce7_auto : bool, default=False
        Whether to automatically construct an ASCE 7-22 Section 12.7.2 compliant mass source
        from model load cases (Dead 100%, Storage Live 25%, Partitions 100%, General Live 0%).

    Returns
    -------
    ToolResult
        Success result with total seismic weight, total mass, and center of mass,
        or structured AgentError.
    """
    if project is None or getattr(project, "model", None) is None:
        return ToolResult.failure(
            "define_mass_source",
            AgentError(
                error_type="missing_input",
                message="No valid Project provided to define_mass_source.",
            ),
        )

    try:
        from struct_core import MassSource, MassSourceFactor
        from struct_load import calculate_mass_source, create_asce7_mass_source
    except ImportError as exc:
        return ToolResult.failure(
            "define_mass_source",
            AgentError(
                error_type="dependency_error",
                message=f"Required packages (struct_core / struct_load) unavailable: {exc}",
            ),
        )

    try:
        if asce7_auto:
            ms = create_asce7_mass_source(
                project.model,
                include_element_mass=include_element_mass,
                include_added_mass=include_added_mass,
                lateral_only=lateral_only,
                source_id=name,
                name=f"ASCE 7-22 Effective Seismic Weight ({name})",
            )
            if factors:
                existing_f = {f.load_case_id: f for f in ms.factors}
                for lc_id, f_val in factors.items():
                    existing_f[lc_id] = MassSourceFactor(load_case_id=lc_id, factor=float(f_val))
                ms.factors = list(existing_f.values())
        else:
            factor_objs = []
            if factors:
                for lc_id, f in factors.items():
                    factor_objs.append(MassSourceFactor(load_case_id=lc_id, factor=float(f)))

            ms = MassSource(
                id=name,
                name=name,
                factors=factor_objs,
                include_element_mass=include_element_mass,
                include_added_mass=include_added_mass,
                lateral_only=lateral_only,
            )

        project.model.mass_source = ms
        if not hasattr(project.model, "mass_sources") or project.model.mass_sources is None:
            project.model.mass_sources = []
        existing_idx = next((i for i, s in enumerate(project.model.mass_sources) if s.id == ms.id), None)
        if existing_idx is not None:
            project.model.mass_sources[existing_idx] = ms
        else:
            project.model.mass_sources.append(ms)

        # Evaluate seismic weight & mass distribution
        eval_res = calculate_mass_source(project, mass_source=ms)

    except Exception as exc:
        return ToolResult.failure(
            "define_mass_source",
            AgentError(
                error_type="mass_source_failure",
                message=f"Failed to define/calculate mass source: {exc}",
            ),
        )

    return ToolResult.success(
        "define_mass_source",
        data={
            "mass_source_id": ms.id,
            "total_weight": eval_res.total_seismic_weight,
            "total_mass": eval_res.total_mass,
            "center_of_mass": eval_res.center_of_mass,
            "story_masses": eval_res.story_masses,
            "factors": {f.load_case_id: f.factor for f in ms.factors},
        },
        next_suggested_stage="LOADS_APPLIED",
    )
