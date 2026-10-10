"""
Loads tools for struct_agent.

Wraps ``struct_load`` into validated, deterministic ToolResult calls.
The agent decides *what* to load (occupancy, criteria); the tools deterministically calculate *how*.
"""

from __future__ import annotations

from typing import Any, Optional, Union
from .base import AgentError, ToolResult


def apply_gravity_loads(
    project: Any,
    *,
    occupancy: str = "office",
    tributary_width: float = 5.0,
    superimposed_dead: float = 1.5,
    include_self_weight: bool = True,
    custom_live_load: Optional[float] = None,
    unit_system: Optional[str] = None,
) -> ToolResult:
    """
    Apply code-compliant Dead and Live gravity loads to a Project's structural model.
    Automatically senses 2D (wy) vs 3D (wz) geometry and infers units from project metadata.

    Parameters
    ----------
    project : Project
        The project container holding the structural model.
    occupancy : str
        Building occupancy type per ASCE 7 / IBC (e.g. 'office', 'residential', 'classroom').
    tributary_width : float
        Tributary framing width (m for SI, ft for US).
    superimposed_dead : float
        Superimposed dead load pressure for finishes and MEP (kPa for SI, psf for US).
    include_self_weight : bool
        Whether to calculate element self-weight from sections and density.
    custom_live_load : float, optional
        Custom live load pressure override.
    unit_system : str, optional
        'si' or 'us'. If None, auto-inferred from project metadata.

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

    try:
        generator = LoadGenerator(unit_system=unit_system)
        dead_case, live_case = generator.apply_gravity_loads(
            project,
            occupancy=occupancy,
            tributary_width=tributary_width,
            superimposed_dead=superimposed_dead,
            include_self_weight=include_self_weight,
            custom_live_load=custom_live_load,
        )
    except Exception as exc:
        return ToolResult.failure(
            "apply_gravity_loads",
            AgentError(
                error_type="load_generation_failure",
                message=f"Failed to generate gravity loads: {exc}",
            ),
        )

    return ToolResult.success(
        "apply_gravity_loads",
        data={
            "occupancy": occupancy,
            "tributary_width": tributary_width,
            "dead_case_id": dead_case.id,
            "live_case_id": live_case.id,
            "n_dead_loads": len(dead_case.element_loads),
            "n_live_loads": len(live_case.element_loads),
        },
        next_suggested_stage="LOADS_APPLIED",
    )


def apply_lateral_loads(
    project: Any,
    *,
    code: str = "ASCE7-22",
    sds: Optional[float] = 0.75,
    sd1: Optional[float] = 0.40,
    r_factor: float = 8.0,
    system_type: str = "rc_smf",
    wind_speed: Optional[float] = None,
    exposure_category: str = "B",
    tributary_width: float = 5.0,
    direction: str = "both",
    unit_system: Optional[str] = None,
) -> ToolResult:
    """
    Calculate and apply ASCE 7 seismic (ELF) and/or wind (MWFRS) lateral forces to the structural model.
    Computes exact model-derived story seismic weights and supports directional lateral loads.
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
        from struct_load import LoadGenerator
    except ImportError as exc:
        return ToolResult.failure(
            "apply_lateral_loads",
            AgentError(
                error_type="dependency_error",
                message=f"struct_load is unavailable: {exc}",
            ),
        )

    results_data = {}
    generator = LoadGenerator(unit_system=unit_system)

    try:
        # Apply seismic if SDS provided
        if sds is not None and sd1 is not None:
            eq_result, elf_res = generator.apply_lateral_seismic(
                project,
                sds=sds,
                sd1=sd1,
                r_factor=r_factor,
                system_type=system_type,
                tributary_width=tributary_width,
                direction=direction,
            )
            if isinstance(eq_result, list):
                results_data["seismic_case_ids"] = [c.id for c in eq_result]
            else:
                results_data["seismic_case_id"] = eq_result.id
            results_data["seismic_base_shear"] = elf_res.base_shear
            results_data["seismic_story_forces"] = elf_res.story_forces

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

    return ToolResult.success(
        "apply_lateral_loads",
        data=results_data,
        next_suggested_stage="LOADS_APPLIED",
    )


def apply_roof_loads(
    project: Any,
    *,
    ground_snow_load: Optional[float] = None,
    roof_live_load: Optional[float] = None,
    tributary_width: float = 5.0,
    exposure_factor_ce: float = 1.0,
    thermal_factor_ct: float = 1.0,
    importance_factor: float = 1.0,
    unit_system: Optional[str] = None,
) -> ToolResult:
    """
    Calculate and apply ASCE 7 Snow and/or Roof Live loads to roof framing beams.
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

    try:
        generator = LoadGenerator(unit_system=unit_system)
        created = generator.apply_roof_loads(
            project,
            ground_snow_load=ground_snow_load,
            roof_live_load=roof_live_load,
            tributary_width=tributary_width,
            exposure_factor_ce=exposure_factor_ce,
            thermal_factor_ct=thermal_factor_ct,
            importance_factor=importance_factor,
        )
    except Exception as exc:
        return ToolResult.failure(
            "apply_roof_loads",
            AgentError(
                error_type="load_generation_failure",
                message=f"Failed to apply roof loads: {exc}",
            ),
        )

    return ToolResult.success(
        "apply_roof_loads",
        data={
            "cases_created": list(created.keys()),
            "load_case_ids": [c.id for c in created.values()],
        },
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
    unit_system: Optional[str] = None,
    orthogonal_seismic: bool = True,
) -> ToolResult:
    """
    Scan existing load cases in the project model and generate standard factored load combinations.
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
    except ImportError as exc:
        return ToolResult.failure(
            "generate_load_combinations",
            AgentError(
                error_type="dependency_error",
                message=f"struct_load is unavailable: {exc}",
            ),
        )

    method = "LRFD" if "LRFD" in standard.upper() else "ASD"
    generator = LoadGenerator(unit_system=unit_system)
    combs = generator.generate_combinations(
        project,
        design_method=method,
        orthogonal_seismic=orthogonal_seismic,
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
    unit_system: Optional[str] = None,
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
            unit_system=unit_system,
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
