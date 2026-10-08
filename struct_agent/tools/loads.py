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
) -> ToolResult:
    """
    Apply code-compliant Dead and Live gravity loads to a Project's structural model.

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
        generator = LoadGenerator(unit_system="si")
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
) -> ToolResult:
    """
    Calculate and apply ASCE 7 seismic (ELF) and/or wind (MWFRS) lateral forces to the structural model.
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
    generator = LoadGenerator(unit_system="si")

    try:
        # Apply seismic if SDS provided
        if sds is not None and sd1 is not None:
            eq_case, elf_res = generator.apply_lateral_seismic(
                project,
                sds=sds,
                sd1=sd1,
                r_factor=r_factor,
                system_type=system_type,
                tributary_width=tributary_width,
            )
            results_data["seismic_case_id"] = eq_case.id
            results_data["seismic_base_shear"] = elf_res.base_shear
            results_data["seismic_story_forces"] = elf_res.story_forces

        # Apply wind if wind_speed provided
        if wind_speed is not None and wind_speed > 0:
            w_case, wind_res = generator.apply_lateral_wind(
                project,
                basic_wind_speed=wind_speed,
                exposure_category=exposure_category,
                tributary_width=tributary_width,
            )
            results_data["wind_case_id"] = w_case.id
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


def generate_load_combinations(
    project: Any,
    *,
    standard: str = "ACI318_LRFD",
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
    generator = LoadGenerator(unit_system="si")
    combs = generator.generate_combinations(project, design_method=method)

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

    Note: When analyzing with run_analysis, solvers (fem2d and fem3d) automatically compute
    and include element self-weight on the fly for cases with include_self_weight=True.
    Calling apply_self_weight explicitly writes DistributedLoad items directly into the
    Project schema (and sets include_self_weight=False to prevent double-counting).

    Parameters
    ----------
    project : Project
        The project container holding the structural model.
    target_case_id : str, optional
        Target load case ID if no case currently has include_self_weight=True.
    unit_system : str, optional
        'si' or 'us'. If omitted, automatically inferred from project.metadata.units.
    direction : str, optional
        Gravity direction: '-z' for 3D frames or '-y' for 2D plane frames. Inferred if omitted.

    Returns
    -------
    ToolResult
        Success ToolResult with summary of materialized loads.
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
