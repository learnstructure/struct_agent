"""
Recording of the seismic design basis on a Project (read by drift/stability checks and by
ACI 318 Chapter 18 design).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def record_seismic_basis(
    project: Any,
    *,
    system_type: str,
    r_factor: float,
    importance_factor: float,
    risk_category: str,
    sds: Optional[float],
    sd1: Optional[float],
    s1: Optional[float],
    procedure: str,
    lateral_case_ids: List[str],
    directions: List[str],
    redundancy_factor: float = 1.0,
) -> Dict[str, Any]:
    """
    Store a ``SeismicDesignBasis`` on ``project`` with Omega_0 and Cd from ASCE 7-22 Table 12.2-1
    and the Seismic Design Category from Section 11.6 (when SDS and SD1 are known).

    Returns a summary for the tool output. ``system_type`` must be a key of
    ``struct_load.SEISMIC_SYSTEM_TYPES`` (ValueError otherwise); 'all_other' is outside Table 12.2-1,
    so no basis is recorded for it and the reason is returned instead.
    """
    from struct_core import SeismicDesignBasis
    from struct_load import determine_seismic_design_category, normalize_system_type, seismic_system_coefficients

    system_type = normalize_system_type(system_type)
    try:
        r_table, omega_0, cd = seismic_system_coefficients(system_type)
    except ValueError as exc:
        project.seismic_basis = None
        return {"recorded": False, "reason": str(exc)}

    sdc = None
    if sds is not None and sd1 is not None:
        sdc = determine_seismic_design_category(sds, sd1, risk_category=risk_category, s1=s1)
    previous = getattr(project, "seismic_basis", None)
    basis = SeismicDesignBasis(
        system=system_type,
        r_factor=r_factor,
        omega_0=omega_0,
        cd=cd,
        importance_factor=importance_factor,
        risk_category=risk_category,
        sds=sds,
        sd1=sd1,
        s1=s1,
        seismic_design_category=sdc,
        redundancy_factor=redundancy_factor,
        procedure=procedure,
        lateral_case_ids=list(lateral_case_ids),
        directions=list(directions),
    )
    if _same_basis(previous, basis):
        # One call per direction (EQ_X, then EQ_Y) describes one basis: the earlier directions stay in it
        kept = [(c, d) for c, d in zip(previous.lateral_case_ids, previous.directions) if d not in basis.directions]
        basis.lateral_case_ids = [c for c, _ in kept] + basis.lateral_case_ids
        basis.directions = [d for _, d in kept] + basis.directions
        basis.elf_inputs = dict(previous.elf_inputs)
    project.seismic_basis = basis
    summary = {
        "recorded": True,
        "system": system_type,
        "R": r_factor,
        "Omega_0": omega_0,
        "Cd": cd,
        "seismic_design_category": sdc,
    }
    if abs(r_factor - r_table) > 1e-9:
        summary["note"] = f"R = {r_factor} differs from ASCE 7-22 Table 12.2-1 value {r_table} for {system_type}."
    return summary


def _same_basis(previous: Any, basis: Any) -> bool:
    """Whether ``basis`` restates ``previous`` (same procedure and seismic parameters), possibly for other directions."""
    if previous is None:
        return False
    fields = ("procedure", "system", "r_factor", "importance_factor", "risk_category", "sds", "sd1", "s1",
              "redundancy_factor")
    return all(getattr(previous, f) == getattr(basis, f) for f in fields)
