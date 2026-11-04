"""
Material helpers shared by the model-creation tools.

Values are in the active model units (stresses in its stress unit, unit weights in its weight-density unit). A property
that is not given takes its default from ``struct_agent.defaults``, converted to the model units; each default used is
returned as an ``AppliedDefault`` for the tool to report.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

from struct_core import AppliedDefault, ConcreteMaterial, RebarMaterial, parse_units

from .. import defaults
from .units_input import check_realistic, unit_label


def concrete_material(
    material_id: str,
    *,
    fc: float,
    units: Any,
    E: Optional[float] = None,
    nu: Optional[float] = None,
    unit_weight: Optional[float] = None,
    name: Optional[str] = None,
) -> Tuple[ConcreteMaterial, List[AppliedDefault]]:
    """Concrete of strength ``fc`` (required); E from f'c (ACI 318-25 §19.2.2.1), ν = 0.2 and normal-weight unit weight
    when not given."""
    units = parse_units(units)
    check_realistic("fc", fc, units)
    check_realistic("unit_weight", unit_weight, units)
    applied: List[AppliedDefault] = []
    if E is None:
        E = defaults.concrete_modulus(fc, units)
        applied.append(AppliedDefault(name="E", value=round(E, 6), unit=unit_label(units, "stress"),
                                      source=defaults.CONCRETE_E_SOURCE,
                                      applies_to=str(material_id)))
    if nu is None:
        nu = defaults.CONCRETE_NU.in_units(units)
        applied.append(defaults.CONCRETE_NU.record(material_id, model_units=units))
    if unit_weight is None:
        unit_weight = defaults.CONCRETE_UNIT_WEIGHT.in_units(units)
        applied.append(defaults.CONCRETE_UNIT_WEIGHT.record(material_id, model_units=units))
    material = ConcreteMaterial(id=material_id, name=name or "Concrete", fc=fc, E=E, nu=nu, unit_weight=unit_weight)
    return material, applied


def rebar_material(
    material_id: str,
    *,
    units: Any,
    fy: Optional[float] = None,
    fu: Optional[float] = None,
    E: Optional[float] = None,
    unit_weight: Optional[float] = None,
    name: Optional[str] = None,
) -> Tuple[RebarMaterial, List[AppliedDefault]]:
    """Reinforcing steel; Grade 60 (fy, and fu with it), Es = 29,000 ksi and steel unit weight for the values not
    given. A given fy without fu leaves fu unset (the Grade 60 fu belongs to the Grade 60 fy)."""
    units = parse_units(units)
    check_realistic("fy", fy, units)
    check_realistic("fu", fu, units)
    applied: List[AppliedDefault] = []
    values = {"fy": fy, "fu": fu, "E": E, "unit_weight": unit_weight}
    for key, default in (("fy", defaults.REBAR_FY), ("fu", defaults.REBAR_FU), ("E", defaults.REBAR_E),
                         ("unit_weight", defaults.REBAR_UNIT_WEIGHT)):
        if key == "fu" and fy is not None:
            continue
        if values[key] is None:
            values[key] = default.in_units(units)
            applied.append(default.record(material_id, model_units=units))
    material = RebarMaterial(id=material_id, name=name or "Reinforcing steel", **values)
    return material, applied
