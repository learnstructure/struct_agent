"""
Realistic range validation for tool inputs.

Tool values are in the model's unit system (``project.metadata.units``). Input values
outside their realistic engineering range are refused as unit slips (e.g., passing
f'c in MPa when the project is in 'kN, m', which expects kPa).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from struct_core import conversion_factor, parse_units

# Unit label -> a unit pair whose stress / pressure (force / length^2) unit it is
STRESS_UNITS: Dict[str, str] = {"kpa": "kN, m", "pa": "N, m", "mpa": "N, mm", "psf": "lbf, ft", "ksf": "kip, ft",
                                "psi": "lbf, in", "ksi": "kip, in"}
# Unit label -> a unit pair whose weight-density (force / length^3) unit it is
WEIGHT_DENSITY_UNITS: Dict[str, str] = {"kn/m3": "kN, m", "kn/m^3": "kN, m", "n/m3": "N, m", "n/m^3": "N, m",
                                        "pcf": "lbf, ft", "lb/ft3": "lbf, ft", "kcf": "kip, ft", "pci": "lbf, in",
                                        "kip/in3": "kip, in"}
_TABLES = {
    "stress": STRESS_UNITS,
    "pressure": STRESS_UNITS,
    "weight_density": WEIGHT_DENSITY_UNITS,
}


def unit_label(model_units: Any, quantity: str) -> str:
    """The model's unit label for ``quantity`` (e.g. 'kPa' for kN, m stress)."""
    if quantity == "length":
        return str(getattr(parse_units(model_units), "length", "m"))
    for label, pair in _TABLES[quantity].items():
        if abs(conversion_factor(pair, model_units, quantity) - 1.0) < 1e-12:
            return {"kpa": "kPa", "mpa": "MPa", "pa": "Pa", "kn/m3": "kN/m3", "n/m3": "N/m3"}.get(label, label)
    return f"{quantity} unit of {model_units}"


# Realistic ranges in a reference unit: (label, quantity, low, high, reference unit pair, reference label).
# A value outside its range is a unit slip (e.g. f'c = 27.6 in a kN, m model is 27.6 kPa, not MPa) and is refused.
REALISTIC_RANGES = {
    "fc": ("concrete f'c", "stress", 2.0, 20.0, "kip, in", "ksi"),
    "fy": ("reinforcement fy", "stress", 30.0, 120.0, "kip, in", "ksi"),
    "fu": ("reinforcement fu", "stress", 40.0, 160.0, "kip, in", "ksi"),
    "unit_weight": ("concrete unit weight", "weight_density", 70.0, 200.0, "lbf, ft", "pcf"),
    "cover": ("clear cover", "length", 0.5, 6.0, "lbf, in", "in"),
    "footing_cover": ("footing clear cover", "length", 1.0, 12.0, "lbf, in", "in"),
    "allowable_bearing_pressure": ("allowable bearing pressure", "pressure", 0.5, 30.0, "kip, ft", "ksf"),
    "soil_depth": ("depth of soil above the footings", "length", 0.0, 4.0, "kN, m", "m"),
    "soil_unit_weight": ("soil unit weight", "weight_density", 10.0, 25.0, "kN, m", "kN/m^3"),
}


def check_realistic(key: str, value: Optional[float], model_units: Any) -> None:
    """Raise ValueError when ``value`` (model units) is outside the realistic range of input ``key``."""
    if value is None:
        return
    label, quantity, low, high, ref_units, ref_label = REALISTIC_RANGES[key]
    factor = conversion_factor(ref_units, model_units, quantity)
    if low * factor <= float(value) <= high * factor:
        return
    model_label = (getattr(parse_units(model_units), "length", "model length unit") if quantity == "length"
                   else unit_label(model_units, quantity))
    raise ValueError(
        f"{label} = {float(value):g} {model_label} is outside realistic bounds ({_plain(low * factor)} to "
        f"{_plain(high * factor)} {model_label}, i.e. {low:g} to {high:g} {ref_label}). All values must be given "
        f"in the project's {model_label} unit.")


def _plain(x: float) -> str:
    """Four significant figures without exponent notation (13,790, 0.0127)."""
    return f"{float(f'{x:.4g}'):,.6f}".rstrip("0").rstrip(".")
