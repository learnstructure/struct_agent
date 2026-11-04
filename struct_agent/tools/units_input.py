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
# Unit label -> a unit pair whose length unit it is
LENGTH_UNITS: Dict[str, str] = {
    "m": "kN, m", "meter": "kN, m", "meters": "kN, m",
    "mm": "N, mm", "millimeter": "N, mm", "millimeters": "N, mm",
    "cm": "kN, cm", "centimeter": "kN, cm", "centimeters": "kN, cm",
    "ft": "kip, ft", "foot": "kip, ft", "feet": "kip, ft",
    "in": "kip, in", "inch": "kip, in", "inches": "kip, in",
}
# Unit label -> a unit pair whose area (length^2) unit it is
AREA_UNITS: Dict[str, str] = {
    "m2": "kN, m", "m^2": "kN, m", "sqm": "kN, m", "square_meter": "kN, m",
    "mm2": "N, mm", "mm^2": "N, mm", "sqmm": "N, mm",
    "ft2": "kip, ft", "ft^2": "kip, ft", "sqft": "kip, ft", "square_foot": "kip, ft", "square_feet": "kip, ft",
    "in2": "kip, in", "in^2": "kip, in", "sqin": "kip, in", "square_inch": "kip, in", "square_inches": "kip, in",
}
# Unit label -> a unit pair whose weight-density (force / length^3) unit it is
WEIGHT_DENSITY_UNITS: Dict[str, str] = {"kn/m3": "kN, m", "kn/m^3": "kN, m", "n/m3": "N, m", "n/m^3": "N, m",
                                        "pcf": "lbf, ft", "lb/ft3": "lbf, ft", "kcf": "kip, ft", "pci": "lbf, in",
                                        "kip/in3": "kip, in"}
# A mass density given for a weight density (e.g. concrete "2400 kg/m3"): its weight under standard gravity, in N/m3
MASS_DENSITY_UNITS: Dict[str, float] = {"kg/m3": 9.80665, "kg/m^3": 9.80665, "t/m3": 9806.65, "t/m^3": 9806.65}
STANDARD_GRAVITY_SOURCE = "standard gravity 9.80665 m/s2"
_TABLES = {
    "stress": STRESS_UNITS,
    "pressure": STRESS_UNITS,
    "weight_density": WEIGHT_DENSITY_UNITS,
    "length": LENGTH_UNITS,
    "area": AREA_UNITS,
}


def to_model_units(value: Optional[float], unit: Optional[str], quantity: str, model_units: Any) -> Optional[float]:
    """``value`` given in ``unit`` (a label such as 'MPa', 'psf', 'pcf'), converted to ``model_units``; unchanged when
    ``unit`` is None (already in model units) or ``value`` is None. Raises ValueError for an unknown unit."""
    if value is None or unit is None:
        return value
    table = _TABLES[quantity]
    label = str(unit).strip().lower().replace(" ", "").replace("³", "3")
    if quantity == "weight_density" and label in MASS_DENSITY_UNITS:  # mass density -> weight density
        return float(value) * MASS_DENSITY_UNITS[label] * conversion_factor("N, m", model_units, quantity)
    source = table.get(label)
    if source is None:
        known = sorted(set(table) | (set(MASS_DENSITY_UNITS) if quantity == "weight_density" else set()))
        raise ValueError(f"Unknown {quantity.replace('_', ' ')} unit '{unit}'; use one of {known}.")
    return float(value) * conversion_factor(source, model_units, quantity)


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
