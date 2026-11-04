"""
Engineering defaults the tools apply when the user gives no value.

Each default is defined once, here, in its reference unit and with its source. A tool converts it to the model units
(``in_units``) and reports it (``record``) in its result's ``defaults_applied``, and design results keep the record, so
the agent, the user, and the drawings see every value that was not given. Values with no meaningful default (geometry,
f'c, SDS, SD1, occupancy, superimposed dead load) have no entry: the tools require them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Union

from struct_core import AppliedDefault, conversion_factor


@dataclass(frozen=True)
class EngineeringDefault:
    """A default value in its reference unit system (``reference_units``, for ``quantity``), with its source."""

    name: str
    value: Union[float, str]
    unit: Optional[str]                 # label of the reference unit, e.g. 'ksi', 'kPa'
    source: str
    quantity: Optional[str] = None      # struct_core dimension key: 'stress', 'pressure', 'length', 'weight_density'
    reference_units: Optional[str] = None  # unit pair the value is expressed in, e.g. 'kip, in'

    def in_units(self, units: Any) -> Union[float, str]:
        """The value converted to ``units`` (the model's), or the value itself when it has no unit."""
        if self.quantity is None:
            return self.value
        return float(self.value) * conversion_factor(self.reference_units, units, self.quantity)

    def record(self, applies_to: Optional[str] = None, model_units: Any = None) -> AppliedDefault:
        """The struct_core record of this default having been applied (to a section or material, if given), in
        ``model_units`` when given."""
        if model_units is not None and self.quantity is not None:
            from .tools.units_input import unit_label

            value: Union[float, str] = round(float(self.in_units(model_units)), 6)
            unit: Optional[str] = unit_label(model_units, self.quantity)
        else:
            value, unit = self.value, self.unit
        return AppliedDefault(name=self.name, value=value, unit=unit, source=self.source,
                              applies_to=None if applies_to is None else str(applies_to))


# Units
UNITS = EngineeringDefault("units", "kN, m", None, "SI default unit system")

# Trial section sizes for preliminary frame sizing and vision compilation
TRIAL_COLUMN_SIZE = EngineeringDefault("column_size", 0.45, "m", "preliminary trial column dimension", "length", "kN, m")
TRIAL_BEAM_WIDTH = EngineeringDefault("beam_width", 0.30, "m", "preliminary trial beam width", "length", "kN, m")
TRIAL_BEAM_DEPTH = EngineeringDefault("beam_depth", 0.50, "m", "preliminary trial beam depth", "length", "kN, m")

# Reinforcement: ASTM A615 / A706 Grade 60, the common grade for ACI 318 design
REBAR_FY = EngineeringDefault("fy", 60.0, "ksi", "ASTM A615 Grade 60", "stress", "kip, in")
REBAR_FU = EngineeringDefault("fu", 90.0, "ksi", "ASTM A615 Grade 60", "stress", "kip, in")
REBAR_E = EngineeringDefault("Es", 29000.0, "ksi", "ACI 318-25 §20.2.2.2", "stress", "kip, in")
REBAR_UNIT_WEIGHT = EngineeringDefault("rebar unit_weight", 490.0, "pcf", "steel", "weight_density", "lbf, ft")

# Concrete (f'c is required; E follows from it, see concrete_modulus)
CONCRETE_UNIT_WEIGHT = EngineeringDefault(
    "concrete unit_weight", 150.0, "pcf", "normal-weight concrete (ACI 318-25 §19.2.4)", "weight_density", "lbf, ft")
CONCRETE_NU = EngineeringDefault("concrete nu", 0.2, None, "common practice for concrete")
CONCRETE_E_SOURCE = "ACI 318-25 §19.2.2.1: Ec = 57,000 sqrt(f'c) psi (normal-weight concrete)"

# Clear cover to the transverse reinforcement (ACI 318-25 Table 20.5.1.3.1)
COVER = EngineeringDefault(
    "cover", 1.5, "in", "ACI 318-25 Table 20.5.1.3.1 (beams and columns not exposed to weather or in contact with "
                        "ground)", "length", "kip, in")
FOOTING_COVER = EngineeringDefault(
    "footing_cover", 3.0, "in", "ACI 318-25 Table 20.5.1.3.1 (cast against and permanently in contact with ground)",
    "length", "kip, in")

# Foundations: a preliminary value; the drawings mark footings designed with it as preliminary
ALLOWABLE_BEARING_PRESSURE = EngineeringDefault(
    "allowable_bearing_pressure", 150.0, "kPa", "assumed for preliminary footing sizing; confirm by geotechnical "
                                                "investigation", "pressure", "kN, m")

# Seismic
RISK_CATEGORY = EngineeringDefault("risk_category", "II", None, "ASCE 7-22 Table 1.5-1 (ordinary buildings)")
SITE_CLASS = EngineeringDefault("site_class", "D", None, "ASCE 7-22 §20.1 (default when soil properties are unknown)")
SYSTEM_TYPE = EngineeringDefault("system_type", "rc_smf", None, "ASCE 7-22 Table 12.2-1 (Special reinforced concrete moment frames, R=8)")
EXPOSURE_CATEGORY = EngineeringDefault("exposure_category", "B", None, "ASCE 7-22 §26.7.3 (urban and suburban areas)")


def concrete_modulus(fc: float, units: Any) -> float:
    """Ec = 57,000 sqrt(f'c [psi]) (ACI 318-25 §19.2.2.1, normal-weight concrete), for ``fc`` and the result in the
    stress unit of ``units``."""
    fc_psi = fc * conversion_factor(units, "lbf, in", "stress")
    return 57000.0 * fc_psi ** 0.5 * conversion_factor("lbf, in", units, "stress")
