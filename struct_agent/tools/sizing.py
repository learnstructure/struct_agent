"""
Preliminary member sizing heuristics for struct_agent.

Gives a frame's sections rule-of-thumb starting sizes, so an LLM can bootstrap an analysable model from
bare geometry.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from struct_core import RectangularSection, classify_frame_members, conversion_factor, parse_units

from .base import AgentError, ToolResult
from .model import modify_project

# Rules of thumb in metres: (dimension step, least beam width, least beam depth, least column size)
_METRIC_RULES = (0.05, 0.25, 0.30, 0.35)
_US_RULES = (0.0508, 0.254, 0.3048, 0.3556)  # 2 in steps; 10 in, 12 in, 14 in
_METRIC_LENGTHS = {"m", "cm", "mm"}
# Sections created for members that have none (stable IDs, as create_frame_model uses)
_SIZED_ROLES = ("column", "beam")


def _ceil_to(value: float, step: float) -> float:
    return math.ceil(value / step - 1e-9) * step


def _element_lengths(model: Any) -> Dict[Any, float]:
    nodes = {n.id: n for n in model.nodes}
    lengths = {}
    for el in model.elements:
        a, b = nodes.get(el.start_node), nodes.get(el.end_node)
        if a is not None and b is not None:
            lengths[el.id] = math.dist((a.x, a.y, getattr(a, "z", 0.0)), (b.x, b.y, getattr(b, "z", 0.0)))
    return lengths


def preliminary_sizing(
    project: Any,
    *,
    min_beam_width: Optional[float] = None,
    min_col_dim: Optional[float] = None,
) -> ToolResult:
    """
    Rule-of-thumb starting sizes for the frame's rectangular beam and column sections.

    - Beam sections: depth h = longest span / 14 (at least 0.30 m / 12 in), width b = 0.55 h (at least
      ``min_beam_width``), with b <= h.
    - Column sections: square, b = h = tallest story / 10 (at least ``min_col_dim``).

    Each section is sized for the longest member assigned to it and keeps its ID, material, reinforcement and
    cover; only b and h change. Dimensions are rounded up to 0.05 m (2 in for US customary models). Members are classified as beams and columns by
    their orientation (``struct_core.classify_frame_members``, any vertical axis). The sizes are applied
    through ``modify_project``, so member self-weight and ELF story forces are regenerated.

    Parameters
    ----------
    min_beam_width, min_col_dim : float, optional
        Least beam width and column dimension, in the model's length unit (defaults 0.25 m and 0.35 m,
        or 10 in and 14 in).
    """
    if project is None or getattr(project, "model", None) is None:
        return ToolResult.failure("preliminary_sizing", AgentError(
            error_type="missing_input", message="No valid Project provided to preliminary_sizing."))

    model = project.model
    units = parse_units(getattr(project.metadata, "units", None) or "kN, m")
    to_m = conversion_factor(units, "kN, m", "length")
    step, beam_b_min, beam_h_min, col_min = _METRIC_RULES if units.length in _METRIC_LENGTHS else _US_RULES
    beam_b_min = beam_b_min if min_beam_width is None else float(min_beam_width) * to_m
    col_min = col_min if min_col_dim is None else float(min_col_dim) * to_m

    roles = classify_frame_members(model)
    lengths = _element_lengths(model)
    sections = {s.id: s for s in model.sections}
    framing = [el for el in model.elements if roles.get(el.id) in _SIZED_ROLES and el.id in lengths]

    no_section = sorted(str(el.id) for el in framing if el.section_id not in sections)
    if no_section:
        return ToolResult.failure("preliminary_sizing", AgentError(
            error_type="invalid_input",
            message=f"Members {no_section[:6]} reference sections that do not exist; define their sections (with a "
                    f"material) first."))

    governing: Dict[str, Tuple[str, float]] = {}  # section id -> (role, longest member in m)
    mixed: List[str] = []
    for el in framing:
        role = roles[el.id]
        sec_id = el.section_id
        previous = governing.get(sec_id)
        if previous is not None and previous[0] != role:
            mixed.append(sec_id)
        governing[sec_id] = (role, max(lengths[el.id] * to_m, previous[1] if previous else 0.0))
    if mixed:
        return ToolResult.failure("preliminary_sizing", AgentError(
            error_type="invalid_input",
            message=f"Sections {sorted(set(mixed))} are assigned to both beams and columns; give each its own section."))

    sizes: List[Dict[str, Any]] = []
    skipped: List[str] = []
    for sec_id, (role, length) in governing.items():
        existing = sections.get(sec_id)
        if existing is not None and not isinstance(existing, RectangularSection):
            skipped.append(sec_id)
            continue
        if role == "beam":
            h = _ceil_to(max(beam_h_min, length / 14.0), step)
            b = min(_ceil_to(max(beam_b_min, 0.55 * h), step), h)
        else:
            h = b = _ceil_to(max(col_min, length / 10.0), step)
        sizes.append({"id": sec_id, "role": role, "governing_length": round(length / to_m, 4),
                      "b": round(b / to_m, 6), "h": round(h / to_m, 6)})
    if not sizes:
        return ToolResult.failure("preliminary_sizing", AgentError(
            error_type="invalid_input", message="The model has no rectangular beam or column sections to size."))

    applied = modify_project(
        project,
        sections=[{"id": s["id"], "b": s["b"], "h": s["h"]} for s in sizes],
    )
    if not applied.ok:
        return ToolResult.failure("preliminary_sizing", applied.error)
    data = {key: value for key, value in applied.data.items() if key != "summary"}
    data["sections_sized"] = sizes
    if skipped:
        data["skipped_sections"] = skipped  # circular or general sections keep their properties
    return ToolResult.success("preliminary_sizing", data=data, next_suggested_stage="MODEL_CREATED")
