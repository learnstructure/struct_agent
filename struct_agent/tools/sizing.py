"""
Preliminary member sizing heuristics for struct_agent.

Provides rule-of-thumb structural sizing so an LLM can bootstrap a complete,
analyzable structural model from bare geometry alone.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional
from struct_core import (
    ElasticMaterial,
    RectangularSection,
)
from .base import AgentError, ToolResult


def preliminary_sizing(
    project: Any,
    *,
    material_id: str = "Concrete_4ksi",
    fc_kpa: float = 27580.0,  # ~4000 psi
    rho_kg_m3: float = 2400.0,
    min_beam_width: float = 0.25,
    min_col_dim: float = 0.35,
) -> ToolResult:
    """
    Apply engineering rules-of-thumb to size RC beams and columns based on member spans.

    - Beams: Depth h ≈ L / 14 (rounded up to nearest 50 mm), width b ≈ 0.55 * h (min 250 mm).
    - Columns: Dimension b = h ≈ max(350 mm, story_height / 10).

    Parameters
    ----------
    project : Project
        The project container holding the structural model.
    material_id : str
        ID for default concrete material.
    fc_kpa : float
        Specified compressive strength f'c (kPa).
    rho_kg_m3 : float
        Unit mass density (kg/m³).
    min_beam_width : float
        Minimum allowable beam width (m).
    min_col_dim : float
        Minimum column cross-sectional dimension (m).

    Returns
    -------
    ToolResult
        Result containing created sections and summary statistics.
    """
    if project is None or getattr(project, "model", None) is None:
        return ToolResult.failure(
            "preliminary_sizing",
            AgentError(
                error_type="missing_input",
                message="No valid Project provided to preliminary_sizing.",
            ),
        )

    model = project.model
    nodes_map = {n.id: n for n in model.nodes}

    # Ensure material exists
    if not any(m.id == material_id for m in model.materials):
        # E = 4700 * sqrt(fc_psi) * 6.89476 kPa ≈ 24,800,000 kPa for 4000 psi
        e_kpa = 4700.0 * math.sqrt(fc_kpa / 6.89476) * 6.89476
        model.materials.append(
            ElasticMaterial(id=material_id, E=round(e_kpa, 1), rho=rho_kg_m3)
        )

    sections_created = {}
    beams_sized = 0
    cols_sized = 0

    for elem in model.elements:
        if getattr(elem, "type", "") != "beam":
            continue

        n_i = nodes_map.get(elem.start_node)
        n_j = nodes_map.get(elem.end_node)
        if not n_i or not n_j:
            continue

        dx = n_j.x - n_i.x
        dy = n_j.y - n_i.y
        span = math.sqrt(dx * dx + dy * dy)

        is_vertical = abs(dx) < 0.05 and abs(dy) > 0.05
        is_horizontal = abs(dy) < 0.05 and abs(dx) > 0.05

        if is_horizontal:
            # Beam sizing: h ≈ L / 14 rounded up to nearest 0.05 m
            h_raw = span / 14.0
            h = max(0.30, math.ceil(h_raw * 20.0) / 20.0)
            b_raw = h * 0.55
            b = max(min_beam_width, math.ceil(b_raw * 20.0) / 20.0)

            sec_id = f"Beam_{int(b*1000)}x{int(h*1000)}"
            if sec_id not in sections_created:
                sections_created[sec_id] = RectangularSection(id=sec_id, b=round(b, 2), h=round(h, 2))
            elem.section_id = sec_id
            elem.material_id = material_id
            beams_sized += 1

        elif is_vertical:
            # Column sizing: b = h ≈ max(350 mm, H / 10)
            dim_raw = max(min_col_dim, span / 10.0)
            dim = math.ceil(dim_raw * 20.0) / 20.0

            sec_id = f"Col_{int(dim*1000)}x{int(dim*1000)}"
            if sec_id not in sections_created:
                sections_created[sec_id] = RectangularSection(id=sec_id, b=round(dim, 2), h=round(dim, 2))
            elem.section_id = sec_id
            elem.material_id = material_id
            cols_sized += 1

    # Attach created sections to model
    existing_sec_ids = {s.id for s in model.sections}
    for sec_id, sec_obj in sections_created.items():
        if sec_id not in existing_sec_ids:
            model.sections.append(sec_obj)

    return ToolResult.success(
        "preliminary_sizing",
        data={
            "beams_sized": beams_sized,
            "columns_sized": cols_sized,
            "sections_created": list(sections_created.keys()),
        },
        next_suggested_stage="MODEL_CREATED",
    )
