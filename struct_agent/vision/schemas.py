"""
struct_agent/vision/schemas.py
══════════════════════════════
Intermediate semantic drafting schemas for vision-extracted architectural
and structural plans, elevations, and grid layouts.
"""

from __future__ import annotations

from typing import List, Literal, Optional, Tuple, Union
from pydantic import BaseModel, ConfigDict, Field


class GridLineDraft(BaseModel):
    """
    Represents a grid axis along X or Y dimension.
    """
    model_config = ConfigDict(extra="ignore")

    label: str = Field(description="Grid mark label, e.g. '1', '2', '3' or 'A', 'B', 'C'")
    direction: Literal["X", "Y"] = Field(description="Direction of grid line or axis perpendicular to it")
    spacing_to_next: float = Field(default=0.0, description="Span distance to the next adjacent grid line in meters")
    coordinate: Optional[float] = Field(default=None, description="Explicit coordinate offset along axis (optional)")


class StoryDraft(BaseModel):
    """
    Represents a vertical story level or elevation datum.
    """
    model_config = ConfigDict(extra="ignore")

    name: str = Field(description="Story name, e.g. 'Ground Floor', 'Story 1', 'Roof'")
    height: float = Field(description="Floor-to-floor vertical height in meters")
    elevation: Optional[float] = Field(default=None, description="Absolute elevation Z in meters (optional)")


class ColumnDraft(BaseModel):
    """
    Represents a column location at a grid intersection.
    """
    model_config = ConfigDict(extra="ignore")

    grid: Tuple[str, str] = Field(description="Grid intersection identifier, e.g. ['A', '1'] or ['1', 'A']")
    section: Optional[str] = Field(default=None, description="Section mark or name, e.g. 'C450x450'")
    b: Optional[float] = Field(default=0.45, description="Column width b in meters")
    h: Optional[float] = Field(default=0.45, description="Column depth h in meters")
    stories: Optional[List[str]] = Field(default=None, description="List of story names where this column exists (None = all)")


class BeamDraft(BaseModel):
    """
    Represents a horizontal framing beam spanning between grid intersections.
    """
    model_config = ConfigDict(extra="ignore")

    from_grid: Tuple[str, str] = Field(description="Start grid intersection, e.g. ['A', '1']")
    to_grid: Tuple[str, str] = Field(description="End grid intersection, e.g. ['A', '2']")
    section: Optional[str] = Field(default=None, description="Section mark or name, e.g. 'B300x500'")
    b: Optional[float] = Field(default=0.30, description="Beam width b in meters")
    h: Optional[float] = Field(default=0.50, description="Beam depth h in meters")
    stories: Optional[List[str]] = Field(default=None, description="List of story names where this beam exists (None = all floor levels)")


class FloorPlanDraft(BaseModel):
    """
    Composite semantic layout representation extracted from drawings.
    """
    model_config = ConfigDict(extra="ignore")

    title: str = Field(default="Architectural Plan Model", description="Project title")
    units: str = Field(default="kN, m", description="Engineering unit system")
    grids_x: List[GridLineDraft] = Field(default_factory=list, description="Grid lines along X axis (e.g. 1, 2, 3...)")
    grids_y: List[GridLineDraft] = Field(default_factory=list, description="Grid lines along Y axis (e.g. A, B, C...)")
    stories: List[StoryDraft] = Field(default_factory=list, description="Vertical story levels (from ground up)")
    columns: List[ColumnDraft] = Field(default_factory=list, description="Explicit column placements (empty = auto-populate all grid intersections)")
    beams: List[BeamDraft] = Field(default_factory=list, description="Explicit beam lines (empty = auto-connect grid lines)")
    support_type: Literal["Fixed", "Pinned"] = Field(default="Fixed", description="Base boundary restraint type")
    concrete_fc: float = Field(default=30.0, description="Characteristic concrete compressive strength in MPa")
    dead_load: float = Field(default=12.0, description="Typical floor dead load on beams in kN/m")
    live_load: float = Field(default=8.0, description="Typical floor live load on beams in kN/m")
    dead_case: str = Field(default="DEAD", description="Dead load case name")
    live_case: str = Field(default="LIVE", description="Live load case name")
    missing_information: List[str] = Field(default_factory=list, description="List of missing parameters requiring user clarification (e.g. story heights)")


def normalize_floor_plan_dict(raw: dict) -> FloorPlanDraft:
    """
    Tolerantly convert arbitrary LLM function calling arguments or raw dictionary
    into a validated FloorPlanDraft.
    """
    raw = dict(raw)

    # 1. Grids X
    gx_raw = raw.get("x_grids") or raw.get("grids_x") or []
    grids_x: List[GridLineDraft] = []
    for item in gx_raw:
        if isinstance(item, GridLineDraft):
            grids_x.append(item)
        elif isinstance(item, dict):
            grids_x.append(
                GridLineDraft(
                    label=str(item.get("name") or item.get("label") or ""),
                    direction="X",
                    spacing_to_next=float(item.get("spacing_to_next") or item.get("spacing") or 0.0),
                    coordinate=float(item["coordinate"]) if ("coordinate" in item and item["coordinate"] is not None) else None,
                )
            )

    # 2. Grids Y
    gy_raw = raw.get("y_grids") or raw.get("grids_y") or []
    grids_y: List[GridLineDraft] = []
    for item in gy_raw:
        if isinstance(item, GridLineDraft):
            grids_y.append(item)
        elif isinstance(item, dict):
            grids_y.append(
                GridLineDraft(
                    label=str(item.get("name") or item.get("label") or ""),
                    direction="Y",
                    spacing_to_next=float(item.get("spacing_to_next") or item.get("spacing") or 0.0),
                    coordinate=float(item["coordinate"]) if ("coordinate" in item and item["coordinate"] is not None) else None,
                )
            )

    # 3. Stories
    s_raw = raw.get("stories") or []
    stories: List[StoryDraft] = []
    for item in s_raw:
        if isinstance(item, StoryDraft):
            stories.append(item)
        elif isinstance(item, dict):
            stories.append(
                StoryDraft(
                    name=str(item.get("name") or f"Story {len(stories)+1}"),
                    height=float(item.get("height") or 3.0),
                    elevation=float(item["elevation"]) if ("elevation" in item and item["elevation"] is not None) else None,
                )
            )

    # 4. Columns
    col_raw = raw.get("columns") or []
    cols: List[ColumnDraft] = []
    default_col_b = float(raw.get("default_col_b") or 0.45)
    default_col_h = float(raw.get("default_col_h") or 0.45)
    for item in col_raw:
        if isinstance(item, ColumnDraft):
            cols.append(item)
        elif isinstance(item, dict):
            if "grid" in item:
                g = tuple(item["grid"])
            else:
                gx = str(item.get("grid_x", ""))
                gy = str(item.get("grid_y", ""))
                g = (gx, gy)
            cols.append(
                ColumnDraft(
                    grid=g,
                    section=item.get("section"),
                    b=float(item.get("section_b") or item.get("b") or default_col_b),
                    h=float(item.get("section_h") or item.get("h") or default_col_h),
                )
            )

    # 5. Beams
    beam_raw = raw.get("beams") or []
    beams: List[BeamDraft] = []
    default_beam_b = float(raw.get("default_beam_b") or 0.30)
    default_beam_h = float(raw.get("default_beam_h") or 0.50)
    for item in beam_raw:
        if isinstance(item, BeamDraft):
            beams.append(item)
        elif isinstance(item, dict):
            if "from_grid" in item and "to_grid" in item:
                fg = tuple(item["from_grid"])
                tg = tuple(item["to_grid"])
            else:
                fg = (str(item.get("start_grid_x", "")), str(item.get("start_grid_y", "")))
                tg = (str(item.get("end_grid_x", "")), str(item.get("end_grid_y", "")))
            beams.append(
                BeamDraft(
                    from_grid=fg,
                    to_grid=tg,
                    section=item.get("section"),
                    b=float(item.get("section_b") or item.get("b") or default_beam_b),
                    h=float(item.get("section_h") or item.get("h") or default_beam_h),
                )
            )

    # Support & Units
    sup = str(raw.get("support_type") or raw.get("base_support") or "Fixed").capitalize()
    if sup not in ("Fixed", "Pinned"):
        sup = "Fixed"

    units = raw.get("units") or "kN, m"

    return FloorPlanDraft(
        title=str(raw.get("title") or "Architectural Plan Model"),
        units=units,
        grids_x=grids_x,
        grids_y=grids_y,
        stories=stories,
        columns=cols,
        beams=beams,
        support_type=sup,
        concrete_fc=float(raw.get("concrete_fc") or 30.0),
        dead_load=float(raw.get("dead_load") or 12.0),
        live_load=float(raw.get("live_load") or 8.0),
    )
