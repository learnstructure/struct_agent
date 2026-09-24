"""
Example: General Drawing Generation for Structural Framing & Trusses.

Generates general arrangement drawings showing outlines of frame members
(beams, columns, trusses) with grid lines and elevation levels, without
rebar details.
"""

from struct_draw import GeneralFrameData, TrussData, render, save_dxf


def run():
    print("Generating General Framing Plan & Elevation drawing...")
    frame = GeneralFrameData.from_grid(
        name="OFFICE-FRAME-3BAY",
        bays=(6000.0, 7500.0, 6000.0),  # mm (metric)
        story_heights=(4200.0, 3600.0, 3600.0),
        col_width=450.0,
        col_depth=450.0,
        beam_width=300.0,
        beam_depth=600.0,
        title="3-BAY 3-STORY FRAMING ELEVATION & PLAN",
        view_mode="both",
        is_metric=True,
    )

    save_dxf(frame, "examples/output/general_frame.dxf")
    print("  -> Saved examples/output/general_frame.dxf")

    print("Generating Roof Truss Outline drawings...")
    # Pratt Truss
    pratt = TrussData.pratt(
        span_in=18000.0,  # 18m
        height_in=2500.0,
        panels=8,
        title="18M PRATT STEEL ROOF TRUSS",
        chord_depth=200.0,
        web_depth=150.0,
        is_metric=True,
    )
    save_dxf(pratt, "examples/output/truss_pratt.dxf")
    print("  -> Saved examples/output/truss_pratt.dxf")

    # Warren Truss
    warren = TrussData.warren(
        span_in=15000.0,
        height_in=1800.0,
        panels=6,
        title="15M WARREN ROOF TRUSS",
        chord_depth=200.0,
        web_depth=150.0,
        is_metric=True,
    )
    save_dxf(warren, "examples/output/truss_warren.dxf")
    print("  -> Saved examples/output/truss_warren.dxf")

    # Pitched Truss
    pitched = TrussData.pitched(
        span_in=12000.0,
        height_in=3000.0,
        panels=6,
        title="12M PITCHED ROOF TRUSS",
        chord_depth=150.0,
        web_depth=100.0,
        is_metric=True,
    )
    save_dxf(pitched, "examples/output/truss_pitched.dxf")
    print("  -> Saved examples/output/truss_pitched.dxf")


if __name__ == "__main__":
    run()
