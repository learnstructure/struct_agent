"""
Example: End-to-End Dynamic Earthquake Analysis & ACI 318 Reinforced Concrete Design.

Demonstrates Option 3 (Hybrid Architecture):
1. Parametric frame generation (struct_core)
2. Modal eigenvalue analysis via structdyn MDF (run_analysis)
3. 1940 El Centro earthquake time-history integration (run_time_history)
4. Dynamic response inspection (peak roof displacement, story drift, base shear)
5. Peak dynamic force enveloping across all time steps
6. ACI 318 member flexure, shear, and detailing design against the earthquake envelope (run_design)
7. Production CAD structural drawing export in DXF format (struct_draw)
"""

from pathlib import Path
import numpy as np

from struct_agent import (
    create_frame_model,
    inspect_analysis_results,
    inspect_design_results,
    inspect_dynamic_results,
    generate_drawings,
    run_analysis,
    run_design,
    run_time_history,
)


def run_dynamic_pipeline():
    print("=" * 75)
    print("STRUCT_AGENT END-TO-END DYNAMIC ANALYSIS & SEISMIC DESIGN PIPELINE")
    print("=" * 75)

    # -------------------------------------------------------------------------
    # 1. Create 3-Story, 2-Bay Moment Frame Model
    # -------------------------------------------------------------------------
    print("\n[Step 1] Creating 3-story, 2-bay RC frame model...")
    create_res = create_frame_model(
        num_bays=2,
        bay_width=6.0,      # 6.0 m bays
        num_stories=3,
        story_height=3.5,   # 3.5 m stories
        dimension="2d",
        support_type="fixed",
        title="3-Story Seismic Moment Frame",
        col_b=0.45,
        col_h=0.45,
        beam_b=0.35,
        beam_h=0.55,
    )
    assert create_res.ok, f"Model creation failed: {create_res.error}"
    project = create_res.data["project"]
    print(f"  -> Model created: {len(project.model.nodes)} nodes, {len(project.model.elements)} elements")

    # Assign story lumped masses (e.g. 15 tonnes per story floor level)
    mass_per_node = 15000.0 / 3.0  # kg per floor node
    for n in project.model.nodes:
        if n.y > 0.0:
            n.mass = mass_per_node

    # -------------------------------------------------------------------------
    # 2. Modal Analysis (structdyn MDF)
    # -------------------------------------------------------------------------
    print("\n[Step 2] Running Eigenvalue Modal Analysis via structdyn MDF...")
    modal_res = run_analysis(
        project,
        analysis_case_id="Modal_Analysis",
        analysis_type="modal",
        solver="auto",
        num_modes=3,
    )
    assert modal_res.ok, f"Modal analysis failed: {modal_res.error}"
    print(f"  -> Solver used: {modal_res.data['solver_used']}")
    print(f"  -> Fundamental Period T1: {modal_res.data['fundamental_period']:.3f} s")

    modal_info = inspect_analysis_results(project, "Modal_Analysis")
    periods = modal_info.data["modal"]["periods"]
    freqs = modal_info.data["modal"]["frequencies"]
    for i, (T, f) in enumerate(zip(periods, freqs), start=1):
        print(f"     Mode {i}: T = {T:.3f} s | f = {f:.2f} Hz")

    # -------------------------------------------------------------------------
    # 3. Dynamic Earthquake Time-History Analysis (structdyn Newmark-Beta)
    # -------------------------------------------------------------------------
    print("\n[Step 3] Running 1940 El Centro Earthquake Time-History Analysis...")
    th_res = run_time_history(
        project,
        analysis_case_id="Earthquake_ElCentro",
        earthquake="elcentro",
        direction="x",
        damping_ratio=0.05,  # 5% critical damping
        n_modes_damping=3,
        method="newmark_beta",
        solver="auto",
        envelope_to_design=True,
    )
    assert th_res.ok, f"Time-history analysis failed: {th_res.error}"
    print(f"  -> Duration: {th_res.data['duration']:.2f} s ({th_res.data['n_steps']} time steps)")
    print(f"  -> Peak Ground Acceleration: {th_res.data['pga'] / 9.80665:.3f} g")
    print(f"  -> Peak Lateral Roof Displacement: {th_res.data['peak_lateral_displacement'] * 1000:.2f} mm")
    print(f"  -> Max Dynamic Bending Moment Envelope: {th_res.data['max_moment_envelope']:.1f} kN-m")
    print(f"  -> Max Dynamic Shear Envelope: {th_res.data['max_shear_envelope']:.1f} kN")

    # Inspect dynamic results
    dyn_inspect = inspect_dynamic_results(project, "Earthquake_ElCentro")
    assert dyn_inspect.ok

    # -------------------------------------------------------------------------
    # 4. ACI 318 RC Member Code Design against Dynamic Earthquake Envelope
    # -------------------------------------------------------------------------
    print("\n[Step 4] Designing RC frame members against peak dynamic envelopes (ACI 318)...")
    design_res = run_design(
        project,
        design_case_id="ACI318_Dynamic_Design",
        analysis_case_id="Earthquake_ElCentro",
        design_code="ACI 318-19",
    )
    assert design_res.ok, f"Design failed: {design_res.error}"
    print(f"  -> Governing Demand-Capacity Ratio (DCR): {design_res.data['governing_dcr']:.2f}")
    print(f"  -> Member Design Status: {'PASSED' if design_res.data['passed'] else 'REBAR SIZED / STRESS CHECK COMPLETE'}")

    design_inspect = inspect_design_results(project, "ACI318_Dynamic_Design")
    print(f"  -> Total Members Checked: {len(design_inspect.data.get('elements', {}))}")

    # -------------------------------------------------------------------------
    # 5. Production CAD Structural Drawing Generation
    # -------------------------------------------------------------------------
    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    dxf_path = out_dir / "dynamic_frame_seismic.dxf"

    print(f"\n[Step 5] Generating CAD structural drawings -> {dxf_path.name}...")
    draw_res = generate_drawings(
        project,
        output_path=str(dxf_path),
        design_case_id="ACI318_Dynamic_Design",
    )
    assert draw_res.ok, f"Drawing generation failed: {draw_res.error}"
    print(f"  -> Saved DXF: {draw_res.data['output_path']}")

    print("\n" + "=" * 75)
    print("DYNAMIC ANALYSIS & SEISMIC DESIGN PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 75)


if __name__ == "__main__":
    run_dynamic_pipeline()
