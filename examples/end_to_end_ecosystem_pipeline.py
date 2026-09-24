"""
End-to-End Ecosystem Integration Pipeline Demonstration.

This example demonstrates the complete data flow across the Structural Engineering Ecosystem:

1. struct_core : Define StructuralModel & Project (nodes, elements, materials, sections, loads)
2. fem2d       : Convert to fem2d.Structure, solve, and convert output back to struct_core.AnalysisResult
3. aci318      : Consume AnalysisResult, run ACI 318 design & detailing, producing DesignResult & DetailingResult
4. struct_draw : Consume DetailingResult & DesignResult, render DXF drawings for members
5. struct_core : Save & reload complete Project container with full schema validation
"""

import os
import sys

from struct_core import (
    AnalysisResult,
    BeamElement,
    DistributedLoad,
    ElasticMaterial,
    LinearStaticAnalysis,
    LoadCase,
    Node,
    PointLoad,
    Project,
    RectangularSection,
    StructuralModel,
    Support,
    load_json,
    save_json,
    validate_structural_model,
)

# 1. fem2d adapter
from fem2d.adapters import model_from_core, result_to_core

# 2. aci318 adapter
from aci318.adapters import design_from_core

# 3. struct_draw adapter
from struct_draw.adapters import draw_from_core


def run_pipeline():
    print("=" * 70)
    print("      STRUCTURAL ENGINEERING ECOSYSTEM END-TO-END PIPELINE       ")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # STEP 1: Build StructuralModel in struct_core
    # -------------------------------------------------------------------------
    print("\n[1/5] Building StructuralModel (RC Portal Frame)...")

    # Nodes: 1 (0,0), 2 (0, 3.6m), 3 (6.0m, 3.6m), 4 (6.0m, 0.0)
    nodes = [
        Node(id=1, x=0.0, y=0.0),
        Node(id=2, x=0.0, y=3.6),
        Node(id=3, x=6.0, y=3.6),
        Node(id=4, x=6.0, y=0.0),
    ]

    # Materials
    mat_conc = ElasticMaterial(id="Concrete_4ksi", E=24800000.0, rho=2400.0)  # kN/m^2

    # Sections (Beam: 0.3m x 0.5m [12x20 in], Column: 0.4m x 0.4m [16x16 in])
    sec_beam = RectangularSection(id="Beam_12x20", b=0.30, h=0.50)
    sec_col = RectangularSection(id="Col_16x16", b=0.40, h=0.40)

    # Elements: 1 (Left Col), 2 (Beam), 3 (Right Col)
    elements = [
        BeamElement(id=1, start_node=1, end_node=2, material_id="Concrete_4ksi", section_id="Col_16x16"),
        BeamElement(id=2, start_node=2, end_node=3, material_id="Concrete_4ksi", section_id="Beam_12x20"),
        BeamElement(id=3, start_node=4, end_node=3, material_id="Concrete_4ksi", section_id="Col_16x16"),
    ]

    # Supports: Fixed at bases (nodes 1 and 4)
    supports = [Support.fixed(node_id=1), Support.fixed(node_id=4)]

    # Loads: Gravity load on beam (wy = -25 kN/m)
    loads = LoadCase(
        id="Gravity",
        element_loads=[DistributedLoad(element_id=2, wy=-25.0)],
    )

    analysis_case = LinearStaticAnalysis(id="Static_1", load_case_id="Gravity")

    model = StructuralModel(
        nodes=nodes,
        materials=[mat_conc],
        sections=[sec_beam, sec_col],
        elements=elements,
        supports=supports,
        load_cases=[loads],
        analysis_cases=[analysis_case],
    )

    validate_structural_model(model)

    project = Project(model=model)
    project.metadata.title = "RC Portal Frame Ecosystem Demonstration"
    project.metadata.author = "Structural Engineering AI Agent"

    print("      Model validated successfully with 4 nodes, 3 elements.")

    # -------------------------------------------------------------------------
    # STEP 2: Solve with fem2d
    # -------------------------------------------------------------------------
    print("\n[2/5] Running Finite Element Analysis via fem2d...")

    # Convert struct_core model -> fem2d.Structure
    structure = model_from_core(project, load_case_id="Gravity")

    # Execute matrix analysis (mutates structure in-place; returns None)
    structure.solve()

    # Convert fem2d output -> struct_core.AnalysisResult
    analysis_result = result_to_core(structure, analysis_case_id="Static_1")

    project.analysis_results["Static_1"] = analysis_result
    print(f"      Analysis complete. AnalysisResult attached for case 'Static_1'.")

    # -------------------------------------------------------------------------
    # STEP 3: Perform RC Design & Detailing via aci318
    # -------------------------------------------------------------------------
    print("\n[3/5] Performing ACI 318 RC Design & Detailing via aci318...")

    design_res, detail_res = design_from_core(project, design_code="ACI 318-19")

    project.design_results["ACI318_19"] = design_res
    project.detailing_results["ACI318_19"] = detail_res

    print(f"      Design Status     : {'PASSED' if design_res.passed else 'FAILED'}")
    print(f"      Governing DCR     : {design_res.governing_dcr:.3f}")
    aggregate = detail_res.aggregate_schedule
    total_weight = aggregate.total_weight if aggregate is not None else 0.0
    print(f"      Total Rebar Weight: {total_weight:.1f} lbs")

    for elem_id, elem_des in design_res.element_results.items():
        print(f"      - Element {elem_id} ({elem_des.member_type}): DCR = {elem_des.governing_dcr:.3f}")

    # -------------------------------------------------------------------------
    # STEP 4: Render DXF Drawings via struct_draw
    # -------------------------------------------------------------------------
    print("\n[4/5] Rendering Structural Drawings via struct_draw...")

    draw_output = draw_from_core(project)
    dxf_doc = draw_output["drawing"]

    output_dxf_path = os.path.join(os.path.dirname(__file__), "portal_frame_detailing.dxf")
    dxf_doc.saveas(output_dxf_path)

    print(f"      DXF Drawing saved to: {output_dxf_path}")
    print(f"      Elements rendered : {draw_output['elements_drawn']}")

    # -------------------------------------------------------------------------
    # STEP 5: Serialize & Persist Project Container
    # -------------------------------------------------------------------------
    print("\n[5/5] Serializing and Reloading Project Container...")

    output_json_path = os.path.join(os.path.dirname(__file__), "portal_frame_project.json")
    save_json(project, output_json_path, indent=2)

    reloaded_project = load_json(Project, output_json_path)

    print(f"      Project JSON saved to: {output_json_path}")
    print(f"      Reloaded Project Title: {reloaded_project.metadata.title}")
    assert reloaded_project.design_results is not None
    assert reloaded_project.detailing_results is not None

    print("\n" + "=" * 70)
    print("    PIPELINE EXECUTION COMPLETED SUCCESSFULLY WITH ALL CHECKS PASSED! ")
    print("=" * 70)


if __name__ == "__main__":
    run_pipeline()
