"""
struct_agent demo pipeline.

Drives the full canonical workflow end-to-end using only the
struct_agent public tool surface. Mirrors the existing
``end_to_end_ecosystem_pipeline.py`` but does so through the
StructAgent orchestrator.
"""

from __future__ import annotations

import os
import sys

from struct_agent import StructAgent


def main(output_dir: str | None = None) -> int:
    output_dir = output_dir or os.path.dirname(os.path.abspath(__file__))

    agent = StructAgent()
    state = agent.run_demo_pipeline(
        output_dxf=os.path.join(output_dir, "demo_pipeline_detailing.dxf"),
        save_json_path=os.path.join(output_dir, "demo_pipeline_project.json"),
    )

    print("=" * 70)
    print("        struct_agent demo pipeline completed successfully       ")
    print("=" * 70)
    print(f"  Final workflow stage : {state.stage.value}")
    print(f"  Tool invocations     : {len(state.tool_log)}")
    for entry in state.tool_log:
        print(f"    - {entry['tool']:<26} {entry['status']}")
    print(f"  Project title        : {state.project.metadata.title}")

    # Extract design and detailing results dynamically from the project
    design_case_id = "ACI318_19"
    design_res = state.project.design_results.get(design_case_id)
    detail_res = state.project.detailing_results.get(design_case_id)

    if design_res is not None:
        print("\n" + "-" * 70)
        print("  DESIGN RESULTS SUMMARY (ACI 318-19)")
        print("-" * 70)
        print(f"  Overall Design Status : {'PASSED' if design_res.passed else 'FAILED'}")
        print(f"  Governing DCR         : {design_res.governing_dcr:.3f}")
        for elem_id, elem_des in design_res.element_results.items():
            print(f"    * Element {elem_id:<4} ({elem_des.member_type:<7}): DCR = {elem_des.governing_dcr:.3f} [{'PASS' if elem_des.passed else 'FAIL'}]")

    if detail_res is not None and detail_res.aggregate_schedule is not None:
        sched = detail_res.aggregate_schedule
        print("\n" + "-" * 70)
        print("  BAR BENDING SCHEDULE (BBS) [Dynamically computed from member design]")
        print("-" * 70)
        print(f"  {'Mark':<8} {'Bar':<6} {'Qty':<6} {'Cut L (in)':<12} {'Tot L (ft)':<12} {'Weight (lbs)':<14} {'Shape':<12} {'Location'}")
        print(f"  {'-'*8} {'-'*6} {'-'*6} {'-'*12} {'-'*12} {'-'*14} {'-'*12} {'-'*24}")
        for it in sched.items:
            tot_l_ft = it.total_length / 12.0
            print(f"  {it.mark:<8} #{it.bar_size:<5} {it.quantity:<6} {it.cut_length:<12.1f} {tot_l_ft:<12.1f} {it.weight:<14.1f} {it.shape_code:<12} {it.location or ''}")
        print(f"  {'-'*8} {'-'*6} {'-'*6} {'-'*12} {'-'*12} {'-'*14} {'-'*12} {'-'*24}")
        print(f"  TOTAL REBAR STEEL: {sched.total_weight:.1f} lbs ({sched.total_weight / 2000.0:.3f} US tons)")
        assert len(sched.items) > 0, "Schedule must contain items from member design"
        assert sched.total_weight > 0.0, "Total rebar weight must be positive"

    print("\n" + "=" * 70)
    print("  Artifacts generated:")
    print(f"    - DXF CAD Drawing  : {os.path.join(output_dir, 'demo_pipeline_detailing.dxf')}")
    print(f"    - Project Database : {os.path.join(output_dir, 'demo_pipeline_project.json')}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main(*(sys.argv[1:]) if len(sys.argv) > 1 else (None,)))