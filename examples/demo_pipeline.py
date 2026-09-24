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
        print(f"    - {entry['tool']:<24} {entry['status']}")
    print(f"  Project title        : {state.project.metadata.title}")

    return 0


if __name__ == "__main__":
    sys.exit(main(*(sys.argv[1:]) if len(sys.argv) > 1 else (None,)))