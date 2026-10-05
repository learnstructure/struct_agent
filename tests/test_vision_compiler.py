"""
tests/test_vision_compiler.py
═════════════════════════════
Unit tests for intermediate semantic schemas and deterministic grid compiler.
"""

import unittest
from struct_agent.vision import (
    BeamDraft,
    ColumnDraft,
    FloorPlanDraft,
    GridLineDraft,
    StoryDraft,
    build_project_from_draft,
)
from struct_agent import run_analysis


class TestVisionCompiler(unittest.TestCase):
    """Test deterministic grid and floor plan compilation."""

    def test_regular_grid_compilation(self):
        """Test regular 2x2 bay, 2-story building plan compilation."""
        draft = FloorPlanDraft(
            title="Office Building Frame",
            grids_x=[
                GridLineDraft(label="1", direction="X", spacing_to_next=6.0),
                GridLineDraft(label="2", direction="X", spacing_to_next=7.5),
                GridLineDraft(label="3", direction="X", spacing_to_next=0.0),
            ],
            grids_y=[
                GridLineDraft(label="A", direction="Y", spacing_to_next=5.0),
                GridLineDraft(label="B", direction="Y", spacing_to_next=6.0),
                GridLineDraft(label="C", direction="Y", spacing_to_next=0.0),
            ],
            stories=[
                StoryDraft(name="Ground Floor", height=3.6),
                StoryDraft(name="Second Floor", height=3.2),
            ],
            support_type="Fixed",
            dead_load=15.0,
            live_load=10.0,
        )

        proj = build_project_from_draft(draft)

        # 3 X-grids * 3 Y-grids = 9 column lines.
        # Story levels: 0 (Ground), 1 (Second Floor), 2 (Roof) -> 3 levels.
        # 9 * 3 = 27 nodes.
        self.assertEqual(len(proj.model.nodes), 27)

        # Base supports at level 0: 9 fixed supports
        self.assertEqual(len(proj.model.supports), 9)
        for s in proj.model.supports:
            self.assertTrue(s.ux and s.uy and s.uz and s.rx and s.ry and s.rz)

        # Columns: 9 column locations * 2 stories = 18 columns
        columns = [e for e in proj.model.elements if str(e.id).startswith("COL")]
        self.assertEqual(len(columns), 18)

        # Beams: (2 spans along X * 3 lines + 2 spans along Y * 3 lines) = 12 beams per floor.
        # 2 floors (level 1, level 2) = 24 beams
        beams = [e for e in proj.model.elements if str(e.id).startswith("BEAM")]
        self.assertEqual(len(beams), 24)

        # Total elements = 18 + 24 = 42
        self.assertEqual(len(proj.model.elements), 42)

        # Verify load cases
        case_ids = [c.id for c in proj.model.load_cases]
        self.assertIn("DEAD", case_ids)
        self.assertIn("LIVE", case_ids)

        # Verify combinations
        combo_ids = [c.id for c in proj.model.load_combinations]
        self.assertIn("COMB1", combo_ids)
        self.assertIn("COMB2", combo_ids)

    def test_custom_columns_and_beams(self):
        """Test L-shaped framing with explicit columns and beams."""
        draft = FloorPlanDraft(
            title="L-Shaped Frame",
            grids_x=[
                GridLineDraft(label="1", direction="X", spacing_to_next=6.0),
                GridLineDraft(label="2", direction="X", spacing_to_next=6.0),
                GridLineDraft(label="3", direction="X", spacing_to_next=0.0),
            ],
            grids_y=[
                GridLineDraft(label="A", direction="Y", spacing_to_next=6.0),
                GridLineDraft(label="B", direction="Y", spacing_to_next=6.0),
                GridLineDraft(label="C", direction="Y", spacing_to_next=0.0),
            ],
            stories=[
                StoryDraft(name="Floor 1", height=3.5),
            ],
            columns=[
                ColumnDraft(grid=["1", "A"], b=0.5, h=0.5),
                ColumnDraft(grid=["2", "A"], b=0.5, h=0.5),
                ColumnDraft(grid=["3", "A"], b=0.5, h=0.5),
                ColumnDraft(grid=["1", "B"], b=0.5, h=0.5),
                ColumnDraft(grid=["2", "B"], b=0.5, h=0.5),
            ],
            beams=[
                BeamDraft(from_grid=["1", "A"], to_grid=["2", "A"], b=0.35, h=0.6),
                BeamDraft(from_grid=["2", "A"], to_grid=["3", "A"], b=0.35, h=0.6),
                BeamDraft(from_grid=["1", "A"], to_grid=["1", "B"], b=0.35, h=0.6),
                BeamDraft(from_grid=["2", "A"], to_grid=["2", "B"], b=0.35, h=0.6),
                BeamDraft(from_grid=["1", "B"], to_grid=["2", "B"], b=0.35, h=0.6),
            ],
            support_type="Pinned",
        )

        proj = build_project_from_draft(draft)

        # 5 columns * 2 levels (0 and 1) = 10 nodes
        self.assertEqual(len(proj.model.nodes), 10)
        self.assertEqual(len(proj.model.supports), 5)
        # Check pinned
        for s in proj.model.supports:
            self.assertTrue(s.ux and s.uy and s.uz)
            self.assertFalse(s.rx or s.ry or s.rz)

        # 5 columns + 5 beams = 10 elements
        self.assertEqual(len(proj.model.elements), 10)

    def test_fem3d_static_analysis_on_compiled_project(self):
        """Run 3D FEM analysis on the compiled project and verify non-zero equilibrium."""
        draft = FloorPlanDraft(
            title="FEA Verified Model",
            grids_x=[
                GridLineDraft(label="1", direction="X", spacing_to_next=6.0),
                GridLineDraft(label="2", direction="X", spacing_to_next=0.0),
            ],
            grids_y=[
                GridLineDraft(label="A", direction="Y", spacing_to_next=6.0),
                GridLineDraft(label="B", direction="Y", spacing_to_next=0.0),
            ],
            stories=[
                StoryDraft(name="Story 1", height=3.5),
            ],
            dead_load=15.0,
            live_load=10.0,
        )

        proj = build_project_from_draft(draft)
        res = run_analysis(proj, analysis_case_id="envelope", solver="fem3d")

        self.assertTrue(res.ok)
        ana = proj.analysis_result
        self.assertIsNotNone(ana)

        # Check total vertical reaction equals total load
        # Under COMB2 (1.2D + 1.6L = 1.2*15 + 1.6*10 = 34 kN/m) on 4 beams * 6m = 816 kN
        reactions = ana.node_reactions
        supported_reactions = [r for r in reactions if abs(r.fz) > 1e-4]
        self.assertEqual(len(supported_reactions), 4)
        total_fz = sum(r.fz for r in reactions)
        self.assertAlmostEqual(total_fz, 816.0, delta=1.0)


if __name__ == "__main__":
    unittest.main()
