"""
Frame-level stability checks evaluated with the analysis:

- ACI 318-25 Table 6.6.3.1.1 cracked-section stiffness for factored-load analysis.
- ACI 318-25 Section 6.6.4 slenderness by moment magnification: the sway moments of the lateral
  load cases are magnified by the story delta_s = 1/(1 - Q) (Eq. 6.6.4.6.2a), and every column
  moment by the member-curvature magnifier delta (Eq. 6.6.4.5.2) with M2 >= M2,min. The limits of
  Sections 6.6.4.6.2 (delta_s <= 1.5 for Eq. a), 6.6.4.5.2 (Pu < 0.75 Pc), and 6.2.5.3
  (second-order end moments <= 1.4 first-order) are reported as code checks.
- ASCE 7-22 Section 12.12.1 story drift and Section 12.8.7 P-delta stability coefficient.

Story shears are the lateral loads applied above each story cut (point loads and global
distributed loads); the loads of lateral cases must therefore be applied as global forces.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from struct_core import (
    AnalysisResult,
    CodeCheckResult,
    ElementResult,
    FrameStory,
    classify_frame_members,
    conversion_factor,
    frame_stories,
    node_elevation,
    parse_units,
    vertical_axis,
)
from struct_core.units import PRESETS

# ACI 318-25 Table 6.6.3.1.1(a): moments of inertia for elastic analysis at factored load level
ACI_CRACKED_INERTIA = {"column": 0.70, "brace": 0.70, "beam": 0.35}


def aci_cracked_stiffness_modifiers(model: Any) -> Dict[Any, float]:
    """Flexural stiffness multipliers per element: 0.70 Ig columns, 0.35 Ig beams (Table 6.6.3.1.1(a))."""
    return {eid: ACI_CRACKED_INERTIA[role] for eid, role in classify_frame_members(model).items()}


def _project_units(project: Any) -> Any:
    units = getattr(project, "units", None) or getattr(getattr(project, "metadata", None), "units", None)
    return parse_units(units or "kN, m")


def _horizontal(axis: str) -> Tuple[str, ...]:
    return ("x", "y") if axis == "z" else ("x",)


def classify_lateral_cases(model: Any, case_results: Dict[str, AnalysisResult]) -> Dict[str, str]:
    """
    Lateral load cases and their direction ('x' or 'y'), from the support reactions: a case is
    lateral when its horizontal reaction resultant exceeds its vertical reaction resultant.
    """
    axis = vertical_axis(model)
    lateral: Dict[str, str] = {}
    for cid, res in case_results.items():
        reactions = res.node_reactions
        if not reactions:
            continue
        sums = {c: abs(sum(getattr(r, f"f{c}") for r in reactions)) for c in ("x", "y", "z")}
        vertical = sums[axis]
        horizontal = {c: sums[c] for c in _horizontal(axis)}
        direction = max(horizontal, key=horizontal.get)
        if horizontal[direction] > vertical and horizontal[direction] > 0.0:
            lateral[cid] = direction
    return lateral


# -----------------------------------------------------------------------------
# Story quantities
# -----------------------------------------------------------------------------
def _column_ends(model: Any, eid: Any) -> Tuple[Any, Any]:
    """(bottom node id, top node id) of a column."""
    nodes = {n.id: n for n in model.nodes}
    el = next(e for e in model.elements if e.id == eid)
    zi = node_elevation(model, nodes[el.start_node])
    zj = node_elevation(model, nodes[el.end_node])
    return (el.start_node, el.end_node) if zi <= zj else (el.end_node, el.start_node)


def _displacement(res: AnalysisResult, nid: Any, direction: str) -> float:
    nr = res.node_results.get(nid)
    if nr is None:
        nr = res.node_results.get(str(nid))
    if nr is None or nr.displacement is None:
        raise ValueError(f"Node {nid!r} has no displacement in result {res.analysis_case_id!r}.")
    return float(getattr(nr.displacement, f"u{direction}"))


def story_drifts(model: Any, res: AnalysisResult, story: FrameStory, direction: str) -> List[float]:
    """Relative lateral displacement (top minus bottom) of every column line in a story."""
    drifts = []
    for eid in story.column_ids:
        bottom, top = _column_ends(model, eid)
        drifts.append(_displacement(res, top, direction) - _displacement(res, bottom, direction))
    return drifts


def story_shear(model: Any, load_case: Any, story: FrameStory, direction: str) -> float:
    """
    Lateral shear in a story from the loads of ``load_case`` applied above the story mid-height.

    Point loads at or above mid-height count fully; global distributed loads count by the part
    of the member length above mid-height. Local-axis distributed loads are not supported.
    """
    nodes = {n.id: n for n in model.nodes}
    elements = {e.id: e for e in model.elements}
    cut = 0.5 * (story.bottom + story.top)
    total = 0.0
    for pl in load_case.point_loads:
        if node_elevation(model, nodes[pl.node_id]) >= cut - 1e-9:
            total += getattr(pl, f"f{direction}")
    for dl in load_case.element_loads:
        w_start = getattr(dl, f"w{direction}")
        w_end = getattr(dl, f"w{direction}_end", w_start)
        if w_start == 0.0 and w_end == 0.0:
            continue
        if dl.is_local:
            raise ValueError(
                f"Story shear of case {load_case.id!r} needs global lateral loads; element "
                f"{dl.element_id!r} carries a local-axis distributed load."
            )
        el = elements[dl.element_id]
        n_i, n_j = nodes[el.start_node], nodes[el.end_node]
        z_i, z_j = node_elevation(model, n_i), node_elevation(model, n_j)
        length = math.dist(
            (n_i.x, n_i.y, getattr(n_i, "z", 0.0)), (n_j.x, n_j.y, getattr(n_j, "z", 0.0))
        )
        xi1 = getattr(dl, "start", 0.0)
        xi2 = getattr(dl, "end", 1.0)
        z1 = z_i + xi1 * (z_j - z_i)
        z2 = z_i + xi2 * (z_j - z_i)
        z_lo, z_hi = min(z1, z2), max(z1, z2)
        load_len = length * (xi2 - xi1)
        if z_hi - z_lo <= 1e-12:
            fraction = 1.0 if z_lo >= cut - 1e-9 else 0.0
            avg_w = 0.5 * (w_start + w_end)
            total += avg_w * load_len * fraction
        else:
            z_a = max(cut, z_lo)
            if z_a < z_hi:
                t_a = (z_a - z_lo) / (z_hi - z_lo)
                w_lo = w_start if z1 <= z2 else w_end
                w_hi = w_end if z1 <= z2 else w_start
                w_cut = w_lo + t_a * (w_hi - w_lo)
                w_trap = 0.5 * (w_cut + w_hi)
                total += w_trap * load_len * (1.0 - t_a)

    for epl in getattr(load_case, "element_point_loads", []):
        f = getattr(epl, f"f{direction}")
        if f == 0.0:
            continue
        if getattr(epl, "is_local", False):
            raise ValueError(
                f"Story shear of case {load_case.id!r} needs global lateral loads; element point load on "
                f"{epl.element_id!r} is local-axis."
            )
        el = elements[epl.element_id]
        n_i, n_j = nodes[el.start_node], nodes[el.end_node]
        z_i, z_j = node_elevation(model, n_i), node_elevation(model, n_j)
        xi = getattr(epl, "position", 0.5)
        z_pl = z_i + xi * (z_j - z_i)
        if z_pl >= cut - 1e-9:
            total += f
    return abs(total)


def _station_p(er: ElementResult, station: float = 0.5) -> float:
    return min(er.forces, key=lambda s: abs(s.station - station)).P


def story_axial_load(
    stories_cols: Sequence[Any], case_results: Dict[str, AnalysisResult], factors: Dict[str, float]
) -> float:
    """Total factored compression sum(Pu) carried by the columns (mid-height axial force)."""
    total = 0.0
    for eid in stories_cols:
        p = 0.0
        for cid, fac in factors.items():
            res = case_results.get(cid)
            if res is None or eid not in res.element_results:
                continue
            p += fac * _station_p(res.element_results[eid])
        total += -p  # solver axial force is tension positive
    return total


# -----------------------------------------------------------------------------
# ACI 318 Section 6.6.4: moment magnification
# -----------------------------------------------------------------------------
def _scaled(er: ElementResult, factor: float) -> ElementResult:
    """Element result with shears and moments multiplied by ``factor`` (axial and torsion kept)."""
    stations = [
        s.model_copy(update={"V2": s.V2 * factor, "V3": s.V3 * factor, "M2": s.M2 * factor, "M3": s.M3 * factor})
        for s in er.forces
    ]
    return er.model_copy(update={"forces": stations, "combination_forces": {}})


def _section_geometry(model: Any, eid: Any) -> Tuple[float, float, float, float]:
    """(I3, I2, depth along local 2, width along local 3) of an element's section."""
    el = next(e for e in model.elements if e.id == eid)
    sec = next(s for s in model.sections if s.id == el.section_id)
    if sec.type == "rectangular":
        return sec.b * sec.h ** 3 / 12.0, sec.h * sec.b ** 3 / 12.0, sec.h, sec.b
    if sec.type == "circular":
        i = math.pi * sec.d ** 4 / 64.0
        return i, i, sec.d, sec.d
    if sec.type == "general":
        depth = math.sqrt(12.0 * sec.I3 / sec.A)
        width = math.sqrt(12.0 * sec.I2 / sec.A) if sec.I2 > 0.0 else depth
        return sec.I3, sec.I2 or sec.I3, depth, width
    raise ValueError(f"Unsupported section type {sec.type!r} for slenderness.")


def _elastic_modulus(model: Any, eid: Any) -> float:
    el = next(e for e in model.elements if e.id == eid)
    mat = model.material_of(el)
    return float(mat.E)


def _clear_height(model: Any, eid: Any, beam_depths_at_node: Dict[Any, float]) -> float:
    """Column length minus the depth of the deepest beam framing into its top joint."""
    bottom, top = _column_ends(model, eid)
    nodes = {n.id: n for n in model.nodes}
    length = node_elevation(model, nodes[top]) - node_elevation(model, nodes[bottom])
    return max(length - beam_depths_at_node.get(top, 0.0), 0.5 * length)


class SecondOrderReport:
    """Story sway magnifiers and the code checks of the slenderness evaluation."""

    def __init__(self) -> None:
        self.sway: Dict[Tuple[int, str], Dict[str, float]] = {}
        self.checks: List[CodeCheckResult] = []

    def as_dict(self) -> Dict[str, Any]:
        return {
            "stories": [
                {"story": s + 1, "direction": d, **vals} for (s, d), vals in sorted(self.sway.items())
            ],
            "checks": [c.model_dump() for c in self.checks],
        }


def magnify_sway_cases(
    project: Any,
    case_results: Dict[str, AnalysisResult],
    combinations: Sequence[Any],
    lateral_cases: Dict[str, str],
    flexibility: Dict[Tuple[int, str], float],
    report: SecondOrderReport,
    sign_less_cases: Iterable[str] = (),
) -> Dict[str, AnalysisResult]:
    """
    Multiply the shears and moments of the lateral (sway) cases by the story magnifier
    delta_s = 1/(1 - Q) (ACI 318 Eq. 6.6.4.6.2a). Columns use the delta_s of their story; beams use
    the larger delta_s of the stories meeting at their level (§6.6.4.6.3).

    ``flexibility[(story, direction)]`` is delta_o / Vus of the story. sum(Pu) is the largest story
    compression among the combinations that contain a lateral case of that direction; sign-less
    (spectral) cases do not contribute to it.
    """
    sign_less = set(sign_less_cases)
    from aci318.strength.slenderness import SWAY_MAGNIFIER_EQ_A_LIMIT, sway_magnifier

    model = project.model
    stories = frame_stories(model)
    roles = classify_frame_members(model)
    nodes = {n.id: n for n in model.nodes}
    magnified = dict(case_results)

    for direction in sorted(set(lateral_cases.values())):
        combos = [
            {f.load_case_id: float(f.factor) for f in c.factors if f.load_case_id not in sign_less}
            for c in combinations
            if any(lateral_cases.get(f.load_case_id) == direction for f in c.factors)
        ]
        if not combos:
            continue
        deltas: Dict[int, float] = {}
        for story in stories:
            sum_pu = max(story_axial_load(story.column_ids, case_results, f) for f in combos)
            if (story.index, direction) not in flexibility:
                raise ValueError(
                    f"No lateral load reaches story {story.index + 1} in direction {direction!r}; "
                    "its stability index Q cannot be computed."
                )
            q = sum_pu * flexibility[(story.index, direction)] / story.height
            delta_s = sway_magnifier(q)
            deltas[story.index] = delta_s
            report.sway[(story.index, direction)] = {"sum_pu": sum_pu, "Q": q, "delta_s": delta_s}
            report.checks.append(
                CodeCheckResult(
                    clause="ACI 318-25 §6.6.4.6.2",
                    description=f"Story {story.index + 1} ({direction}): sway magnifier delta_s = 1/(1-Q) within Eq. (a) limit 1.5",
                    demand=delta_s if math.isfinite(delta_s) else 1e9,
                    capacity=SWAY_MAGNIFIER_EQ_A_LIMIT,
                    dcr=(delta_s / SWAY_MAGNIFIER_EQ_A_LIMIT) if math.isfinite(delta_s) else 1e9,
                    passed=bool(delta_s <= SWAY_MAGNIFIER_EQ_A_LIMIT),
                    messages=[f"Q = {q:.4f}" + (" (nonsway, Q <= 0.05)" if q <= 0.05 else "")],
                )
            )

        story_of_column = {eid: s.index for s in stories for eid in s.column_ids}
        level_delta: Dict[float, float] = {}
        for s in stories:
            for z in (s.bottom, s.top):
                key = round(z, 6)
                level_delta[key] = max(level_delta.get(key, 1.0), deltas[s.index])

        for cid, d in lateral_cases.items():
            if d != direction:
                continue
            res = case_results[cid]
            new_elements = {}
            for eid, er in res.element_results.items():
                if roles.get(eid) == "column" and eid in story_of_column:
                    factor = deltas[story_of_column[eid]]
                elif roles.get(eid) == "beam":
                    el = next(e for e in model.elements if e.id == eid)
                    factor = level_delta.get(round(node_elevation(model, nodes[el.start_node]), 6), 1.0)
                else:
                    factor = 1.0
                new_elements[eid] = _scaled(er, factor if math.isfinite(factor) else 1.0)
            magnified[cid] = res.model_copy(update={"element_results": new_elements})
    return magnified


def magnify_member_curvature(
    project: Any,
    combination_results: Sequence[AnalysisResult],
    first_order_results: Dict[str, AnalysisResult],
    case_results: Dict[str, AnalysisResult],
    combinations: Sequence[Any],
    dead_case_ids: Iterable[str],
    report: SecondOrderReport,
) -> List[AnalysisResult]:
    """
    Apply Mc = delta * max(M2, M2,min) to every column of every combination (ACI 318 §6.6.4.5,
    required for sway frames by §6.6.4.6.4), and check §6.6.4.5.2 (Pu < 0.75 Pc) and §6.2.5.3
    (second-order end moments <= 1.4 first-order end moments).

    k = 1.0, (EI)eff = 0.4 Ec Ig / (1 + beta_dns) with beta_dns the dead-load share of the column
    axial force, Cm from the (sway-magnified) end moments. M2,min is applied about each axis.
    """
    from aci318.strength import slenderness as sl

    model = project.model
    units = _project_units(project)
    to_in = conversion_factor(units, PRESETS["kip, in, s, °F"], "length")
    roles = classify_frame_members(model)
    nodes = {n.id: n for n in model.nodes}
    stories = frame_stories(model)
    story_of_column = {eid: s.index for s in stories for eid in s.column_ids}
    dead_ids = set(dead_case_ids)
    comb_factors = {c.id: {f.load_case_id: float(f.factor) for f in c.factors} for c in combinations}

    beam_depths: Dict[Any, float] = {}
    for el in model.elements:
        if roles.get(el.id) == "beam":
            depth = _section_geometry(model, el.id)[2]
            for nid in (el.start_node, el.end_node):
                beam_depths[nid] = max(beam_depths.get(nid, 0.0), depth)

    worst_pc: Dict[int, Tuple[float, Any]] = {}
    worst_ratio: Dict[int, Tuple[float, Any]] = {}
    out: List[AnalysisResult] = []
    for comb_res in combination_results:
        factors = comb_factors[comb_res.analysis_case_id]
        first = first_order_results[comb_res.analysis_case_id]
        new_elements = dict(comb_res.element_results)
        for eid, er in comb_res.element_results.items():
            if roles.get(eid) != "column" or eid not in story_of_column:
                continue
            i3, i2, depth, width = _section_geometry(model, eid)
            ec = _elastic_modulus(model, eid)
            lu = _clear_height(model, eid, beam_depths)
            pu = -_station_p(er)
            p_dead = -sum(
                fac * _station_p(case_results[cid].element_results[eid])
                for cid, fac in factors.items()
                if cid in dead_ids and cid in case_results
            )
            beta_dns = min(max(p_dead / pu, 0.0), 1.0) if pu > 0.0 else 0.0
            new_stations = list(er.forces)
            for comp, inertia, h in (("M3", i3, depth), ("M2", i2, width)):
                end_i, end_j = getattr(er.forces[0], comp), getattr(er.forces[-1], comp)
                m2 = max(abs(end_i), abs(end_j))
                pc = sl.critical_buckling_load(sl.effective_stiffness(ec, inertia, beta_dns), 1.0, lu)
                if pu > 0.0:
                    ratio_pc = pu / (0.75 * pc)
                    s_idx = story_of_column[eid]
                    if ratio_pc > worst_pc.get(s_idx, (0.0, None))[0]:
                        worst_pc[s_idx] = (ratio_pc, eid)
                m2_min = max(pu, 0.0) * sl.minimum_eccentricity(h * to_in) / to_in
                cm = sl.equivalent_moment_factor(end_i, end_j) if m2 >= m2_min else 1.0
                delta = sl.member_magnifier(cm, pu, pc)
                if not math.isfinite(delta):
                    continue  # reported through the Pu < 0.75 Pc check
                m_design = delta * max(m2, m2_min)
                if m2 > 0.0:
                    factor = m_design / m2
                    new_stations = [s.model_copy(update={comp: getattr(s, comp) * factor}) for s in new_stations]
                elif m_design > 0.0:
                    new_stations = [s.model_copy(update={comp: m_design}) for s in new_stations]
                # §6.2.5.3 ratio of second-order to first-order end moments (M2,min excluded)
                first_er = first.element_results[eid]
                m1st = max(abs(getattr(first_er.forces[0], comp)), abs(getattr(first_er.forces[-1], comp)))
                if m1st > 1e-9 and m2 > 0.0:
                    ratio = delta * m2 / m1st
                    s_idx = story_of_column[eid]
                    if ratio > worst_ratio.get(s_idx, (0.0, None))[0]:
                        worst_ratio[s_idx] = (ratio, eid)
            new_elements[eid] = er.model_copy(update={"forces": new_stations})
        out.append(comb_res.model_copy(update={"element_results": new_elements}))

    for s in stories:
        ratio_pc, eid = worst_pc.get(s.index, (0.0, None))
        report.checks.append(
            CodeCheckResult(
                clause="ACI 318-25 §6.6.4.5.2",
                description=f"Story {s.index + 1}: column stability Pu < 0.75 Pc (governing column {eid})",
                demand=ratio_pc, capacity=1.0, dcr=ratio_pc, passed=bool(ratio_pc < 1.0),
            )
        )
        ratio, eid = worst_ratio.get(s.index, (1.0, None))
        report.checks.append(
            CodeCheckResult(
                clause="ACI 318-25 §6.2.5.3",
                description=f"Story {s.index + 1}: second-order end moment <= 1.4 x first-order (governing column {eid})",
                demand=ratio, capacity=sl.SECOND_ORDER_MOMENT_RATIO_LIMIT,
                dcr=ratio / sl.SECOND_ORDER_MOMENT_RATIO_LIMIT,
                passed=bool(ratio <= sl.SECOND_ORDER_MOMENT_RATIO_LIMIT),
            )
        )
    return out


def static_flexibility(
    project: Any,
    case_results: Dict[str, AnalysisResult],
    lateral_cases: Dict[str, str],
) -> Dict[Tuple[int, str], float]:
    """Story flexibility delta_o / Vus from the lateral load cases (mean drift of the column lines)."""
    model = project.model
    cases = {lc.id: lc for lc in model.load_cases}
    flexibility: Dict[Tuple[int, str], float] = {}
    for cid, direction in lateral_cases.items():
        res = case_results[cid]
        for story in frame_stories(model):
            v = story_shear(model, cases[cid], story, direction)
            if v <= 0.0:
                continue
            drift = sum(story_drifts(model, res, story, direction)) / len(story.column_ids)
            key = (story.index, direction)
            flexibility[key] = max(flexibility.get(key, 0.0), abs(drift) / v)
    return flexibility


def pattern_flexibility(
    project: Any,
    solve_case: Callable[[Any], AnalysisResult],
    directions: Iterable[str],
) -> Dict[Tuple[int, str], float]:
    """
    Story flexibility delta_o / Vus from an auxiliary lateral pattern (story forces proportional to
    height, applied equally to the column-top nodes of each level). Used where the lateral effect is
    a response spectrum rather than a static load case. ``solve_case`` analyses a LoadCase on the
    cracked model and returns its AnalysisResult.
    """
    from struct_core import LoadCase, PointLoad

    model = project.model
    stories = frame_stories(model)
    flexibility: Dict[Tuple[int, str], float] = {}
    for direction in directions:
        point_loads = []
        for story in stories:
            tops = sorted({_column_ends(model, eid)[1] for eid in story.column_ids}, key=str)
            for nid in tops:
                point_loads.append(PointLoad(node_id=nid, **{f"f{direction}": story.top / len(tops)}))
        case = LoadCase(id=f"_stability_pattern_{direction}", point_loads=point_loads)
        res = solve_case(case)
        for story in stories:
            v = story_shear(model, case, story, direction)
            drift = sum(story_drifts(model, res, story, direction)) / len(story.column_ids)
            flexibility[(story.index, direction)] = abs(drift) / v
    return flexibility


# -----------------------------------------------------------------------------
# ASCE 7-22 Sections 12.12.1 and 12.8.7
# -----------------------------------------------------------------------------
def gravity_story_load(
    project: Any, case_results: Dict[str, AnalysisResult], dead_ids: Sequence[str], live_ids: Sequence[str]
) -> Dict[int, float]:
    """
    Px per story: column compression under 1.0D + 0.4L. ASCE 7-22 §12.8.6.1 expected gravity is at
    least 1.0D + 0.5L with L = 0.4 L0 (0.8 L0 above 100 psf); 0.5 x 0.8 = 0.4 covers both.
    """
    factors = {cid: 1.0 for cid in dead_ids}
    factors.update({cid: 0.4 for cid in live_ids})
    return {
        s.index: story_axial_load(s.column_ids, case_results, factors) for s in frame_stories(project.model)
    }


def drift_and_stability_checks(
    project: Any,
    drifts: Dict[Tuple[int, str], Tuple[float, float]],
    shears: Dict[Tuple[int, str], float],
    px: Dict[int, float],
) -> List[CodeCheckResult]:
    """
    Story drift (ASCE 7-22 §12.12.1, Table 12.12-1, §12.12.1.1) and stability coefficient
    (§12.8.7) checks.

    ``drifts[(story, dir)]`` = (largest elastic story drift of the column lines, mean elastic story
    drift) under the seismic design forces; ``shears[(story, dir)]`` = seismic design story shear Vx.
    Where 0.10 < theta <= theta_max, the drift is multiplied by 1/(1 - theta).
    """
    from struct_load import allowable_story_drift, max_stability_coefficient, stability_coefficient

    basis = project.seismic_basis
    model = project.model
    stories = {s.index: s for s in frame_stories(model)}
    theta_max = max_stability_coefficient(basis.cd)
    checks: List[CodeCheckResult] = []
    for (s_idx, direction), (delta_max, delta_mean) in sorted(drifts.items()):
        story = stories[s_idx]
        vx = shears[(s_idx, direction)]
        theta = stability_coefficient(px[s_idx], delta_mean, vx, story.height)
        amplification = 1.0 / (1.0 - theta) if 0.10 < theta < 1.0 else 1.0
        design_drift = basis.cd * abs(delta_max) / basis.importance_factor * amplification
        allowable = allowable_story_drift(
            story.height,
            risk_category=basis.risk_category,
            system_type=basis.system,
            seismic_design_category=basis.seismic_design_category,
            redundancy_factor=basis.redundancy_factor,
        )
        checks.append(
            CodeCheckResult(
                clause="ASCE 7-22 §12.12.1",
                description=f"Story {s_idx + 1} ({direction}): design story drift Cd*delta_xe/Ie <= Delta_a",
                demand=design_drift, capacity=allowable, dcr=design_drift / allowable,
                passed=bool(design_drift <= allowable),
                messages=[f"drift ratio {design_drift / story.height:.4f}" + (
                    f", P-delta amplification 1/(1-theta) = {amplification:.3f}" if amplification > 1.0 else "")],
            )
        )
        checks.append(
            CodeCheckResult(
                clause="ASCE 7-22 §12.8.7",
                description=f"Story {s_idx + 1} ({direction}): stability coefficient theta <= theta_max",
                demand=theta, capacity=theta_max, dcr=theta / theta_max, passed=bool(theta <= theta_max),
            )
        )
    return checks
