"""
Test runner for StateTransitionTracker.

Run as a script (prints a full report + summary table, exits non-zero on failure):

    python test_state_manager.py                      # every case
    python test_state_manager.py --only spotify 2     # cases whose name contains "spotify", plus case #2
    python test_state_manager.py --no-debug           # skip the debug artifacts (faster)
    python test_state_manager.py --clean --verbose    # wipe old debug output, print full diagnostics

...or with pytest (one test per case; cases whose image files are missing are skipped):

    pytest test_state_manager.py -v

Case format (every `expect*` key is optional; a case with no expectations is reported as INFO):

    {
        "name": "Spotify - Context Menu Opened",
        "base_img": "path/to/frame_t.png",
        "new_img": "path/to/frame_t1.png",
        "layout_path": "path/to/layout_of_frame_t.json",
        "action": {"type": "click", "x": 0.88, "y": 0.12},     # x/y in [0,1] = normalized; else original px

        "expected_type": "MODAL_OVERLAY",                       # str, or a list of acceptable types
        "expected_changed_containers": ["main_content"],        # exact set of ids to re-map
        "expected_motion": {                                    # container id -> motion type / spec
            "main_a": "TRANSLATION",
            "queue": {"type": "TRANSLATION", "delta": (0, -27), "tol": 3},   # delta in 854x480 px
        },
        "expect_crop": True,                                    # True: crop required, False: must be None
        "expected_crop_bbox": {"x": 0.3, "y": 0.2, "w": 0.4, "h": 0.5},   # normalized, true popup rect
        "min_crop_coverage": 0.8,                               # share of expected bbox inside the crop
        "min_crop_iou": 0.3,
        "max_unaccounted_ratio": 0.2,
        "max_ms": 1500,                                         # pipeline runtime budget
    }

Every case also writes a staged debug folder (see StateDebugger): open
`<debug_dir>/08_result/final_overlay.png` first, then walk 01_input ... 07_components to see where a
wrong verdict came from.
"""
import argparse
import json
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from services.mapper.state_transition_tracker import StateTransitionTracker, TransitionResult

DEBUG_ROOT = Path("tests/state_transition_tracker/debug_outputs")

TYPE_EMOJIS = {
    "NO_CHANGE": "⚪",
    "IN_PLACE_CONTROL": "🟢",
    "REFLOW_UPDATE": "🔵",
    "CONTENT_REPLACED": "🟠",
    "MODAL_OVERLAY": "🟣",
    "GLOBAL_NAVIGATION": "🛑",
}
STATUS_ICONS = {"PASS": "✅", "FAIL": "❌", "ERROR": "💥", "SKIP": "⏭️ ", "INFO": "ℹ️ "}


# ======================================================================================
# Test cases
# ======================================================================================
TEST_CASES: List[Dict[str, Any]] = [
    {
        "name": "Spotify 1 - Search Bar Clicked (Recent Searches Dropdown)",
        "base_img": "tests/layout_mapper/inputs/spotify_test.png",
        "new_img": "tests/state_transition_tracker/inputs/spotify_test_1.png",
        "layout_path": "tests/layout_mapper/outputs/spotify_test.json",
        "action": {"type": "click", "x": 0.480, "y": 0.065},
        "expected_type": "MODAL_OVERLAY",
        "expect_crop": True,
        "expected_crop_bbox": {"x": 0.386, "y": 0.095, "w": 0.257, "h": 0.374},
        "min_crop_coverage": 0.80,
        "min_crop_iou": 0.30,
    },
    {
        "name": "Spotify 2 - Main Container Scrolled Down",
        "base_img": "tests/layout_mapper/inputs/spotify_test.png",
        "new_img": "tests/state_transition_tracker/inputs/spotify_test_2.png",
        "layout_path": "tests/layout_mapper/outputs/spotify_test.json",
        "action": {"type": "scroll", "x": 0.500, "y": 0.500},
        "expected_type": "REFLOW_UPDATE",
        "expect_crop": True,
    },
    {
        "name": "Spotify 3 - Profile Icon Clicked (Dropdown Menu)",
        "base_img": "tests/layout_mapper/inputs/spotify_test.png",
        "new_img": "tests/state_transition_tracker/inputs/spotify_test_3.png",
        "layout_path": "tests/layout_mapper/outputs/spotify_test.json",
        "action": {"type": "click", "x": 0.981, "y": 0.065},
        "expected_type": "MODAL_OVERLAY",
        "expect_crop": True,
        "expected_crop_bbox": {"x": 0.816, "y": 0.096, "w": 0.179, "h": 0.385},
        "min_crop_coverage": 0.80,
        "min_crop_iou": 0.30,
    },
    {
        "name": "Spotify 4 - Liked Songs Clicked in Left Sidebar (Feed Replaced)",
        "base_img": "tests/layout_mapper/inputs/spotify_test.png",
        "new_img": "tests/state_transition_tracker/inputs/spotify_test_4.png",
        "layout_path": "tests/layout_mapper/outputs/spotify_test.json",
        "action": {"type": "click", "x": 0.065, "y": 0.268},
        "expected_type": ["CONTENT_REPLACED", "MODAL_OVERLAY"],
        "expect_crop": True,
    },
    {
        "name": "Spotify 5 - Hovered Over Bottom Timestamp Bar (Scrubber & Tooltip)",
        "base_img": "tests/layout_mapper/inputs/spotify_test.png",
        "new_img": "tests/state_transition_tracker/inputs/spotify_test_5.png",
        "layout_path": "tests/layout_mapper/outputs/spotify_test.json",
        "action": {"type": "hover", "x": 0.402, "y": 0.976},
        "expected_type": "IN_PLACE_CONTROL",
        "expect_crop": False,
    },
    {
        "name": "Spotify 6 - Expand Now Playing View Clicked (Full View Takeover)",
        "base_img": "tests/layout_mapper/inputs/spotify_test.png",
        "new_img": "tests/state_transition_tracker/inputs/spotify_test_6.png",
        "layout_path": "tests/layout_mapper/outputs/spotify_test.json",
        "action": {"type": "click", "x": 0.978, "y": 0.125},
        "expected_type": "GLOBAL_NAVIGATION",
        "expect_crop": False,
    },
    {
        "name": "Spotify 7 - Podcasts Tab Clicked in Main Content",
        "base_img": "tests/layout_mapper/inputs/spotify_test.png",
        "new_img": "tests/state_transition_tracker/inputs/spotify_test_7.png",
        "layout_path": "tests/layout_mapper/outputs/spotify_test.json",
        "action": {"type": "click", "x": 0.306, "y": 0.128},
        "expected_type": ["CONTENT_REPLACED", "MODAL_OVERLAY"],
        "expect_crop": True,
    },
    {
        "name": "Spotify 8 - Hovered Over 'Chill Mix' Album Card (Play Button Overlay)",
        "base_img": "tests/layout_mapper/inputs/spotify_test.png",
        "new_img": "tests/state_transition_tracker/inputs/spotify_test_8.png",
        "layout_path": "tests/layout_mapper/outputs/spotify_test.json",
        "action": {"type": "hover", "x": 0.495, "y": 0.485},
        "expected_type": "IN_PLACE_CONTROL",
        "expect_crop": False,
    },
]


# ======================================================================================
# Helpers
# ======================================================================================
@dataclass
class CaseOutcome:
    index: int
    name: str
    status: str                                   # PASS | FAIL | ERROR | SKIP | INFO
    failures: List[str] = field(default_factory=list)
    result: Optional[TransitionResult] = None
    wall_ms: float = 0.0
    debug_dir: Optional[str] = None
    detail: str = ""


def _slug(text: str) -> str:
    keep = "".join(ch.lower() if ch.isalnum() else "_" for ch in text)
    while "__" in keep:
        keep = keep.replace("__", "_")
    return keep.strip("_")[:60] or "case"


def _bbox_px(b: Dict[str, float]) -> List[float]:
    return [b["x"], b["y"], b["w"], b["h"]]


def _intersection(a: List[float], b: List[float]) -> float:
    ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    return ix * iy


def bbox_iou(a: Dict[str, float], b: Dict[str, float]) -> float:
    ba, bb = _bbox_px(a), _bbox_px(b)
    inter = _intersection(ba, bb)
    union = ba[2] * ba[3] + bb[2] * bb[3] - inter
    return inter / union if union > 0 else 0.0


def bbox_coverage(expected: Dict[str, float], crop: Dict[str, float]) -> float:
    """Share of the expected rect that lies inside the crop (the LLM must SEE the whole popup)."""
    be, bc = _bbox_px(expected), _bbox_px(crop)
    area = be[2] * be[3]
    return _intersection(be, bc) / area if area > 0 else 0.0


def evaluate_expectations(result: TransitionResult, tc: Dict[str, Any]) -> List[str]:
    """Returns a list of human-readable failures (empty list = all expectations met)."""
    failures: List[str] = []

    expected = tc.get("expected_type")
    if expected is not None:
        allowed = [expected] if isinstance(expected, str) else list(expected)
        if result.transition_type not in allowed:
            failures.append(f"transition_type: expected {allowed}, got {result.transition_type}")

    if "expected_changed_containers" in tc:
        want, got = set(tc["expected_changed_containers"]), set(result.changed_containers)
        if want != got:
            failures.append(f"changed_containers: expected {sorted(want)}, got {sorted(got)}")

    if "expected_motion" in tc:
        moved = {c["id"]: c for c in result.updated_containers if c.get("motion_type")}
        for cid, spec in tc["expected_motion"].items():
            spec = {"type": spec} if isinstance(spec, str) else dict(spec)
            c = moved.get(cid)
            if c is None:
                failures.append(f"motion[{cid}]: expected {spec}, but that container has no motion")
                continue
            if "type" in spec and c["motion_type"] != spec["type"]:
                failures.append(f"motion[{cid}]: expected type {spec['type']}, got {c['motion_type']}")
            if "delta" in spec:
                tol = spec.get("tol", 3)
                dx, dy = c["delta"]
                if abs(dx - spec["delta"][0]) > tol or abs(dy - spec["delta"][1]) > tol:
                    failures.append(f"motion[{cid}]: expected delta {tuple(spec['delta'])} +/-{tol}, got {(dx, dy)}")

    crop = result.target_crop_bbox
    if "expect_crop" in tc:
        if tc["expect_crop"] and crop is None:
            failures.append("target_crop_bbox: expected a crop, got None")
        if not tc["expect_crop"] and crop is not None:
            failures.append(f"target_crop_bbox: expected no crop, got {crop}")
    if "expected_crop_bbox" in tc:
        if crop is None:
            failures.append("target_crop_bbox: expected a crop around the target, got None")
        else:
            cov = bbox_coverage(tc["expected_crop_bbox"], crop)
            iou = bbox_iou(tc["expected_crop_bbox"], crop)
            if cov < tc.get("min_crop_coverage", 0.8):
                failures.append(f"crop coverage {cov:.2f} < {tc.get('min_crop_coverage', 0.8)} (popup cut off)")
            if iou < tc.get("min_crop_iou", 0.3):
                failures.append(f"crop IoU {iou:.2f} < {tc.get('min_crop_iou', 0.3)} (crop too loose)")

    if "max_unaccounted_ratio" in tc and result.unaccounted_ratio > tc["max_unaccounted_ratio"]:
        failures.append(f"unaccounted_ratio {result.unaccounted_ratio:.3f} > {tc['max_unaccounted_ratio']}")

    if "max_ms" in tc:
        ms = result.diagnostics.get("timings_ms", {}).get("pipeline_total")
        if ms is not None and ms > tc["max_ms"]:
            failures.append(f"pipeline runtime {ms:.0f}ms > budget {tc['max_ms']}ms")

    return failures


def _has_expectations(tc: Dict[str, Any]) -> bool:
    keys = ("expected_type", "expected_changed_containers", "expected_motion", "expect_crop",
            "expected_crop_bbox", "max_unaccounted_ratio", "max_ms")
    return any(k in tc for k in keys)


def _stage_listing(debug_dir: Path) -> List[str]:
    if not debug_dir.exists():
        return []
    lines = []
    for stage in sorted(p for p in debug_dir.iterdir() if p.is_dir()):
        n = sum(1 for f in stage.rglob("*") if f.is_file())
        lines.append(f"{stage.name:<16} {n:>3} files")
    return lines


def _print_result(result: TransitionResult, verbose: bool) -> None:
    emoji = TYPE_EMOJIS.get(result.transition_type, "❓")
    print(f"{emoji} RESULT: {result.transition_type} (Confidence: {result.confidence * 100:.1f}%)")
    print(f"   ├─ Scaffold Intact:     {result.scaffold_intact}")
    print(f"   ├─ Unaccounted Surface: {result.unaccounted_ratio * 100:.2f}%")

    if result.changed_containers:
        print(f"   ├─ Changed Containers:  {result.changed_containers}  <-- re-map these")

    if result.target_crop_bbox:
        print(f"   ├─ Target Crop BBox:    {result.target_crop_bbox}  <-- Sent to LLM Updater")

    moved = [c for c in result.updated_containers if c.get("motion_type")]
    if moved:
        print(f"   ├─ Containers w/ motion: {len(moved)}")
        for c in moved:
            delta = c.get("delta") or c.get("scroll_delta", (0, 0))
            extra = f", Scale: {c['scale_factor']}" if c.get("scale_factor") else ""
            peers = f", peers={c['delta_peers']}" if c.get("delta_peers") else ""
            quality = f", q={c['motion_quality']}" if c.get("motion_quality") is not None else ""
            print(f"   │   ↳ [{c.get('id', '?')}] {c['motion_type']}, Delta: (dx={delta[0]}, dy={delta[1]}){extra}{peers}{quality}")
            if c.get("revealed_region"):
                print(f"   │       revealed region: {c['revealed_region']}")

    diag = result.diagnostics or {}
    if diag.get("reason"):
        print(f"   ├─ Reason:              {diag['reason']}")
    if diag.get("component_kinds"):
        print(f"   ├─ Residual components: {diag['component_kinds']}")
    timings = diag.get("timings_ms")
    if timings:
        ordered = ", ".join(f"{k}={v:.0f}" for k, v in timings.items() if k not in ("pipeline_total", "debug_io"))
        print(f"   ├─ Timings (ms):        total={timings.get('pipeline_total', 0):.0f}  [{ordered}]")
    if verbose:
        print(f"   └─ Diagnostics:         {json.dumps(diag, default=str)}")
    else:
        print("   └─ (use --verbose for the full diagnostics dict)")


# ======================================================================================
# Runner
# ======================================================================================
def run_case(
    tracker: StateTransitionTracker,
    tc: Dict[str, Any],
    index: int,
    total: int,
    debug_root: Optional[Path],
    verbose: bool = False,
) -> CaseOutcome:
    name = tc.get("name", f"Test Case {index}")
    action = tc.get("action", {"type": "click", "x": 0.5, "y": 0.5})
    base_img, new_img, layout_path = tc["base_img"], tc["new_img"], tc["layout_path"]

    print(f"--- [{index}/{total}] {name} ---")
    print(f"Action: {action}")

    outcome = CaseOutcome(index=index, name=name, status="ERROR")

    missing = [p for p in (base_img, new_img, layout_path) if not Path(p).exists()]
    if missing:
        outcome.status = "SKIP"
        outcome.detail = f"missing files: {missing}"
        print(f"⏭️  SKIP: missing required files: {missing}\n")
        return outcome

    try:
        with open(layout_path, "r", encoding="utf-8") as f:
            layout_map = json.load(f)
    except Exception as e:
        outcome.detail = f"layout JSON error: {e}"
        print(f"💥 ERROR: Failed to parse layout JSON '{layout_path}': {e}\n")
        return outcome

    debug_dir: Optional[Path] = None
    if debug_root is not None:
        debug_dir = debug_root / f"{index:02d}_{_slug(name)}"
        if debug_dir.exists():
            shutil.rmtree(debug_dir)
        outcome.debug_dir = str(debug_dir)

    t0 = time.perf_counter()
    try:
        result = tracker.track_transition(
            frame_t_path=base_img,
            frame_t1_path=new_img,
            action=action,
            layout_t=layout_map,
            debug_dir=str(debug_dir) if debug_dir else None,
        )
    except Exception as e:
        import traceback
        outcome.detail = f"{type(e).__name__}: {e}"
        print(f"💥 ERROR: Exception during transition tracking: {e}")
        print(traceback.format_exc())
        return outcome
    outcome.wall_ms = (time.perf_counter() - t0) * 1000.0
    outcome.result = result

    _print_result(result, verbose)

    if _has_expectations(tc):
        outcome.failures = evaluate_expectations(result, tc)
        outcome.status = "FAIL" if outcome.failures else "PASS"
    else:
        outcome.status = "INFO"

    print(f"{STATUS_ICONS[outcome.status]} {outcome.status}", end="")
    if outcome.failures:
        print()
        for f in outcome.failures:
            print(f"     ✗ {f}")
    else:
        print()

    if debug_dir is not None:
        print(f"📸 Debug artifacts: {debug_dir}")
        for line in _stage_listing(debug_dir):
            print(f"     {line}")
        print(f"     open first -> {debug_dir / '08_result' / 'final_overlay.png'}")
        errors = debug_dir / "debugger_errors.txt"
        if errors.exists():
            print(f"     ⚠️  a debug dump failed, see {errors}")
    print()
    return outcome


def _select(test_cases: List[Dict[str, Any]], only: Optional[List[str]]) -> List[int]:
    """Indices (0-based) of the cases to run. `only` items are 1-based numbers or name substrings."""
    if not only:
        return list(range(len(test_cases)))
    chosen = set()
    for token in only:
        if token.isdigit():
            chosen.add(int(token) - 1)
        else:
            chosen.update(i for i, tc in enumerate(test_cases) if token.lower() in tc.get("name", "").lower())
    return sorted(i for i in chosen if 0 <= i < len(test_cases))


def print_summary(outcomes: List[CaseOutcome]) -> None:
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for o in outcomes:
        got = o.result.transition_type if o.result else "-"
        ms = f"{o.result.diagnostics.get('timings_ms', {}).get('pipeline_total', 0):.0f}ms" if o.result else "-"
        print(f"{STATUS_ICONS[o.status]} [{o.index:>2}] {o.status:<5} {got:<18} {ms:>7}  {o.name}")
        for f in o.failures:
            print(f"          ✗ {f}")
        if o.status in ("ERROR", "SKIP") and o.detail:
            print(f"          {o.detail}")
    counts = {s: sum(1 for o in outcomes if o.status == s) for s in STATUS_ICONS}
    print("-" * 70)
    print("  ".join(f"{s}: {n}" for s, n in counts.items() if n))
    print("=" * 70)


def run_transition_tracker_tests(
    test_cases: List[Dict[str, Any]],
    debug: bool = True,
    debug_root: Path = DEBUG_ROOT,
    only: Optional[List[str]] = None,
    clean: bool = False,
    verbose: bool = False,
    tracker: Optional[StateTransitionTracker] = None,
) -> List[CaseOutcome]:
    """Runs the cases and returns one CaseOutcome each. Never raises for a failing case."""
    if debug and clean and debug_root.exists():
        shutil.rmtree(debug_root)
    if debug:
        debug_root.mkdir(parents=True, exist_ok=True)

    tracker = tracker or StateTransitionTracker()
    selected = _select(test_cases, only)

    print(f"\n{'=' * 70}")
    print("🚀 RUNNING STATE TRANSITION TRACKER TEST SUITE")
    print(f"{'=' * 70}\n")

    outcomes = [
        run_case(tracker, test_cases[i], i + 1, len(test_cases), debug_root if debug else None, verbose)
        for i in selected
    ]
    print_summary(outcomes)
    return outcomes


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="StateTransitionTracker test runner")
    parser.add_argument("--only", nargs="+", metavar="N|NAME", help="case numbers (1-based) or name substrings")
    parser.add_argument("--no-debug", action="store_true", help="do not write debug artifacts")
    parser.add_argument("--clean", action="store_true", help="delete the whole debug output folder first")
    parser.add_argument("--debug-dir", default=str(DEBUG_ROOT), help="root folder for debug artifacts")
    parser.add_argument("--verbose", action="store_true", help="print the full diagnostics dict")
    args = parser.parse_args(argv)

    outcomes = run_transition_tracker_tests(
        TEST_CASES,
        debug=not args.no_debug,
        debug_root=Path(args.debug_dir),
        only=args.only,
        clean=args.clean,
        verbose=args.verbose,
    )
    return 1 if any(o.status in ("FAIL", "ERROR") for o in outcomes) else 0


# ======================================================================================
# pytest integration (one test per case; only defined when pytest is importable)
# ======================================================================================
try:
    import pytest
except ImportError:  # plain-script usage needs no pytest
    pytest = None

if pytest is not None:
    _TRACKER = StateTransitionTracker()

    @pytest.mark.parametrize("case", TEST_CASES, ids=[_slug(tc.get("name", "case")) for tc in TEST_CASES])
    def test_transition_tracker(case):
        outcome = run_case(_TRACKER, case, 1, 1, DEBUG_ROOT)
        if outcome.status == "SKIP":
            pytest.skip(outcome.detail)
        assert outcome.status != "ERROR", outcome.detail
        assert not outcome.failures, "; ".join(outcome.failures)


if __name__ == "__main__":
    sys.exit(main())