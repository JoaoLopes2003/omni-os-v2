import time

import cv2
import numpy as np
from collections import Counter
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Tuple, Literal, Optional, Any

from services.mapper.state_debugger import StateDebugger


TransitionType = Literal[
    "NO_CHANGE",
    "IN_PLACE_CONTROL",
    "REFLOW_UPDATE",
    "CONTENT_REPLACED",
    "MODAL_OVERLAY",
    "GLOBAL_NAVIGATION"
]

BBox = List[int]  # [x, y, w, h] in target-resolution pixels


@dataclass
class TransitionResult:
    transition_type: TransitionType
    confidence: float  # Heuristic, derived from evidence margins (NOT a calibrated probability)
    scaffold_intact: bool
    updated_containers: List[Dict[str, Any]]
    target_crop_bbox: Optional[Dict[str, float]]  # Normalized bbox (0.0 to 1.0) for targeted LLM mapping
    unaccounted_ratio: float
    diagnostics: Dict[str, Any]
    # Ids of non-volatile containers whose content changed and must be re-mapped
    changed_containers: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class StateTransitionTracker:
    """
    Action-conditioned delta tracker for T -> T+1 frame transitions.

    Pipeline:
      1. Gain-compensated, per-channel diff (handles dim scrims and dark themes).
      2. Per-container motion estimation (static anchors, translation, internal scroll,
         single-gap expansion/contraction, volatile/dynamic containers).
      3. Motion-COMPENSATED residual: a changed pixel is "explained" only if T+1 really matches
         T shifted by a verified container motion. Containers are never blindly painted over,
         so overlays that appear on top of existing containers survive in the residual.
      4. Residual components are categorised (churn / reflow / in-place artifact /
         content replacement / new surface) and classified.

    Note on `scrollable`: the layout has no container hierarchy, so every child of a scrolling
    parent is flagged scrollable and the whole child translates when the parent scrolls. The
    attribute is therefore NOT used to decide between "internal scroll" and "translation".
    That is decided from pixels (static perimeter vs. moving core). It is only used as a weak
    tie-breaker when a vertical size change cannot be corroborated by a neighbouring container.
    """

    MOVING_TYPES = ("TRANSLATION", "INTERNAL_SCROLL", "EXPANSION", "CONTRACTION", "DYNAMIC_RESIZE")

    def __init__(
        self,
        target_resolution: Tuple[int, int] = (854, 480),
        tile_size: int = 32,
        variance_threshold: float = 8.0,
        match_threshold: float = 0.80,
        action_proximity_radius: int = 45,
        global_nav_threshold: float = 0.55,
        border_depth_ratio: float = 0.05,      # 5% of dimension by default
        max_border_depth_px: int = 8,          # Ceiling for large containers
        max_margin_fraction: float = 0.15,     # Never eat more than 15% per side (retains >= 70% core)
        # --- diff / noise ---
        diff_threshold: int = 6,               # Per-channel 0-255. Dark themes differ by only ~7 levels
        compensation_threshold: int = 10,      # Tolerance when verifying a motion hypothesis
        min_noise_area: int = 8,               # Connected components smaller than this (px) are noise
        max_caret_length: int = 16,            # Slivers <=2px wide and <= this long are carets/blinks
        # --- motion estimation ---
        static_ncc: float = 0.95,              # Tile unchanged at the same spot => static anchor
        ambiguity_margin: float = 0.05,        # Reject tiles whose 2nd-best match is this close to the best
        peak_exclusion_px: int = 6,
        expansion_min_px: int = 10,            # Min divergence to call a vertical expansion/contraction
        min_inlier_ratio: float = 0.60,
        min_explained_ratio: float = 0.50,     # A motion must explain at least this share of the container's diff
        # --- residual analysis ---
        merge_px: int = 5,                     # Closing kernel used to merge glyph clusters
        min_surface_size: Tuple[int, int] = (16, 9),
        min_surface_area_ratio: float = 0.002,  # ~820px at 854x480
        min_surface_solidity: float = 0.45,
        dense_surface_density: float = 0.60,   # Dense (solid fill) residual => popup, not replaced text
        replace_cover_ratio: float = 0.25,     # Union of a container's residual must cover this share of it
        replace_span_ratio: float = 0.60,      # ...and span this share of its height or width
        edge_similarity_threshold: float = 0.85,
        crop_padding_px: int = 8,
    ):
        self.target_res = target_resolution
        self.tile_size = tile_size
        self.variance_threshold = variance_threshold
        self.match_threshold = match_threshold
        self.action_radius = action_proximity_radius
        self.global_nav_threshold = global_nav_threshold

        self.border_ratio = border_depth_ratio
        self.max_border_px = max_border_depth_px
        self.max_margin_frac = max_margin_fraction

        self.diff_threshold = diff_threshold
        self.comp_threshold = compensation_threshold
        self.min_noise_area = min_noise_area
        self.max_caret_len = max_caret_length

        self.static_ncc = static_ncc
        self.ambiguity_margin = ambiguity_margin
        self.peak_exclusion_px = peak_exclusion_px
        self.expansion_min_px = expansion_min_px
        self.min_inlier_ratio = min_inlier_ratio
        self.min_explained_ratio = min_explained_ratio

        self.merge_px = merge_px
        self.min_surface_size = min_surface_size
        self.min_surface_area_ratio = min_surface_area_ratio
        self.min_surface_solidity = min_surface_solidity
        self.dense_surface_density = dense_surface_density
        self.replace_cover_ratio = replace_cover_ratio
        self.replace_span_ratio = replace_span_ratio
        self.edge_sim_threshold = edge_similarity_threshold
        self.crop_padding = crop_padding_px

    # =========================================================================
    # Primary Entry Point
    # =========================================================================
    def track_transition(
        self,
        frame_t_path: str,
        frame_t1_path: str,
        action: Dict[str, Any],
        layout_t: Dict[str, Any],
        debug_dir: Optional[str] = None
    ) -> TransitionResult:
        dbg = StateDebugger(debug_dir)  # complete no-op when debug_dir is None
        timings: Dict[str, float] = {}
        t_mark = time.perf_counter()

        def lap(name: str) -> None:
            """Accumulates elapsed ms under `name` (debug I/O is tracked separately as 'debug_io')."""
            nonlocal t_mark
            now = time.perf_counter()
            timings[name] = round(timings.get(name, 0.0) + (now - t_mark) * 1000.0, 2)
            t_mark = now

        tw, th = self.target_res

        # 1. Load frames
        img_t = cv2.imread(frame_t_path)
        img_t1 = cv2.imread(frame_t1_path)
        if img_t is None or img_t1 is None:
            raise ValueError("Unable to read one or both screenshot paths.")

        if img_t.shape[:2] != img_t1.shape[:2]:
            dbg.stage_early_exit(
                "Frame size changed between T and T+1",
                {"shape_t": list(img_t.shape), "shape_t1": list(img_t1.shape)}
            )
            return self._global_nav_result(1.0, 0.99, {"reason": "Frame size changed between T and T+1"})

        orig_h, orig_w = img_t.shape[:2]
        bgr_t = cv2.resize(img_t, self.target_res, interpolation=cv2.INTER_AREA)
        bgr_t1 = cv2.resize(img_t1, self.target_res, interpolation=cv2.INTER_AREA)
        gray_t = cv2.cvtColor(bgr_t, cv2.COLOR_BGR2GRAY)
        gray_t1 = cv2.cvtColor(bgr_t1, cv2.COLOR_BGR2GRAY)
        lap("load_resize")

        # 2. Scrim-aware, per-channel diff
        gain, scrim_support, is_scrimmed = self._estimate_scrim(gray_t, gray_t1)
        if is_scrimmed:
            ref_t = np.clip(bgr_t.astype(np.float32) * gain, 0, 255).astype(np.uint8)
        else:
            ref_t = bgr_t
        changed, raw_diff, raw_mask = self._compute_changed_mask(ref_t, bgr_t1)
        lap("diff")

        # 3. Containers and action anchor
        containers = layout_t.get("layout", {}).get("containers", [])
        if not containers:
            containers = layout_t.get("containers", [])
        abs_containers = self._denormalize_containers(containers)
        anchor = self._resolve_action_anchor(action, orig_w, orig_h)
        lap("containers")

        dbg.stage_input(
            bgr_t, bgr_t1, abs_containers, anchor, action, self.action_radius,
            {"original_size": [orig_w, orig_h], "target_resolution": [tw, th],
             "frame_t_path": frame_t_path, "frame_t1_path": frame_t1_path}
        )
        dbg.stage_diff(
            bgr_t, ref_t, bgr_t1, raw_diff, raw_mask, changed,
            {"gain": round(gain, 4), "support": round(scrim_support, 4), "is_scrimmed": is_scrimmed,
             "diff_threshold": self.diff_threshold, "min_noise_area": self.min_noise_area}
        )
        lap("debug_io")

        # 4. Triage (core vs. perimeter ring)
        stable_containers, dirty_containers, triage_stats = self._triage_containers(abs_containers, changed)
        dirty_ids = {c["id"] for c in dirty_containers}
        lap("triage")
        dbg.stage_triage(bgr_t1, changed, abs_containers, triage_stats, dirty_ids)
        lap("debug_io")

        # 5. Scaffold stability
        scaffold_intact, survival, scaffold_details = self._verify_scaffolding(
            abs_containers, dirty_ids, gray_t, gray_t1, changed, gain
        )
        lap("scaffold")
        dbg.stage_scaffold(bgr_t1, abs_containers, scaffold_details, survival, scaffold_intact)
        lap("debug_io")

        # 6. Motion estimation (rigid containers first, dynamic/volatile second)
        trace: Dict[str, Any] = {}
        motions, in_place_containers, known_delta = self._resolve_motions(
            dirty_containers, abs_containers, gray_t, gray_t1, changed, ref_t, bgr_t1, trace
        )

        moved_containers: List[Dict[str, Any]] = []
        dynamic_containers: List[Dict[str, Any]] = []
        for c in dirty_containers:
            m = motions.get(c["id"])
            if m is None:
                continue
            c_updated = self._apply_motion(c, m)
            if m["type"] == "STABLE_DYNAMIC":
                dynamic_containers.append(c_updated)
            else:
                moved_containers.append(c_updated)
        self._tag_delta_peers(moved_containers)

        insertion_gaps = [
            m["insertion_gap"] for m in motions.values()
            if m["type"] == "EXPANSION" and m.get("insertion_gap")
        ]
        lap("motion")
        dbg.stage_motion(
            bgr_t, bgr_t1, abs_containers, trace, motions,
            [c["id"] for c in in_place_containers], known_delta
        )
        lap("debug_io")

        # 7. Motion-compensated residual
        hypotheses = [h for m in motions.values() for h in m.get("hypotheses", [])]
        explained = self._explain_with_hypotheses(ref_t, bgr_t1, hypotheses)
        residual_raw = changed.copy()
        residual_raw[explained] = 0
        residual = self._remove_noise_components(residual_raw)
        lap("residual")
        dbg.stage_residual(bgr_t1, changed, explained, residual_raw, residual, hypotheses)
        lap("debug_io")

        # 8. Categorise residual components
        volatile_regions = self._volatile_regions(abs_containers, motions)
        swept_boxes = self._swept_boxes(dirty_containers, motions)
        comps = self._analyze_residual(
            residual, abs_containers, volatile_regions, swept_boxes, anchor, gray_t, gray_t1, trace
        )
        lap("components")
        dbg.stage_components(
            bgr_t, bgr_t1, trace, comps, anchor, self.action_radius, volatile_regions, swept_boxes
        )
        lap("debug_io")

        # 9. Classify
        replaced_ids = sorted({c["container_id"] for c in comps if c["kind"] == "replacement"})
        accounted = self._build_updated_containers(
            stable_containers, moved_containers, dynamic_containers,
            in_place_containers, replaced_ids
        )
        result = self._classify_transition(
            comps=comps,
            scaffold_intact=scaffold_intact,
            survival=survival,
            moved_containers=moved_containers,
            insertion_gaps=insertion_gaps,
            replaced_ids=replaced_ids,
            abs_containers=abs_containers,
            accounted_containers=accounted,
            extra_diag={
                "is_scrimmed": is_scrimmed,
                "scrim_gain": round(gain, 3),
                "scrim_support": round(scrim_support, 3),
                "scaffold_survival": round(survival, 3),
                "dirty_containers": sorted(dirty_ids),
                "known_delta": known_delta,
                "churn_containers": sorted({c["container_id"] for c in comps if c["kind"] == "churn"}),
            }
        )
        lap("classify")

        pipeline_ms = round(sum(v for k, v in timings.items() if k != "debug_io"), 2)
        result.diagnostics["timings_ms"] = {**timings, "pipeline_total": pipeline_ms}

        dbg.stage_result(bgr_t, bgr_t1, result, anchor, self.action_radius, result.diagnostics["timings_ms"])
        dbg.finalize()
        return result

    # =========================================================================
    # Step 1: Scrim estimation and diff
    # =========================================================================
    def _estimate_scrim(self, g0: np.ndarray, g1: np.ndarray) -> Tuple[float, float, bool]:
        """
        Detects a global multiplicative luminance change (dim/brighten scrim). Dark UIs change by
        only a few absolute levels under a scrim, so the test is ratio-based, not difference-based.
        """
        lit = g0 > 40
        if np.count_nonzero(lit) < 0.02 * g0.size:
            return 1.0, 0.0, False

        ratio = g1[lit].astype(np.float32) / g0[lit].astype(np.float32)
        median = float(np.median(ratio))
        support = float(np.mean(np.abs(ratio - median) < 0.08))
        is_scrim = support > 0.5 and abs(median - 1.0) > 0.12
        return (median if is_scrim else 1.0), support, is_scrim

    def _compute_changed_mask(
        self, ref_t: np.ndarray, bgr_t1: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (clean changed mask, raw per-channel max diff, raw mask before noise removal)."""
        diff = cv2.absdiff(ref_t, bgr_t1).max(axis=2)
        raw_mask = (diff >= self.diff_threshold).astype(np.uint8) * 255
        return self._remove_noise_components(raw_mask), diff, raw_mask

    def _remove_noise_components(self, mask: np.ndarray) -> np.ndarray:
        """Drops sub-pixel noise and caret-like slivers. Counts pixels (contourArea is 0 for 1px lines)."""
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if n <= 1:
            return mask
        w = stats[:, cv2.CC_STAT_WIDTH]
        h = stats[:, cv2.CC_STAT_HEIGHT]
        area = stats[:, cv2.CC_STAT_AREA]
        keep = area >= self.min_noise_area
        sliver = (np.minimum(w, h) <= 2) & (np.maximum(w, h) <= self.max_caret_len)
        keep &= ~sliver
        keep[0] = False
        return keep[labels].astype(np.uint8) * 255

    # =========================================================================
    # Step 2: Container triage and scaffold
    # =========================================================================
    def _triage_containers(
        self, abs_containers: List[Dict[str, Any]], changed: np.ndarray
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        """Splits containers into stable / dirty. Also returns per-container stats (for debugging)."""
        stable, dirty = [], []
        stats: Dict[str, Dict[str, Any]] = {}
        for c in abs_containers:
            x, y, w, h = c["pixel_bbox"]
            roi = changed[y:y + h, x:x + w]
            overlap_ratio = cv2.countNonZero(roi) / float(max(1, w * h))
            entry: Dict[str, Any] = {"overlap_ratio": round(overlap_ratio, 5), "core_ratio": None, "core_rect": None}
            stats[c["id"]] = entry

            if overlap_ratio < 0.01:
                entry["dirty"] = False
                stable.append(c)
                continue

            depth_x, depth_y = self._compute_adaptive_insets(w, h)
            core_w, core_h = w - 2 * depth_x, h - 2 * depth_y
            if core_w > 0 and core_h > 0:
                core = roi[depth_y:depth_y + core_h, depth_x:depth_x + core_w]
                core_ratio = cv2.countNonZero(core) / float(core_w * core_h)
                entry["core_rect"] = [x + depth_x, y + depth_y, core_w, core_h]
            else:
                core_ratio = overlap_ratio

            entry["core_ratio"] = round(core_ratio, 5)
            entry["dirty"] = core_ratio >= 0.01
            (stable if core_ratio < 0.01 else dirty).append(c)
        return stable, dirty, stats

    def _verify_scaffolding(
        self,
        all_containers: List[Dict[str, Any]],
        dirty_ids: set,
        gray_t: np.ndarray,
        gray_t1: np.ndarray,
        changed: np.ndarray,
        gain: float
    ) -> Tuple[bool, float, List[Dict[str, Any]]]:
        """
        Area-weighted structural survival with occlusion discount.
        Returns (intact, survival_ratio, per-container verdicts for debugging).
        """
        scaffold = [
            c for c in all_containers
            if not c.get("is_volatile", False)
            and not c.get("is_overlay", False)
            and c.get("scrollable", "none") == "none"
        ]
        if not scaffold:
            return True, 1.0, []

        total_area = sum(c["pixel_bbox"][2] * c["pixel_bbox"][3] for c in scaffold)
        intact_area = 0
        details: List[Dict[str, Any]] = []

        for c in scaffold:
            x, y, w, h = c["pixel_bbox"]
            area = w * h
            verdict: Dict[str, Any] = {"id": c["id"], "area": area}
            details.append(verdict)

            if c["id"] not in dirty_ids:
                intact_area += area
                verdict["status"] = "clean"
                continue

            dx, dy = self._compute_adaptive_insets(w, h)
            cw, ch = w - 2 * dx, h - 2 * dy
            if cw <= 4 or ch <= 4:
                intact_area += area
                verdict["status"] = "intact_too_small_to_verify"
                continue

            crop0 = gray_t[y + dy:y + dy + ch, x + dx:x + dx + cw]
            crop1 = gray_t1[y + dy:y + dy + ch, x + dx:x + dx + cw]

            if np.std(crop0) < 5.0:
                # Flat background: intact only if it is STILL flat with a consistent tone
                if np.std(crop1) < 8.0 and abs(float(crop1.mean()) - gain * float(crop0.mean())) <= 12.0:
                    intact_area += area
                    verdict["status"] = "intact_flat"
                    continue
                verdict["flat_changed"] = True
            else:
                ncc = float(np.nan_to_num(cv2.matchTemplate(crop1, crop0, cv2.TM_CCOEFF_NORMED)[0][0]))
                verdict["ncc"] = round(ncc, 4)
                if ncc >= 0.50:
                    intact_area += area
                    verdict["status"] = "intact_ncc"
                    continue

            # Occlusion discount (a dropdown/modal covering part of the container)
            coverage = cv2.countNonZero(changed[y:y + h, x:x + w]) / float(area)
            verdict["coverage"] = round(coverage, 4)
            if coverage < 0.40:
                intact_area += int(area * (1.0 - coverage))
                verdict["status"] = "intact_occluded"
            else:
                verdict["status"] = "BROKEN"

        survival = intact_area / float(max(1, total_area))
        return survival >= 0.60, survival, details

    # =========================================================================
    # Step 3: Motion estimation
    # =========================================================================
    def _resolve_motions(
        self,
        dirty: List[Dict[str, Any]],
        all_containers: List[Dict[str, Any]],
        gray_t: np.ndarray,
        gray_t1: np.ndarray,
        changed: np.ndarray,
        ref_t: np.ndarray,
        bgr_t1: np.ndarray,
        trace: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]], Optional[Tuple[int, int]]]:
        trace = trace if trace is not None else {}
        attempts: Dict[str, Dict[str, Any]] = trace.setdefault("attempts", {})

        def att(cid: str) -> Dict[str, Any]:
            return attempts.setdefault(cid, {"id": cid, "events": []})

        def log(cid: str, **event: Any) -> None:
            att(cid)["events"].append(event)

        motions: Dict[str, Dict[str, Any]] = {}
        unresolved, pending_dynamic = [], []

        # Pass 1: rigid (non-volatile) containers via structural tile matching.
        # A motion hypothesis is only accepted if it actually explains the container's changed pixels;
        # similar-looking content (another page of rows, another file) can fool tile matching.
        for c in dirty:
            cid = c["id"]
            if c.get("is_volatile", False):
                pending_dynamic.append(c)
                log(cid, stage="structural", status="skipped_volatile")
                continue

            m = self._estimate_container_motion(c, gray_t, gray_t1, changed, att(cid))
            if not m.get("matched"):
                log(cid, stage="structural", status="no_motion_found", reason=m.get("reason"))
                unresolved.append(c)
                continue

            ratio = self._explained_ratio(c, m, ref_t, bgr_t1, changed)
            m["explained_ratio"] = round(ratio, 3)
            accepted = ratio >= self.min_explained_ratio
            log(cid, stage="structural", status="accepted" if accepted else "rejected_unexplained",
                type=m["type"], delta=m["delta"], quality=round(float(m.get("quality", 0.0)), 3),
                explained_ratio=round(ratio, 3), threshold=self.min_explained_ratio)
            if accepted:
                motions[cid] = m
            else:
                unresolved.append(c)

        self._disambiguate_scroll_vs_reflow(all_containers, motions)
        for cid, m in motions.items():
            if m.get("note"):
                log(cid, stage="disambiguate", status="relabeled", note=m["note"], type=m["type"])
        known_delta = self._dominant_delta(motions)

        # Pass 2a: low-texture containers that may simply be moving with their siblings
        in_place: List[Dict[str, Any]] = []
        for c in unresolved:
            cid = c["id"]
            if not known_delta:
                log(cid, stage="sibling", status="no_known_delta")
                in_place.append(c)
                continue
            m = self._verify_sibling_delta(c, known_delta, gray_t, gray_t1)
            if m.get("matched"):
                log(cid, stage="sibling", status="accepted", delta=m["delta"], quality=round(m["quality"], 3))
                motions[cid] = m
            else:
                log(cid, stage="sibling", status="not_verified", known_delta=known_delta)
                in_place.append(c)  # changed in place; its diff stays in the residual

        # Pass 2b: volatile/dynamic containers (video, editor, terminal) using sibling kinematics
        for c in pending_dynamic:
            m = self._resolve_pure_dynamic_container(c, changed, known_delta)
            log(c["id"], stage="dynamic", status="resolved", type=m["type"], delta=m["delta"])
            motions[c["id"]] = m

        return motions, in_place, known_delta

    def _explained_ratio(
        self,
        container: Dict[str, Any],
        motion: Dict[str, Any],
        ref_t: np.ndarray,
        bgr_t1: np.ndarray,
        changed: np.ndarray
    ) -> float:
        """Share of the container's changed pixels that its motion hypothesis really accounts for."""
        hyps = motion.get("hypotheses", [])
        if not hyps:
            return 1.0
        x, y, w, h = container["pixel_bbox"]
        nx, ny, nw, nh = motion["new_pixel_bbox"]
        x1, y1 = min(x, nx), min(y, ny)
        x2, y2 = max(x + w, nx + nw), max(y + h, ny + nh)

        region_changed = changed[y1:y2, x1:x2] > 0
        total = int(np.count_nonzero(region_changed))
        if total == 0:
            return 1.0
        explained = self._explain_with_hypotheses(ref_t, bgr_t1, hyps)[y1:y2, x1:x2]
        return float(np.count_nonzero(region_changed & explained)) / total

    def _verify_planar_topology(
        self, tile_coords: np.ndarray, vectors: np.ndarray, slack: int = 4
    ) -> np.ndarray:
        """Prunes tile matches that invert spatial order relative to many neighbours (vectorised)."""
        n = len(tile_coords)
        if n < 3:
            return np.ones(n, dtype=bool)

        src = tile_coords.astype(np.int32)
        dst = src + vectors.astype(np.int32)

        def inversions(axis: int) -> np.ndarray:
            s, d = src[:, axis], dst[:, axis]
            before = s[:, None] < s[None, :] - slack
            after = d[:, None] > d[None, :] + slack
            return (before & after).sum(axis=1)

        violations = inversions(0) + inversions(1)
        return violations <= max(2, int(0.3 * n))

    def _estimate_container_motion(
        self,
        container: Dict[str, Any],
        gray_t: np.ndarray,
        gray_t1: np.ndarray,
        changed: np.ndarray,
        trace_entry: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Structural motion estimation for a dirty rigid container.
        Tiles are first tested for zero motion (static anchors); only the rest are searched.
        Ambiguous (periodic) matches are rejected.
        """
        x, y, w, h = container["pixel_bbox"]
        tw, th = self.target_res
        crop_t = gray_t[y:y + h, x:x + w]

        step_y = min(self.tile_size, max(8, h // 2))
        step_x = min(self.tile_size, max(8, w // 4))

        sy1, sy2 = max(0, y - 60), min(th, y + h + 60)
        sx1, sx2 = max(0, x - 40), min(tw, x + w + 40)
        search_roi = gray_t1[sy1:sy2, sx1:sx2]

        tiles: List[Tuple[int, int, int, int, bool]] = []  # (tx, ty, dx, dy, is_perimeter)
        tile_log: List[Dict[str, Any]] = []                # debug: one entry per candidate tile
        tile_entries: List[Dict[str, Any]] = []            # log entry of each tile in `tiles` (same order)
        r = self.peak_exclusion_px
        if trace_entry is not None:
            trace_entry.update({"tile_log": tile_log, "step": (step_x, step_y), "search_box": (sx1, sy1, sx2, sy2)})

        for ty in range(0, h - step_y + 1, step_y):
            for tx in range(0, w - step_x + 1, step_x):
                entry: Dict[str, Any] = {"tx": tx, "ty": ty, "w": step_x, "h": step_y, "status": "flat",
                                         "dx": None, "dy": None, "best": None, "second": None}
                tile_log.append(entry)

                tile = crop_t[ty:ty + step_y, tx:tx + step_x]
                if np.std(tile) < self.variance_threshold:
                    continue

                is_perim = tx == 0 or (tx + step_x) >= w or ty == 0 or (ty + step_y) >= h

                # Static-anchor shortcut
                is_static = cv2.countNonZero(changed[y + ty:y + ty + step_y, x + tx:x + tx + step_x]) == 0
                if not is_static:
                    same = gray_t1[y + ty:y + ty + step_y, x + tx:x + tx + step_x]
                    ncc0 = float(np.nan_to_num(cv2.matchTemplate(same, tile, cv2.TM_CCOEFF_NORMED)[0][0]))
                    is_static = ncc0 >= self.static_ncc
                if is_static:
                    entry.update(status="static", dx=0, dy=0)
                    tiles.append((tx, ty, 0, 0, is_perim))
                    tile_entries.append(entry)
                    continue

                if search_roi.shape[0] < tile.shape[0] or search_roi.shape[1] < tile.shape[1]:
                    entry["status"] = "no_roi"
                    continue

                res = cv2.matchTemplate(search_roi, tile, cv2.TM_CCOEFF_NORMED)
                _, best, _, loc = cv2.minMaxLoc(res)
                entry["best"] = round(float(best), 3)
                if best < self.match_threshold:
                    entry["status"] = "no_match"
                    continue

                # Ambiguity rejection for periodic content (list rows, grids, indentation guides)
                masked = res.copy()
                masked[max(0, loc[1] - r):loc[1] + r + 1, max(0, loc[0] - r):loc[0] + r + 1] = -1.0
                second = float(masked.max())
                entry["second"] = round(second, 3)
                if second >= best - self.ambiguity_margin:
                    entry["status"] = "ambiguous"
                    continue

                dx = sx1 + loc[0] - (x + tx)
                dy = sy1 + loc[1] - (y + ty)
                if abs(dx) <= 1 and abs(dy) <= 1:
                    dx = dy = 0
                entry.update(status="static" if (dx, dy) == (0, 0) else "moved", dx=int(dx), dy=int(dy))
                tiles.append((tx, ty, dx, dy, is_perim))
                tile_entries.append(entry)

        if len(tiles) < 3:
            return {"matched": False, "reason": "insufficient_structure"}

        coords = np.array([(t[0], t[1]) for t in tiles])
        vectors = np.array([(t[2], t[3]) for t in tiles])
        keep = self._verify_planar_topology(coords, vectors)
        for e, k in zip(tile_entries, keep):
            if not k:
                e["status"] = "topology_pruned"
        tiles = [t for t, k in zip(tiles, keep) if k]
        if len(tiles) < 3:
            return {"matched": False, "reason": "topology_pruned"}

        vectors = np.array([(t[2], t[3]) for t in tiles])
        perim = np.array([t[4] for t in tiles], dtype=bool)
        moved = np.hypot(vectors[:, 0], vectors[:, 1]) > 1.5
        if np.count_nonzero(moved) < 2:
            return {"matched": False, "reason": "no_motion"}

        # A) Internal scroll: static perimeter, moving core.
        # Only plausible for containers that can scroll. Because the layout has no hierarchy, `scrollable`
        # is True for every child of a scrolling parent, so it is a necessary condition here, never a
        # sufficient one: scrollable containers are still free to translate (see segmentation below).
        core = ~perim
        can_scroll = container.get("scrollable", "none") != "none"
        if can_scroll and np.count_nonzero(perim) >= 2 and np.count_nonzero(core) >= 2:
            norms = np.hypot(vectors[:, 0], vectors[:, 1])
            perim_static = float(np.mean(norms[perim] <= 3))
            core_moved = float(np.mean(norms[core] > 5))
            if perim_static >= 0.7 and core_moved >= 0.7:
                core_dx = int(np.median(vectors[core & (norms > 5), 0]))
                core_dy = int(np.median(vectors[core & (norms > 5), 1]))
                return {
                    "matched": True,
                    "type": "INTERNAL_SCROLL",
                    "new_pixel_bbox": [x, y, w, h],
                    "delta": (core_dx, core_dy),
                    "hypotheses": [([x, y, w, h], (core_dx, core_dy))],
                    "quality": core_moved
                }

        # B) Row-wise segmentation: uniform translation vs. single-gap expansion/contraction
        segments = self._segment_rows(tiles)
        if len(segments) >= 2:
            # Drop outlier segments (a lone mismatched tile must not look like a second region)
            total = len(tiles)
            big = [s for s in segments if len(s["tiles"]) >= max(2, int(0.10 * total))]
            if len(big) < len(segments):
                segments = self._segment_rows([t for s in big for t in s["tiles"]]) if big else []

        if not segments or len(segments) > 2:
            return {"matched": False, "reason": "complex_motion"}

        if len(segments) == 2:
            top, bot = segments
            divergence = bot["dy"] - top["dy"]
            if abs(divergence) >= self.expansion_min_px and abs(bot["dx"] - top["dx"]) <= 3:
                return self._build_vertical_resize(
                    container, gray_t, gray_t1, step_y, top, bot, divergence, len(tiles)
                )

        # Uniform translation (all tiles)
        dxm = int(np.median(vectors[:, 0]))
        dym = int(np.median(vectors[:, 1]))
        if np.hypot(dxm, dym) <= 1.5:
            return {"matched": False, "reason": "no_motion"}
        inliers = np.mean((np.abs(vectors[:, 0] - dxm) <= 3) & (np.abs(vectors[:, 1] - dym) <= 3))
        if inliers < self.min_inlier_ratio:
            return {"matched": False, "reason": "incoherent_motion"}

        new_bbox = [
            max(0, min(tw - w, x + dxm)),
            max(0, min(th - h, y + dym)),
            w, h
        ]
        return {
            "matched": True,
            "type": "TRANSLATION",
            "new_pixel_bbox": new_bbox,
            "delta": (dxm, dym),
            "hypotheses": [(new_bbox, (dxm, dym))],
            "quality": float(inliers)
        }

    def _segment_rows(self, tiles: List[Tuple[int, int, int, int, bool]]) -> List[Dict[str, Any]]:
        """Groups tile rows into runs of (near-)constant motion, ordered top to bottom."""
        rows: Dict[int, List[Tuple[int, int, int, int, bool]]] = {}
        for t in tiles:
            rows.setdefault(t[1], []).append(t)

        segments: List[Dict[str, Any]] = []
        for ty in sorted(rows):
            row_dx = float(np.median([t[2] for t in rows[ty]]))
            row_dy = float(np.median([t[3] for t in rows[ty]]))
            if segments and abs(row_dy - segments[-1]["dy"]) <= 3 and abs(row_dx - segments[-1]["dx"]) <= 3:
                seg = segments[-1]
                seg["tiles"].extend(rows[ty])
                seg["rows"].append(ty)
            else:
                segments.append({"tiles": list(rows[ty]), "rows": [ty]})
                seg = segments[-1]
            seg["dx"] = int(np.median([t[2] for t in seg["tiles"]]))
            seg["dy"] = int(np.median([t[3] for t in seg["tiles"]]))
        return segments

    def _build_vertical_resize(
        self,
        container: Dict[str, Any],
        gray_t: np.ndarray,
        gray_t1: np.ndarray,
        step_y: int,
        top: Dict[str, Any],
        bot: Dict[str, Any],
        divergence: int,
        total_tiles: int
    ) -> Dict[str, Any]:
        x, y, w, h = container["pixel_bbox"]
        tw, th = self.target_res

        top_end = y + max(top["rows"]) + step_y      # abs T coords: first row below the top segment
        bot_start = y + min(bot["rows"])             # abs T coords: first row of the bottom segment
        dy_a, dy_b = top["dy"], bot["dy"]

        # Refine where the top segment's content really stops / the bottom's begins
        while top_end < bot_start and self._row_matches(gray_t, gray_t1, x, w, top_end, dy_a):
            top_end += 1
        while bot_start - 1 >= top_end and self._row_matches(gray_t, gray_t1, x, w, bot_start - 1, dy_b):
            bot_start -= 1

        gap_start = top_end + dy_a
        gap_end = bot_start + dy_b
        new_y = max(0, y + dy_a)
        new_h = max(20, h + divergence)
        new_bbox = [x, new_y, w, min(new_h, th - new_y)]

        if divergence > 0:
            if gap_end <= gap_start:
                gap_end = gap_start + divergence
            region_top = [x, new_y, w, max(0, gap_start - new_y)]
            region_bot = [x, gap_end, w, max(0, new_bbox[1] + new_bbox[3] - gap_end)]
            extra = {"insertion_gap": [x, gap_start, w, gap_end - gap_start]}
            mtype = "EXPANSION"
        else:
            boundary = bot_start + dy_b
            region_top = [x, new_y, w, max(0, boundary - new_y)]
            region_bot = [x, boundary, w, max(0, new_bbox[1] + new_bbox[3] - boundary)]
            extra = {}
            mtype = "CONTRACTION"

        hyps = []
        if region_top[2] > 0 and region_top[3] > 0:
            hyps.append((region_top, (top["dx"], dy_a)))
        if region_bot[2] > 0 and region_bot[3] > 0:
            hyps.append((region_bot, (bot["dx"], dy_b)))

        inl = len(top["tiles"]) + len(bot["tiles"])
        return {
            "matched": True,
            "type": mtype,
            "new_pixel_bbox": new_bbox,
            "delta": (0, divergence),
            "hypotheses": hyps,
            "segment_dys": (dy_a, dy_b),
            "quality": inl / float(max(1, total_tiles)),
            **extra
        }

    def _row_matches(self, gray_t, gray_t1, x: int, w: int, y_src: int, dy: int, tol: float = 6.0) -> bool:
        y_dst = y_src + dy
        if not (0 <= y_src < gray_t.shape[0] and 0 <= y_dst < gray_t1.shape[0]):
            return False
        a = gray_t[y_src, x:x + w].astype(np.int16)
        b = gray_t1[y_dst, x:x + w].astype(np.int16)
        return float(np.mean(np.abs(a - b))) < tol

    def _disambiguate_scroll_vs_reflow(
        self, containers: List[Dict[str, Any]], motions: Dict[str, Dict[str, Any]]
    ) -> None:
        """
        One part static + one part moved is pixel-identical for (a) a container that grew/shrank and
        (b) a scroll body moving under a static header. A real reflow also shifts the neighbour on the
        far side of the moved part. For containers that CAN scroll, relabel as INTERNAL_SCROLL unless a
        neighbour corroborates a reflow.
        """
        by_id = {c["id"]: c for c in containers}
        for cid, m in motions.items():
            if m["type"] not in ("EXPANSION", "CONTRACTION"):
                continue
            c = by_id.get(cid)
            if c is None or c.get("scrollable", "none") == "none":
                continue

            dy_a, dy_b = m["segment_dys"]
            if min(abs(dy_a), abs(dy_b)) > 1:
                continue  # both parts moved: genuine reflow

            x, y, w, h = c["pixel_bbox"]
            bottom_moves = abs(dy_b) >= abs(dy_a)
            moving_dy = dy_b if bottom_moves else dy_a

            corroborated = False
            for oid, om in motions.items():
                if oid == cid or oid not in by_id:
                    continue
                ox, oy, ow, oh = by_id[oid]["pixel_bbox"]
                on_far_side = (oy >= y + h - 8) if bottom_moves else (oy + oh <= y + 8)
                if on_far_side and abs(om["delta"][1] - moving_dy) <= 3 and abs(om["delta"][0]) <= 3:
                    corroborated = True
                    break

            if not corroborated:
                m["revealed_region"] = m.pop("insertion_gap", None)
                m["note"] = f"relabeled {m['type']} -> INTERNAL_SCROLL (scrollable container, no neighbour corroborates a reflow)"
                m["type"] = "INTERNAL_SCROLL"
                m["delta"] = (0, moving_dy)
                m["new_pixel_bbox"] = [x, y, w, h]

    def _dominant_delta(self, motions: Dict[str, Dict[str, Any]]) -> Optional[Tuple[int, int]]:
        deltas = [
            (int(round(m["delta"][0])), int(round(m["delta"][1])))
            for m in motions.values()
            if m["type"] in ("TRANSLATION", "INTERNAL_SCROLL") and np.hypot(*m["delta"]) > 1.5
        ]
        if not deltas:
            return None
        return Counter(deltas).most_common(1)[0][0]

    def _verify_sibling_delta(
        self,
        container: Dict[str, Any],
        delta: Tuple[int, int],
        gray_t: np.ndarray,
        gray_t1: np.ndarray
    ) -> Dict[str, Any]:
        """Verifies (by pixels) that a low-texture container moved with its siblings."""
        x, y, w, h = container["pixel_bbox"]
        tw, th = self.target_res
        nx = max(0, min(tw - w, x + delta[0]))
        ny = max(0, min(th - h, y + delta[1]))
        if (nx, ny) == (x, y):
            return {"matched": False}

        a = gray_t[y:y + h, x:x + w].astype(np.int16)
        b = gray_t1[ny:ny + h, nx:nx + w].astype(np.int16)
        if np.std(a) < self.variance_threshold:
            return {"matched": False}  # flat: motion is unverifiable
        frac = float(np.mean(np.abs(a - b) <= 12))
        if frac >= 0.85:
            return {
                "matched": True,
                "type": "TRANSLATION",
                "new_pixel_bbox": [nx, ny, w, h],
                "delta": (nx - x, ny - y),
                "hypotheses": [([nx, ny, w, h], (nx - x, ny - y))],
                "quality": frac
            }
        return {"matched": False}

    def _resolve_pure_dynamic_container(
        self,
        container: Dict[str, Any],
        changed: np.ndarray,
        sibling_delta: Optional[Tuple[int, int]]
    ) -> Dict[str, Any]:
        """
        Volatile containers (video, editor, terminal): their internal pixels are never verified.
        Scenario B: moved with siblings. Scenario C: resized (aspect ratio kept). Scenario A: stationary.
        Always returns a match; the container's churn is suppressed later, but ONLY non-surface-like churn.
        """
        x, y, w, h = container["pixel_bbox"]
        tw, th = self.target_res

        # Scenario B: kinematic coupling
        if sibling_delta is not None and (sibling_delta[0] != 0 or sibling_delta[1] != 0):
            dx_s, dy_s = sibling_delta
            px = max(0, min(tw - w, x + dx_s))
            py = max(0, min(th - h, y + dy_s))
            proj = changed[py:py + h, px:px + w]
            if cv2.countNonZero(proj) / float(max(1, w * h)) >= 0.20:
                return {
                    "matched": True, "type": "TRANSLATION",
                    "new_pixel_bbox": [px, py, w, h], "delta": (px - x, py - y),
                    "hypotheses": [], "quality": 0.5
                }

        # Scenario C: resize (e.g. theater mode)
        pad = 80
        sy1, sy2 = max(0, y - pad), min(th, y + h + pad)
        sx1, sx2 = max(0, x - pad), min(tw, x + w + pad)
        local = changed[sy1:sy2, sx1:sx2]
        contours, _ = cv2.findContours(local, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            cx, cy, cw, ch = cv2.boundingRect(cnt)
            gx, gy = sx1 + cx, sy1 + cy

            ar_old = w / float(max(1, h))
            ar_new = cw / float(max(1, ch))
            alpha_w, alpha_h = cw / float(w), ch / float(h)
            size_changed = abs(alpha_w - 1.0) >= 0.10 or abs(alpha_h - 1.0) >= 0.10
            inter = self._intersection_area([gx, gy, cw, ch], [x, y, w, h])
            overlaps_old = inter >= 0.5 * min(w * h, cw * ch)

            # A genuine resize is a bounded, solid region of plausible scale. A region that runs into the
            # search-window edge (unless that edge is the screen edge) is a bigger change, not a resize.
            bounded = (
                (gx > sx1 or sx1 == 0) and (gy > sy1 or sy1 == 0)
                and (gx + cw < sx2 or sx2 == tw) and (gy + ch < sy2 or sy2 == th)
            )
            solid = cv2.contourArea(cnt) / float(max(1, cw * ch)) >= 0.5
            plausible_scale = 0.5 <= alpha_w <= 1.6 and 0.5 <= alpha_h <= 1.6

            if (abs(ar_new - ar_old) / ar_old <= 0.15 and size_changed and cw >= 40 and ch >= 40
                    and overlaps_old and bounded and solid and plausible_scale):
                return {
                    "matched": True, "type": "DYNAMIC_RESIZE",
                    "new_pixel_bbox": [gx, gy, cw, ch], "delta": (gx - x, gy - y),
                    "scale_factor": (round(alpha_w, 3), round(alpha_h, 3)),
                    "hypotheses": [], "quality": 0.5
                }

        # Scenario A: stationary
        return {
            "matched": True, "type": "STABLE_DYNAMIC",
            "new_pixel_bbox": [x, y, w, h], "delta": (0, 0),
            "hypotheses": [], "quality": 1.0
        }

    def _apply_motion(self, c: Dict[str, Any], m: Dict[str, Any]) -> Dict[str, Any]:
        tw, th = self.target_res
        cu = c.copy()
        cu["motion_type"] = m["type"]
        cu["delta"] = tuple(int(v) for v in m["delta"])
        cu["delta_norm"] = (round(cu["delta"][0] / tw, 4), round(cu["delta"][1] / th, 4))
        cu["motion_quality"] = round(float(m.get("quality", 0.0)), 3)

        if m["type"] == "INTERNAL_SCROLL":
            cu["scroll_delta"] = cu["delta"]
        if "scale_factor" in m:
            cu["scale_factor"] = m["scale_factor"]
        if m.get("revealed_region"):
            cu["revealed_region"] = self._normalize_bbox(m["revealed_region"])
        if m["type"] not in ("INTERNAL_SCROLL", "STABLE_DYNAMIC"):
            cu["pixel_bbox"] = list(m["new_pixel_bbox"])
            cu["bbox"] = self._normalize_bbox(m["new_pixel_bbox"])
        return cu

    def _tag_delta_peers(self, moved: List[Dict[str, Any]]) -> None:
        """
        Containers sharing the same non-zero delta almost certainly belong to one scrolling/translating
        parent (the layout has no hierarchy). Expose that so the map layer can infer the parent.
        """
        cands = [c for c in moved if c["motion_type"] in ("TRANSLATION", "INTERNAL_SCROLL")
                 and np.hypot(*c["delta"]) > 1.5]
        for c in cands:
            peers = [o["id"] for o in cands if o is not c
                     and abs(o["delta"][0] - c["delta"][0]) <= 3 and abs(o["delta"][1] - c["delta"][1]) <= 3]
            if peers:
                c["delta_peers"] = peers

    # =========================================================================
    # Step 4: Motion-compensated residual
    # =========================================================================
    def _explain_with_hypotheses(
        self,
        ref_t: np.ndarray,
        bgr_t1: np.ndarray,
        hypotheses: List[Tuple[BBox, Tuple[int, int]]]
    ) -> np.ndarray:
        """
        Marks pixels of T+1 that match T shifted by a verified motion. A +/-1px jitter tolerance absorbs
        sub-pixel shifts introduced by down-scaling. Nothing is explained by a bbox alone.
        """
        H, W = bgr_t1.shape[:2]
        explained = np.zeros((H, W), dtype=bool)

        for (rx, ry, rw, rh), (dx, dy) in hypotheses:
            dx, dy = int(dx), int(dy)
            x1, y1 = max(rx, 0, dx), max(ry, 0, dy)
            x2, y2 = min(rx + rw, W, W + dx), min(ry + rh, H, H + dy)
            if x2 <= x1 or y2 <= y1:
                continue

            dst = bgr_t1[y1:y2, x1:x2]
            best = None
            for jy in (-1, 0, 1):
                for jx in (-1, 0, 1):
                    sx, sy = x1 - dx + jx, y1 - dy + jy
                    if sx < 0 or sy < 0 or sx + (x2 - x1) > W or sy + (y2 - y1) > H:
                        continue
                    src = ref_t[sy:sy + (y2 - y1), sx:sx + (x2 - x1)]
                    d = cv2.absdiff(src, dst).max(axis=2)
                    best = d if best is None else np.minimum(best, d)
            if best is not None:
                explained[y1:y2, x1:x2] |= best <= self.comp_threshold
        return explained

    def _volatile_regions(
        self, abs_containers: List[Dict[str, Any]], motions: Dict[str, Dict[str, Any]]
    ) -> List[Tuple[str, BBox]]:
        regions = []
        for c in abs_containers:
            if not c.get("is_volatile", False):
                continue
            regions.append((c["id"], list(c["pixel_bbox"])))
            m = motions.get(c["id"])
            if m and list(m["new_pixel_bbox"]) != list(c["pixel_bbox"]):
                regions.append((c["id"], list(m["new_pixel_bbox"])))
        return regions

    def _swept_boxes(
        self, dirty: List[Dict[str, Any]], motions: Dict[str, Dict[str, Any]]
    ) -> List[BBox]:
        """Areas touched by genuine container motion: where revealed strips / insertion gaps may appear."""
        boxes = []
        for c in dirty:
            m = motions.get(c["id"])
            if not m or m["type"] not in self.MOVING_TYPES or np.hypot(*m["delta"]) <= 1.5:
                continue
            x, y, w, h = c["pixel_bbox"]
            nx, ny, nw, nh = m["new_pixel_bbox"]
            ux, uy = min(x, nx), min(y, ny)
            ux2, uy2 = max(x + w, nx + nw), max(y + h, ny + nh)
            boxes.append(self._expand_bbox([ux, uy, ux2 - ux, uy2 - uy], 6))
        return boxes

    # =========================================================================
    # Step 5: Residual component analysis
    # =========================================================================
    def _resolve_action_anchor(
        self, action: Dict[str, Any], orig_w: int, orig_h: int
    ) -> Optional[Tuple[float, float]]:
        """
        Returns the action point in target-resolution pixels, or None for actions without coordinates
        (keyboard input). Coordinates in [0,1] are normalized unless action["normalized"] says otherwise;
        anything else is treated as pixels of the ORIGINAL screenshot.
        """
        ax, ay = action.get("x"), action.get("y")
        if ax is None or ay is None:
            return None
        try:
            ax, ay = float(ax), float(ay)
        except (TypeError, ValueError):
            return None

        tw, th = self.target_res
        normalized = action.get("normalized")
        if normalized is None:
            normalized = 0.0 <= ax <= 1.0 and 0.0 <= ay <= 1.0
        if normalized:
            return ax * tw, ay * th
        return ax * tw / float(orig_w), ay * th / float(orig_h)

    def _edge_similarity(self, gray_t: np.ndarray, gray_t1: np.ndarray, bbox: BBox) -> float:
        """
        Gradient-structure constancy of a patch between T and T+1 (true Sobel magnitude, NCC).
        Hover tints / focus rings / selection fills keep their edges (~1.0).
        A popup or new content creates or destroys edges (~0.0).
        """
        x, y, w, h = self._expand_bbox(bbox, 1)
        patch_t = gray_t[y:y + h, x:x + w]
        patch_1 = gray_t1[y:y + h, x:x + w]
        if patch_t.size == 0:
            return 0.0

        def mag(p: np.ndarray) -> np.ndarray:
            gx = cv2.Sobel(p, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(p, cv2.CV_32F, 0, 1, ksize=3)
            return cv2.magnitude(gx, gy)

        g0, g1 = mag(patch_t), mag(patch_1)
        e0, e1 = float(g0.mean()), float(g1.mean())
        if max(e0, e1) < 6.0:
            return 1.0                      # both flat: a pure tint change
        if min(e0, e1) < 0.25 * max(e0, e1):
            return 0.0                      # structure appeared / vanished
        a, b = g0 - e0, g1 - e1
        denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
        return float((a * b).sum() / denom) if denom > 1e-6 else 0.0

    def _analyze_residual(
        self,
        residual: np.ndarray,
        abs_containers: List[Dict[str, Any]],
        volatile_regions: List[Tuple[str, BBox]],
        swept_boxes: List[BBox],
        anchor: Optional[Tuple[float, float]],
        gray_t: np.ndarray,
        gray_t1: np.ndarray,
        trace: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Splits the residual into components and assigns each a kind:
        churn (volatile content) > reflow (by-product of verified motion) > replacement (a container's
        content changed) > artifact (in-place control feedback near the action) > surface (new popup/modal)
        > minor. Replacement is decided per CONTAINER, because at low resolution replaced text shatters
        into per-line fragments that are individually too small to recognise.
        """
        tw, th = self.target_res
        total = float(tw * th)

        kernel = np.ones((self.merge_px, self.merge_px), np.uint8)
        closed = cv2.morphologyEx(residual, cv2.MORPH_CLOSE, kernel)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
        if trace is not None:
            trace["closed"], trace["labels"] = closed, labels

        non_volatile = [c for c in abs_containers if not c.get("is_volatile", False)]
        comps: List[Dict[str, Any]] = []

        # --- Pass 1: build components; resolve churn and reflow ---------------------------------
        for i in range(1, n):
            x, y, w, h, closed_area = [int(v) for v in stats[i]]
            sub = labels[y:y + h, x:x + w] == i
            pix = int(np.count_nonzero(sub & (residual[y:y + h, x:x + w] > 0)))
            if pix < self.min_noise_area:
                continue

            comp = {
                "bbox": [x, y, w, h], "pix": pix, "closed_area": closed_area,
                "solidity": closed_area / float(w * h), "density": pix / float(w * h),
                "kind": None, "container_id": None, "edge_sim": None, "label": i
            }
            comp["surface_like"] = self._is_surface_like(comp, total)
            comps.append(comp)

            # Churn inside volatile containers (typing, video, text replaced in an editor/terminal).
            # Only a SOLID, compact, sub-container-sized region survives as a candidate surface.
            vol = self._best_region(comp["bbox"], volatile_regions, min_inside=0.7)
            if vol is not None:
                vid, vbox = vol
                solid = comp["density"] >= self.dense_surface_density
                if (not comp["surface_like"]) or (not solid) or self._iou(comp["bbox"], vbox) >= 0.6:
                    comp.update(kind="churn", container_id=vid)
                    continue

            # Reflow by-products: revealed strips / insertion gaps next to genuinely moved containers
            if any(self._intersection_area(comp["bbox"], sb) > 0 for sb in swept_boxes):
                comp["kind"] = "reflow"

        # --- Pass 2: per-container content replacement ------------------------------------------
        for c in non_volatile:
            cb = c["pixel_bbox"]
            cb_exp = self._expand_bbox(cb, 8)
            members = []
            for comp in comps:
                if comp["kind"] is not None:
                    continue
                if comp["surface_like"] and comp["density"] >= self.dense_surface_density:
                    continue  # solid popup, not replaced text
                inside = self._intersection_area(comp["bbox"], cb_exp) / float(
                    max(1, comp["bbox"][2] * comp["bbox"][3]))
                if inside >= 0.7:
                    members.append(comp)
            if not members:
                continue

            ub = self._union_bbox([m["bbox"] for m in members])
            span = max(ub[2] / float(cb[2]), ub[3] / float(cb[3]))
            area_cover = (ub[2] * ub[3]) / float(max(1, cb[2] * cb[3]))
            if span >= self.replace_span_ratio and area_cover >= self.replace_cover_ratio:
                for m in members:
                    m.update(kind="replacement", container_id=c["id"])

        # --- Pass 3: in-place artifacts vs. new surfaces ----------------------------------------
        for comp in comps:
            if comp["kind"] is not None:
                continue

            near_action = (
                anchor is not None
                and self._point_rect_distance(anchor, comp["bbox"]) <= self.action_radius
            )
            if near_action:
                if not comp["surface_like"]:
                    comp["kind"] = "artifact"
                    continue
                sim = self._edge_similarity(gray_t, gray_t1, comp["bbox"])
                comp["edge_sim"] = sim
                if sim >= self.edge_sim_threshold:
                    comp["kind"] = "artifact"
                    continue

            comp["kind"] = "surface" if comp["surface_like"] else "minor"

        return comps

    def _is_surface_like(self, comp: Dict[str, Any], total_px: float) -> bool:
        _, _, w, h = comp["bbox"]
        min_w, min_h = self.min_surface_size
        return (
            w >= min_w and h >= min_h
            and comp["closed_area"] >= self.min_surface_area_ratio * total_px
            and comp["solidity"] >= self.min_surface_solidity
        )

    # =========================================================================
    # Step 6: Classification
    # =========================================================================
    def _classify_transition(
        self,
        comps: List[Dict[str, Any]],
        scaffold_intact: bool,
        survival: float,
        moved_containers: List[Dict[str, Any]],
        insertion_gaps: List[BBox],
        replaced_ids: List[str],
        abs_containers: List[Dict[str, Any]],
        accounted_containers: List[Dict[str, Any]],
        extra_diag: Dict[str, Any]
    ) -> TransitionResult:
        tw, th = self.target_res
        total = float(tw * th)

        unaccounted_px = sum(c["pix"] for c in comps if c["kind"] not in ("churn", "artifact"))
        unaccounted_ratio = unaccounted_px / total

        surfaces = [c for c in comps if c["kind"] == "surface"]
        reflow = [c for c in comps if c["kind"] == "reflow"]
        artifacts = [c for c in comps if c["kind"] == "artifact"]
        real_moves = [c for c in moved_containers
                      if c["motion_type"] in self.MOVING_TYPES and np.hypot(*c["delta"]) > 1.5]

        diag = dict(extra_diag)
        diag.update({
            "component_kinds": dict(Counter(c["kind"] for c in comps)),
            "moved_containers": [c["id"] for c in real_moves],
        })

        # 1. Global navigation
        if not scaffold_intact or unaccounted_ratio >= self.global_nav_threshold:
            margin = max(min(1.0, unaccounted_ratio / self.global_nav_threshold), 1.0 - survival / 0.60)
            diag["reason"] = "Scaffolding broken or unaccounted area exceeded threshold"
            return self._global_nav_result(unaccounted_ratio, min(0.99, 0.6 + 0.4 * max(0.0, margin)), diag)

        # 2. New surface over (or beside) the existing layout
        if surfaces:
            main = max(surfaces, key=lambda c: c["pix"])
            group = self._cluster_near(main, surfaces, gap=24)
            crop = self._union_bbox([c["bbox"] for c in group])
            diag["surfaces"] = [self._normalize_bbox(c["bbox"]) for c in surfaces]
            diag["overlay_area_ratio"] = crop[2] * crop[3] / total
            diag["reflow_present"] = bool(real_moves)
            conf = 0.6 + 0.35 * min(1.0, main["solidity"]) * min(
                1.0, main["closed_area"] / (4 * self.min_surface_area_ratio * total))
            return TransitionResult(
                transition_type="MODAL_OVERLAY",
                confidence=round(conf, 3),
                scaffold_intact=True,
                updated_containers=accounted_containers,
                target_crop_bbox=self._normalize_bbox(self._pad_bbox(crop)),
                unaccounted_ratio=unaccounted_ratio,
                diagnostics=diag,
                changed_containers=replaced_ids
            )

        # 3. Reflow / scroll: verified container motion (and/or insertion gap)
        if real_moves or insertion_gaps or reflow:
            boxes = [c["bbox"] for c in reflow] + list(insertion_gaps)
            crop = self._normalize_bbox(self._pad_bbox(self._union_bbox(boxes))) if boxes else None
            quality = float(np.mean([c["motion_quality"] for c in real_moves])) if real_moves else 0.5
            diag["repositioned_count"] = len(real_moves)
            return TransitionResult(
                transition_type="REFLOW_UPDATE",
                confidence=round(0.6 + 0.4 * quality, 3),
                scaffold_intact=True,
                updated_containers=accounted_containers,
                target_crop_bbox=crop,
                unaccounted_ratio=unaccounted_ratio,
                diagnostics=diag,
                changed_containers=replaced_ids
            )

        # 4. A container's content was replaced inside an intact scaffold
        if replaced_ids:
            boxes = [c["pixel_bbox"] for c in abs_containers if c["id"] in replaced_ids]
            return TransitionResult(
                transition_type="CONTENT_REPLACED",
                confidence=0.8,
                scaffold_intact=True,
                updated_containers=accounted_containers,
                target_crop_bbox=self._normalize_bbox(self._union_bbox(boxes)),
                unaccounted_ratio=unaccounted_ratio,
                diagnostics=diag,
                changed_containers=replaced_ids
            )

        # 5. In-place control interaction (hover/selection/focus/icon swap near the action)
        if artifacts:
            diag["reason"] = "Action artifact resolved within click radius"
            sims = [c["edge_sim"] for c in artifacts if c["edge_sim"] is not None]
            conf = 0.7 + 0.28 * (float(np.mean(sims)) if sims else 0.8)
            return TransitionResult(
                transition_type="IN_PLACE_CONTROL",
                confidence=round(min(0.98, conf), 3),
                scaffold_intact=True,
                updated_containers=accounted_containers,
                target_crop_bbox=None,
                unaccounted_ratio=unaccounted_ratio,
                diagnostics=diag
            )

        # 6. Nothing structural
        diag["info"] = "Negligible structural residual detected"
        return TransitionResult(
            transition_type="NO_CHANGE",
            confidence=round(0.7 + 0.29 * (1.0 - min(1.0, unaccounted_ratio / 0.01)), 3),
            scaffold_intact=True,
            updated_containers=accounted_containers,
            target_crop_bbox=None,
            unaccounted_ratio=unaccounted_ratio,
            diagnostics=diag
        )

    def _global_nav_result(
        self, unaccounted_ratio: float, confidence: float, diagnostics: Dict[str, Any]
    ) -> TransitionResult:
        return TransitionResult(
            transition_type="GLOBAL_NAVIGATION",
            confidence=round(confidence, 3),
            scaffold_intact=False,
            updated_containers=[],
            target_crop_bbox=None,
            unaccounted_ratio=unaccounted_ratio,
            diagnostics=diagnostics
        )

    def _build_updated_containers(
        self,
        stable: List[Dict[str, Any]],
        moved: List[Dict[str, Any]],
        dynamic: List[Dict[str, Any]],
        in_place: List[Dict[str, Any]],
        replaced_ids: List[str]
    ) -> List[Dict[str, Any]]:
        out = list(stable) + list(moved) + list(dynamic) + list(in_place)
        for c in out:
            if c["id"] in replaced_ids:
                c["content_replaced"] = True
        return out

    # =========================================================================
    # Geometry / coordinate helpers
    # =========================================================================
    def _denormalize_containers(self, containers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Converts normalized bboxes to clamped pixel bboxes; skips degenerate/missing ones."""
        tw, th = self.target_res
        denorm = []
        for i, c in enumerate(containers):
            b = c.get("bbox")
            if not b:
                continue
            try:
                x1 = int(round(b["x"] * tw))
                y1 = int(round(b["y"] * th))
                x2 = int(round((b["x"] + b["w"]) * tw))
                y2 = int(round((b["y"] + b["h"]) * th))
            except (KeyError, TypeError):
                continue
            x1, x2 = max(0, min(tw, x1)), max(0, min(tw, x2))
            y1, y2 = max(0, min(th, y1)), max(0, min(th, y2))
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            c_copy = c.copy()
            c_copy["id"] = c.get("id", f"container_{i}")
            c_copy["pixel_bbox"] = [x1, y1, x2 - x1, y2 - y1]
            denorm.append(c_copy)
        return denorm

    def _normalize_bbox(self, pixel_bbox: List[int]) -> Dict[str, float]:
        tw, th = self.target_res
        return {
            "x": round(pixel_bbox[0] / tw, 4),
            "y": round(pixel_bbox[1] / th, 4),
            "w": round(pixel_bbox[2] / tw, 4),
            "h": round(pixel_bbox[3] / th, 4)
        }

    def _denormalize_bbox(self, norm_bbox: Dict[str, float]) -> List[int]:
        tw, th = self.target_res
        return [
            int(round(norm_bbox["x"] * tw)),
            int(round(norm_bbox["y"] * th)),
            int(round(norm_bbox["w"] * tw)),
            int(round(norm_bbox["h"] * th))
        ]

    def _compute_adaptive_insets(self, w: int, h: int) -> Tuple[int, int]:
        """Calculates scale-aware perimeter depth preserving core surface area."""
        def get_dim_depth(dim: int) -> int:
            if dim <= 8:
                return 0  # Micro-elements cannot support a perimeter ring
            raw_depth = max(1, int(dim * self.border_ratio))
            capped_depth = min(self.max_border_px, raw_depth)
            return min(capped_depth, int(dim * self.max_margin_frac))

        return get_dim_depth(w), get_dim_depth(h)

    def _expand_bbox(self, bbox: BBox, pad: int) -> BBox:
        tw, th = self.target_res
        x1, y1 = max(0, bbox[0] - pad), max(0, bbox[1] - pad)
        x2, y2 = min(tw, bbox[0] + bbox[2] + pad), min(th, bbox[1] + bbox[3] + pad)
        return [x1, y1, x2 - x1, y2 - y1]

    def _pad_bbox(self, bbox: BBox) -> BBox:
        return self._expand_bbox(bbox, self.crop_padding)

    @staticmethod
    def _union_bbox(boxes: List[BBox]) -> BBox:
        x1 = min(b[0] for b in boxes)
        y1 = min(b[1] for b in boxes)
        x2 = max(b[0] + b[2] for b in boxes)
        y2 = max(b[1] + b[3] for b in boxes)
        return [x1, y1, x2 - x1, y2 - y1]

    @staticmethod
    def _intersection_area(a: BBox, b: BBox) -> int:
        ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
        iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
        return ix * iy

    def _iou(self, a: BBox, b: BBox) -> float:
        inter = self._intersection_area(a, b)
        union = a[2] * a[3] + b[2] * b[3] - inter
        return inter / float(max(1, union))

    def _best_region(
        self, bbox: BBox, regions: List[Tuple[str, BBox]], min_inside: float
    ) -> Optional[Tuple[str, BBox]]:
        """Region containing at least `min_inside` of the component's area (best such region)."""
        area = float(max(1, bbox[2] * bbox[3]))
        best, best_frac = None, 0.0
        for rid, rbox in regions:
            frac = self._intersection_area(bbox, rbox) / area
            if frac >= min_inside and frac > best_frac:
                best, best_frac = (rid, rbox), frac
        return best

    @staticmethod
    def _point_rect_distance(p: Tuple[float, float], bbox: BBox) -> float:
        dx = max(bbox[0] - p[0], 0.0, p[0] - (bbox[0] + bbox[2]))
        dy = max(bbox[1] - p[1], 0.0, p[1] - (bbox[1] + bbox[3]))
        return float(np.hypot(dx, dy))

    def _cluster_near(self, main: Dict[str, Any], comps: List[Dict[str, Any]], gap: int) -> List[Dict[str, Any]]:
        """Main component plus every surface within `gap` px of the growing cluster."""
        group, box = [main], list(main["bbox"])
        grew = True
        while grew:
            grew = False
            for c in comps:
                if c in group:
                    continue
                grown = self._expand_bbox(box, gap)
                if self._intersection_area(c["bbox"], grown) > 0:
                    group.append(c)
                    box = self._union_bbox([box, c["bbox"]])
                    grew = True
        return group