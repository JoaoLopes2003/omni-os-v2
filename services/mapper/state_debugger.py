import json
import traceback
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

BBox = Sequence[int]

# --------------------------------------------------------------------------------------
# Palette (BGR)
# --------------------------------------------------------------------------------------
GREEN = (80, 200, 60)
RED = (50, 50, 240)
BLUE = (240, 150, 40)
CYAN = (230, 220, 40)
YELLOW = (40, 220, 240)
ORANGE = (0, 140, 255)
MAGENTA = (230, 60, 230)
PURPLE = (200, 60, 140)
GRAY = (150, 150, 150)
DARK = (70, 70, 70)
WHITE = (255, 255, 255)

KIND_COLORS = {
    "churn": GRAY, "reflow": BLUE, "replacement": ORANGE,
    "artifact": GREEN, "surface": MAGENTA, "minor": DARK,
}
TILE_COLORS = {
    "flat": DARK, "static": GREEN, "moved": CYAN, "ambiguous": ORANGE,
    "no_match": RED, "no_roi": RED, "topology_pruned": PURPLE,
}
MOTION_COLORS = {
    "TRANSLATION": CYAN, "INTERNAL_SCROLL": BLUE, "EXPANSION": GREEN,
    "CONTRACTION": ORANGE, "DYNAMIC_RESIZE": MAGENTA, "STABLE_DYNAMIC": GRAY,
}
TRANSITION_COLORS = {
    "NO_CHANGE": GRAY, "IN_PLACE_CONTROL": GREEN, "REFLOW_UPDATE": BLUE,
    "CONTENT_REPLACED": ORANGE, "MODAL_OVERLAY": MAGENTA, "GLOBAL_NAVIGATION": RED,
}

FONT = cv2.FONT_HERSHEY_SIMPLEX


def _jsonable(o: Any) -> Any:
    """Recursively converts numpy / tuples / paths / NaN into JSON-safe values."""
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return {"__ndarray_shape__": list(o.shape)} if o.size > 64 else o.tolist()
    if isinstance(o, np.generic):
        return _jsonable(o.item())
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def _safe(fn):
    """A debugging bug must never break tracking: swallow errors and log them instead."""
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        if not self.is_active:
            return None
        try:
            return fn(self, *args, **kwargs)
        except Exception:
            self._record_error(fn.__name__, traceback.format_exc())
            return None
    return wrapper


class StateDebugger:
    """
    Saves visual + JSON artifacts for every stage of the StateTransitionTracker pipeline.
    If `debug_dir` is None the debugger is a complete no-op (zero overhead).

    Output layout (one sub-folder per pipeline stage):

        <debug_dir>/
          01_input/        T / T+1 frames, layout overlay, action anchor, input.json
          02_diff/         scrim estimate, raw diff heat-map, changed mask before/after noise removal
          03_triage/       dirty vs. stable containers, overlap/core ratios, core rectangles
          04_scaffold/     scaffold containers, per-container verdict, survival ratio
          05_motion/       per-container tile vectors (T | T+1), accepted/rejected motions, overview
          06_residual/     explained mask, motion hypotheses, residual before/after noise removal
          07_components/   closed mask, categorised components, surface/replacement crops
          08_result/       final overlay, target crop sent to the LLM, result.json, timings.json
          summary.json     stage index + headline result
          debugger_errors.txt   (only if a debug dump itself failed)
    """

    STAGES = {
        "input": "01_input",
        "diff": "02_diff",
        "triage": "03_triage",
        "scaffold": "04_scaffold",
        "motion": "05_motion",
        "residual": "06_residual",
        "components": "07_components",
        "result": "08_result",
    }

    def __init__(self, debug_dir: Optional[str], upscale: int = 2):
        self.is_active = bool(debug_dir)
        self.scale = max(1, int(upscale))
        self.root: Optional[Path] = None
        self._headline: Dict[str, Any] = {}
        if self.is_active:
            self.root = Path(debug_dir)
            self.root.mkdir(parents=True, exist_ok=True)

    # ======================================================================================
    # Stage 0: early exit (no pipeline ran)
    # ======================================================================================
    @_safe
    def stage_early_exit(self, reason: str, details: Dict[str, Any]) -> None:
        self._write_json(self.root, "00_early_exit.json", {"reason": reason, **details})
        self._headline = {"transition_type": "GLOBAL_NAVIGATION", "early_exit": reason}
        self.finalize()

    # ======================================================================================
    # Stage 1: inputs
    # ======================================================================================
    @_safe
    def stage_input(
        self,
        bgr_t: np.ndarray,
        bgr_t1: np.ndarray,
        containers: List[Dict[str, Any]],
        anchor: Optional[Tuple[float, float]],
        action: Dict[str, Any],
        action_radius: int,
        meta: Dict[str, Any]
    ) -> None:
        st = self.STAGES["input"]
        self._write(st, "T.png", bgr_t)
        self._write(st, "T1.png", bgr_t1)
        self._write(st, "T_vs_T1_side_by_side.png", np.hstack([bgr_t, bgr_t1]))

        cv = self._canvas(bgr_t)
        for c in containers:
            self._box(cv, c["pixel_bbox"], self._container_color(c), self._container_label(c), 1)
        self._write(st, "T_layout_overlay.png", self._with_banner(cv, "T + layout containers  (V=volatile O=overlay S=scrollable)"))

        cv1 = self._canvas(bgr_t1)
        if anchor is not None:
            self._draw_anchor(cv1, anchor, action_radius, f"action {action.get('type', '?')}")
        else:
            cv1 = self._with_banner(cv1, "T+1 (action has no coordinates: no proximity anchor)")
        self._write(st, "T1_action_anchor.png", cv1)

        self._write_json(st, "input.json", {
            "action": action,
            "anchor_px": anchor,
            "action_radius_px": action_radius,
            "meta": meta,
            "containers": [
                {"id": c["id"], "pixel_bbox": c["pixel_bbox"], "bbox": c.get("bbox"),
                 "scrollable": c.get("scrollable"), "is_volatile": c.get("is_volatile"),
                 "is_overlay": c.get("is_overlay")}
                for c in containers
            ],
        })

    # ======================================================================================
    # Stage 2: diff
    # ======================================================================================
    @_safe
    def stage_diff(
        self,
        bgr_t: np.ndarray,
        ref_t: np.ndarray,
        bgr_t1: np.ndarray,
        diff: np.ndarray,
        raw_mask: np.ndarray,
        changed: np.ndarray,
        scrim: Dict[str, Any]
    ) -> None:
        st = self.STAGES["diff"]
        heat = cv2.applyColorMap(np.clip(diff.astype(np.int32) * 8, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        self._write(st, "1_raw_diff_heatmap_x8.png", heat)
        self._write(st, "2_changed_mask_raw.png", raw_mask)
        self._write(st, "3_changed_mask_clean.png", changed)
        self._write(st, "4_noise_removed.png", cv2.bitwise_and(raw_mask, cv2.bitwise_not(changed)))

        cv = self._canvas(bgr_t1)
        self._tint(cv, changed, RED, 0.55)
        self._write(st, "5_changed_overlay_on_T1.png", self._with_banner(cv, "changed pixels (red) over T+1"))

        if scrim.get("is_scrimmed"):
            self._write(st, "6_T_scrim_compensated.png", ref_t)

        total = float(changed.size)
        self._write_json(st, "diff.json", {
            "scrim": scrim,
            "raw_changed_px": int(np.count_nonzero(raw_mask)),
            "clean_changed_px": int(np.count_nonzero(changed)),
            "clean_changed_ratio": round(np.count_nonzero(changed) / total, 5),
            "max_channel_diff": int(diff.max()) if diff.size else 0,
        })

    # ======================================================================================
    # Stage 3: triage
    # ======================================================================================
    @_safe
    def stage_triage(
        self,
        bgr_t1: np.ndarray,
        changed: np.ndarray,
        containers: List[Dict[str, Any]],
        stats: Dict[str, Dict[str, Any]],
        dirty_ids: set
    ) -> None:
        st = self.STAGES["triage"]
        cv = self._canvas(bgr_t1)
        self._tint(cv, changed, RED, 0.35)
        for c in containers:
            s = stats.get(c["id"], {})
            dirty = c["id"] in dirty_ids
            label = f"{c['id']} ov={self._pct(s.get('overlap_ratio'))} core={self._pct(s.get('core_ratio'))} {'DIRTY' if dirty else 'stable'}"
            self._box(cv, c["pixel_bbox"], RED if dirty else GREEN, label, 2 if dirty else 1)
            if s.get("core_rect"):
                self._box(cv, s["core_rect"], YELLOW, None, 1, dashed=True)
        self._write(st, "triage_overlay.png", self._with_banner(
            cv, "dirty = core change >= 1% (red) | stable (green) | dashed yellow = core (perimeter ring excluded)"))
        self._write_json(st, "triage.json", {
            "dirty": sorted(dirty_ids),
            "stable": sorted(c["id"] for c in containers if c["id"] not in dirty_ids),
            "per_container": stats,
        })

    # ======================================================================================
    # Stage 4: scaffold
    # ======================================================================================
    @_safe
    def stage_scaffold(
        self,
        bgr_t1: np.ndarray,
        containers: List[Dict[str, Any]],
        details: List[Dict[str, Any]],
        survival: float,
        intact: bool
    ) -> None:
        st = self.STAGES["scaffold"]
        by_id = {d["id"]: d for d in details}
        cv = self._canvas(bgr_t1)
        for c in containers:
            d = by_id.get(c["id"])
            if d is None:
                self._box(cv, c["pixel_bbox"], DARK, f"{c['id']} (not scaffold)", 1, dashed=True)
            else:
                ok = d["status"].startswith("intact") or d["status"] == "clean"
                self._box(cv, c["pixel_bbox"], GREEN if ok else RED, f"{c['id']} {d['status']}", 2)
        banner = f"scaffold {'INTACT' if intact else 'BROKEN'}  survival={survival:.1%} (needs >= 60%)"
        self._write(st, "scaffold_overlay.png", self._with_banner(cv, banner, GREEN if intact else RED))
        self._write_json(st, "scaffold.json", {"intact": intact, "survival": survival, "containers": details})

    # ======================================================================================
    # Stage 5: motion
    # ======================================================================================
    @_safe
    def stage_motion(
        self,
        bgr_t: np.ndarray,
        bgr_t1: np.ndarray,
        containers: List[Dict[str, Any]],
        trace: Dict[str, Any],
        motions: Dict[str, Dict[str, Any]],
        in_place_ids: List[str],
        known_delta: Optional[Tuple[int, int]]
    ) -> None:
        st = self.STAGES["motion"]
        by_id = {c["id"]: c for c in containers}
        attempts = trace.get("attempts", {})

        # Per-container tile figure + attempt json
        for cid, att in attempts.items():
            if att.get("tile_log") and att.get("search_box"):
                fig = self._tile_figure(bgr_t, bgr_t1, by_id[cid]["pixel_bbox"], att["search_box"], att["tile_log"])
                self._write(st, f"{self._slug(cid)}_tiles.png", fig)
            self._write_json(st, f"{self._slug(cid)}_attempt.json", att)

        # Overview: old box (dashed gray) -> new box (colored by motion type)
        cv = self._canvas(bgr_t1)
        for cid, m in motions.items():
            if cid not in by_id:
                continue
            old = by_id[cid]["pixel_bbox"]
            color = MOTION_COLORS.get(m["type"], WHITE)
            moved = list(m["new_pixel_bbox"]) != list(old)
            if moved:
                self._box(cv, old, GRAY, None, 1, dashed=True)
            dx, dy = m["delta"]
            label = f"{cid} {m['type']} ({dx},{dy}) q={m.get('quality', 0):.2f}"
            self._box(cv, m["new_pixel_bbox"], color, label, 2)
            if m.get("insertion_gap"):
                self._box(cv, m["insertion_gap"], MAGENTA, "insertion gap", 1)
            if m.get("revealed_region"):
                self._box(cv, m["revealed_region"], MAGENTA, "revealed", 1)
        for cid in in_place_ids:
            if cid in by_id:
                self._box(cv, by_id[cid]["pixel_bbox"], YELLOW, f"{cid} in-place (no motion)", 2)
        banner = f"motion overview | known sibling delta={known_delta}"
        self._write(st, "motion_overview.png", self._with_banner(cv, banner))

        self._write_json(st, "motion_summary.json", {
            "known_delta": known_delta,
            "in_place": in_place_ids,
            "accepted": {
                cid: {
                    "type": m["type"], "delta": m["delta"], "quality": m.get("quality"),
                    "new_pixel_bbox": m["new_pixel_bbox"], "hypotheses": m.get("hypotheses"),
                    "insertion_gap": m.get("insertion_gap"), "revealed_region": m.get("revealed_region"),
                    "note": m.get("note"),
                } for cid, m in motions.items()
            },
            "attempt_events": {cid: att.get("events", []) for cid, att in attempts.items()},
        })

    # ======================================================================================
    # Stage 6: motion-compensated residual
    # ======================================================================================
    @_safe
    def stage_residual(
        self,
        bgr_t1: np.ndarray,
        changed: np.ndarray,
        explained: np.ndarray,
        residual_raw: np.ndarray,
        residual: np.ndarray,
        hypotheses: List[Tuple[BBox, Tuple[int, int]]]
    ) -> None:
        st = self.STAGES["residual"]
        expl = (explained.astype(np.uint8)) * 255
        self._write(st, "1_explained_mask.png", expl)

        dim = (bgr_t1.astype(np.float32) * 0.35).astype(np.uint8)
        cv = self._canvas(dim)
        ch = changed > 0
        self._tint(cv, (ch & explained).astype(np.uint8) * 255, GREEN, 0.9)
        self._tint(cv, (ch & ~explained).astype(np.uint8) * 255, RED, 0.9)
        self._write(st, "2_changed_vs_explained.png", self._with_banner(
            cv, "changed pixels: green = explained by verified motion | red = UNEXPLAINED (residual)"))

        hv = self._canvas(bgr_t1)
        for rect, (dx, dy) in hypotheses:
            x, y, w, h = rect
            m = np.zeros(changed.shape, np.uint8)
            m[max(0, y):max(0, y + h), max(0, x):max(0, x + w)] = 255
            self._tint(hv, m, CYAN, 0.18)
            self._box(hv, rect, CYAN, f"d=({dx},{dy})", 1)
        self._write(st, "3_motion_hypotheses.png", self._with_banner(hv, f"{len(hypotheses)} verified motion hypotheses (T+1 coordinates)"))

        self._write(st, "4_residual_before_noise.png", residual_raw)
        self._write(st, "5_residual_after_noise.png", residual)
        rv = self._canvas(bgr_t1)
        self._tint(rv, residual, MAGENTA, 0.7)
        self._write(st, "6_residual_overlay.png", self._with_banner(rv, "residual (magenta) = changed AND not explained, after noise removal"))

        self._write_json(st, "residual.json", {
            "changed_px": int(np.count_nonzero(changed)),
            "explained_changed_px": int(np.count_nonzero(ch & explained)),
            "residual_before_noise_px": int(np.count_nonzero(residual_raw)),
            "residual_after_noise_px": int(np.count_nonzero(residual)),
            "residual_ratio": round(np.count_nonzero(residual) / float(residual.size), 5),
            "hypotheses": [{"region": r, "delta": d} for r, d in hypotheses],
        })

    # ======================================================================================
    # Stage 7: residual components
    # ======================================================================================
    @_safe
    def stage_components(
        self,
        bgr_t: np.ndarray,
        bgr_t1: np.ndarray,
        trace: Dict[str, Any],
        comps: List[Dict[str, Any]],
        anchor: Optional[Tuple[float, float]],
        action_radius: int,
        volatile_regions: List[Tuple[str, BBox]],
        swept_boxes: List[BBox]
    ) -> None:
        st = self.STAGES["components"]
        closed = trace.get("closed")
        labels = trace.get("labels")
        if closed is not None:
            self._write(st, "1_closed_mask.png", closed)

        if labels is not None:
            kind_img = np.zeros(bgr_t1.shape, np.uint8)
            for c in comps:
                if c.get("label") is not None:
                    kind_img[labels == c["label"]] = KIND_COLORS.get(c["kind"], WHITE)
            self._write(st, "2_components_by_kind.png", kind_img)

        cv = self._canvas(bgr_t1)
        for rid, rb in volatile_regions:
            self._box(cv, rb, ORANGE, f"volatile:{rid}", 1, dashed=True)
        for sb in swept_boxes:
            self._box(cv, sb, BLUE, "swept", 1, dashed=True)
        for i, c in enumerate(comps):
            color = KIND_COLORS.get(c["kind"], WHITE)
            sim = f" sim={c['edge_sim']:.2f}" if c.get("edge_sim") is not None else ""
            cid = f" [{c['container_id']}]" if c.get("container_id") else ""
            label = f"#{i} {c['kind']}{cid} px={c['pix']} d={c['density']:.2f} s={c['solidity']:.2f}{sim}"
            self._box(cv, c["bbox"], color, label, 2 if c["kind"] in ("surface", "replacement") else 1)
        if anchor is not None:
            self._draw_anchor(cv, anchor, action_radius, "action")
        legend = "  ".join(f"{k}" for k in KIND_COLORS)
        self._write(st, "3_components_overlay.png", self._with_banner(cv, f"components | kinds: {legend}"))

        # Side-by-side crops of the components that matter downstream. Surfaces and replacements are
        # always saved; reflow fragments (revealed text lines etc.) are capped to the largest few.
        indexed = list(enumerate(comps))
        keep = [(i, c) for i, c in indexed if c["kind"] in ("surface", "replacement")][:12]
        keep += sorted([(i, c) for i, c in indexed if c["kind"] == "reflow"], key=lambda t: -t[1]["pix"])[:4]
        for i, c in keep:
            crop = self._side_by_side_crop(bgr_t, bgr_t1, c["bbox"], pad=8, zoom=3)
            self._write(st + "/crops", f"comp_{i}_{c['kind']}.png", crop)

        self._write_json(st, "components.json", [
            {"index": i, **{k: v for k, v in c.items() if k != "label"}} for i, c in enumerate(comps)
        ])

    # ======================================================================================
    # Stage 8: result
    # ======================================================================================
    @_safe
    def stage_result(
        self,
        bgr_t: np.ndarray,
        bgr_t1: np.ndarray,
        result: Any,
        anchor: Optional[Tuple[float, float]],
        action_radius: int,
        timings: Dict[str, float]
    ) -> None:
        st = self.STAGES["result"]
        h, w = bgr_t1.shape[:2]
        cv = self._canvas(bgr_t1)

        for c in result.updated_containers:
            mtype = c.get("motion_type")
            if mtype and mtype != "STABLE_DYNAMIC":
                self._box(cv, c["pixel_bbox"], MOTION_COLORS.get(mtype, WHITE), f"{c['id']} {mtype}", 1)
            if c.get("content_replaced"):
                self._box(cv, c["pixel_bbox"], ORANGE, f"{c['id']} REPLACED", 2, dashed=True)

        if anchor is not None:
            self._draw_anchor(cv, anchor, action_radius, "action")

        crop_px = None
        if result.target_crop_bbox:
            b = result.target_crop_bbox
            crop_px = [int(round(b["x"] * w)), int(round(b["y"] * h)), int(round(b["w"] * w)), int(round(b["h"] * h))]
            self._box(cv, crop_px, MAGENTA, "TARGET CROP (sent to LLM updater)", 3)

        color = TRANSITION_COLORS.get(result.transition_type, WHITE)
        banner = (f"{result.transition_type}  conf={result.confidence:.2f}  "
                  f"scaffold={'ok' if result.scaffold_intact else 'BROKEN'}  unaccounted={result.unaccounted_ratio:.2%}")
        self._write(st, "final_overlay.png", self._with_banner(cv, banner, color))

        if crop_px is not None:
            self._write(st, "target_crop.png", self._side_by_side_crop(bgr_t, bgr_t1, crop_px, pad=0, zoom=3))

        self._write_json(st, "result.json", result.to_dict())
        self._write_json(st, "timings.json", timings)
        self._headline = {
            "transition_type": result.transition_type,
            "confidence": result.confidence,
            "unaccounted_ratio": result.unaccounted_ratio,
            "changed_containers": result.changed_containers,
            "target_crop_bbox": result.target_crop_bbox,
            "total_ms": round(sum(timings.values()), 1),
        }

    # ======================================================================================
    # Finalize
    # ======================================================================================
    @_safe
    def finalize(self) -> None:
        index: Dict[str, List[str]] = {}
        for path in sorted(self.root.rglob("*")):
            if path.is_file() and path.name != "summary.json":
                rel = path.relative_to(self.root)
                index.setdefault(str(rel.parent) if str(rel.parent) != "." else ".", []).append(rel.name)
        self._write_json(self.root, "summary.json", {"headline": self._headline, "files": index})

    # ======================================================================================
    # Drawing helpers
    # ======================================================================================
    def _canvas(self, img: np.ndarray, scale: Optional[int] = None) -> np.ndarray:
        s = scale or self.scale
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if s == 1:
            return img.copy()
        return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)

    def _with_banner(self, cv: np.ndarray, text: str, color=WHITE) -> np.ndarray:
        fs = 0.32 * self.scale + 0.12
        (tw, th), base = cv2.getTextSize(text, FONT, fs, 1)
        bh = th + base + 10
        banner = np.zeros((bh, cv.shape[1], 3), np.uint8)
        cv2.putText(banner, text, (6, th + 4), FONT, fs, color, 1, cv2.LINE_AA)
        return np.vstack([banner, cv])

    def _label(self, cv: np.ndarray, text: str, x: int, y: int, color, s: int) -> None:
        fs = min(0.28 * s + 0.10, 0.55)
        (tw, th), base = cv2.getTextSize(text, FONT, fs, 1)
        ty = y - 3 if y - th - 5 >= 0 else y + th + 4
        x = max(0, min(x, cv.shape[1] - tw - 3))
        cv2.rectangle(cv, (x, ty - th - 2), (x + tw + 2, ty + base), (0, 0, 0), -1)
        cv2.putText(cv, text, (x + 1, ty), FONT, fs, color, 1, cv2.LINE_AA)

    def _box(self, cv: np.ndarray, bbox: BBox, color, label: Optional[str] = None,
             thick: int = 1, dashed: bool = False, s: Optional[int] = None) -> None:
        s = s or self.scale
        x, y, w, h = [int(round(v * s)) for v in bbox]
        if dashed:
            self._dashed_rect(cv, (x, y), (x + w, y + h), color, thick)
        else:
            cv2.rectangle(cv, (x, y), (x + w, y + h), color, thick)
        if label:
            self._label(cv, label, x, y, color, s)

    @staticmethod
    def _dashed_line(cv, a, b, color, thick, dash: int = 6) -> None:
        dist = int(np.hypot(b[0] - a[0], b[1] - a[1]))
        if dist == 0:
            return
        for i in range(0, dist, dash * 2):
            s0, e0 = i / dist, min(i + dash, dist) / dist
            p = (int(a[0] + (b[0] - a[0]) * s0), int(a[1] + (b[1] - a[1]) * s0))
            q = (int(a[0] + (b[0] - a[0]) * e0), int(a[1] + (b[1] - a[1]) * e0))
            cv2.line(cv, p, q, color, thick)

    def _dashed_rect(self, cv, p1, p2, color, thick) -> None:
        (x1, y1), (x2, y2) = p1, p2
        for a, b in (((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)), ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1))):
            self._dashed_line(cv, a, b, color, thick)

    @staticmethod
    def _tint(cv: np.ndarray, mask: np.ndarray, color, alpha: float = 0.5) -> None:
        m = cv2.resize((mask > 0).astype(np.uint8), (cv.shape[1], cv.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        if not m.any():
            return
        col = np.array(color, np.float32)
        cv[m] = (cv[m].astype(np.float32) * (1.0 - alpha) + col * alpha).astype(np.uint8)

    def _draw_anchor(self, cv: np.ndarray, anchor: Tuple[float, float], radius: int, label: str) -> None:
        s = self.scale
        cx, cy = int(anchor[0] * s), int(anchor[1] * s)
        cv2.circle(cv, (cx, cy), int(radius * s), YELLOW, 1, cv2.LINE_AA)
        cv2.line(cv, (cx - 8 * s, cy), (cx + 8 * s, cy), YELLOW, 1)
        cv2.line(cv, (cx, cy - 8 * s), (cx, cy + 8 * s), YELLOW, 1)
        self._label(cv, f"{label} r={radius}", cx + 4, cy - 4, YELLOW, s)

    @staticmethod
    def _container_color(c: Dict[str, Any]):
        if c.get("is_overlay"):
            return MAGENTA
        if c.get("is_volatile"):
            return ORANGE
        if c.get("scrollable", "none") != "none":
            return BLUE
        return GREEN

    @staticmethod
    def _container_label(c: Dict[str, Any]) -> str:
        flags = ""
        if c.get("is_volatile"):
            flags += " V"
        if c.get("is_overlay"):
            flags += " O"
        if c.get("scrollable", "none") != "none":
            flags += f" S:{c['scrollable'][0]}"
        return f"{c['id']}{flags}"

    def _side_by_side_crop(self, bgr_t: np.ndarray, bgr_t1: np.ndarray, bbox: BBox, pad: int, zoom: int) -> np.ndarray:
        h, w = bgr_t1.shape[:2]
        x, y, bw, bh = bbox
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(w, x + bw + pad), min(h, y + bh + pad)
        if x2 <= x1 or y2 <= y1:
            return np.zeros((10, 10, 3), np.uint8)
        left = self._canvas(bgr_t[y1:y2, x1:x2], zoom)
        right = self._canvas(bgr_t1[y1:y2, x1:x2], zoom)
        gap = np.full((left.shape[0], 6, 3), 255, np.uint8)
        fig = np.hstack([left, gap, right])
        banner = np.zeros((18, fig.shape[1], 3), np.uint8)
        cv2.putText(banner, "T", (4, 13), FONT, 0.45, WHITE, 1, cv2.LINE_AA)
        cv2.putText(banner, "T+1", (left.shape[1] + 10, 13), FONT, 0.45, WHITE, 1, cv2.LINE_AA)
        return np.vstack([banner, fig])

    def _tile_figure(self, bgr_t: np.ndarray, bgr_t1: np.ndarray, container_bbox: BBox,
                     search_box: BBox, tile_log: List[Dict[str, Any]]) -> np.ndarray:
        """T (tiles) | T+1 (matched destinations + arrows) over the container's search window."""
        sx1, sy1, sx2, sy2 = search_box
        s = 3 if (sx2 - sx1) < 320 else 2
        left = self._canvas(bgr_t[sy1:sy2, sx1:sx2], s)
        right = self._canvas(bgr_t1[sy1:sy2, sx1:sx2], s)
        cx, cy, cw, ch = container_bbox
        rel = (cx - sx1, cy - sy1, cw, ch)
        self._box(left, rel, WHITE, None, 1, dashed=True, s=s)
        self._box(right, rel, WHITE, None, 1, dashed=True, s=s)

        # Dominant motion vector: only tiles that DISAGREE with it get their own label (avoids clutter)
        moved_vecs = [(e["dx"], e["dy"]) for e in tile_log if e["status"] == "moved"]
        modal = max(set(moved_vecs), key=moved_vecs.count) if moved_vecs else None

        for e in tile_log:
            col = TILE_COLORS.get(e["status"], GRAY)
            bx, by = cx - sx1 + e["tx"], cy - sy1 + e["ty"]
            self._box(left, (bx, by, e["w"], e["h"]), col, None, 1, s=s)
            if e.get("dx") is not None and e["status"] in ("moved", "topology_pruned"):
                dx, dy = e["dx"], e["dy"]
                self._box(right, (bx + dx, by + dy, e["w"], e["h"]), col, None, 1, s=s)
                p = (int((bx + e["w"] / 2) * s), int((by + e["h"] / 2) * s))
                q = (int((bx + dx + e["w"] / 2) * s), int((by + dy + e["h"] / 2) * s))
                cv2.arrowedLine(right, p, q, YELLOW, 1, tipLength=0.25)
                if (dx, dy) != modal or e["status"] == "topology_pruned":
                    self._label(right, f"{dx},{dy}", int((bx + dx) * s), int((by + dy) * s), col, s)

        fig = np.hstack([left, np.full((left.shape[0], 6, 3), 255, np.uint8), right])
        legend = np.zeros((22, fig.shape[1], 3), np.uint8)
        x = 6
        for name, col in TILE_COLORS.items():
            n = sum(1 for e in tile_log if e["status"] == name)
            txt = f"{name}:{n}"
            cv2.putText(legend, txt, (x, 15), FONT, 0.4, col, 1, cv2.LINE_AA)
            x += cv2.getTextSize(txt, FONT, 0.4, 1)[0][0] + 14
        if modal is not None:
            cv2.putText(legend, f"| dominant vector: dx={modal[0]} dy={modal[1]}", (x, 15), FONT, 0.4, YELLOW, 1, cv2.LINE_AA)
        return np.vstack([legend, fig])

    # ======================================================================================
    # IO helpers
    # ======================================================================================
    @staticmethod
    def _pct(value: Optional[float]) -> str:
        return "-" if value is None else f"{value:.1%}"

    @staticmethod
    def _slug(text: str) -> str:
        return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in text)

    def _stage_path(self, stage: Any) -> Path:
        p = self.root / stage if not isinstance(stage, Path) else stage
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _write(self, stage: Any, name: str, img: np.ndarray) -> None:
        cv2.imwrite(str(self._stage_path(stage) / name), img)

    def _write_json(self, stage: Any, name: str, obj: Any) -> None:
        with open(self._stage_path(stage) / name, "w", encoding="utf-8") as f:
            json.dump(_jsonable(obj), f, indent=2, ensure_ascii=False)

    def _record_error(self, where: str, tb: str) -> None:
        try:
            with open(self.root / "debugger_errors.txt", "a", encoding="utf-8") as f:
                f.write(f"--- {where} ---\n{tb}\n")
        except Exception:
            pass