import json
import math
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np


BBox = List[int]  # [x, y, w, h] in target_resolution pixels


@dataclass
class ContainerVariant:
    variant_id: str
    container_id: str
    bbox: Dict[str, float]                  # Normalized {x, y, w, h} in [0.0, 1.0]
    pixel_bbox: BBox                        # [x, y, w, h] in target_resolution (854x480)
    elements: List[Dict[str, Any]] = field(default_factory=list)
    item_template: Optional[Dict[str, Any]] = None
    relation_to_node: str = "INITIAL"       # INITIAL | SUBSET | SUPERSET | SHIFTED_COORDS | OVERLAP
    crop_path: Optional[str] = None


@dataclass
class ContainerNode:
    container_id: str
    description: str
    scrollable: str = "none"
    is_volatile: bool = False
    # Union of all unique elements ever seen across all variants of this container
    known_elements: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    known_item_template: Optional[Dict[str, Any]] = None
    variants: Dict[str, ContainerVariant] = field(default_factory=dict)

    def latest_variant(self) -> Optional[ContainerVariant]:
        if not self.variants:
            return None
        return list(self.variants.values())[-1]

    def merge_into_catalog(
        self, elements: List[Dict[str, Any]], item_template: Optional[Dict[str, Any]]
    ) -> None:
        """Maintains the superset catalog of all elements ever observed in this container."""
        for el in elements:
            eid = el.get("id")
            if not eid:
                continue
            if eid not in self.known_elements:
                self.known_elements[eid] = dict(el)
            else:
                existing = self.known_elements[eid]
                new_opens = el.get("opens_container") or []
                old_opens = existing.get("opens_container") or []
                merged_opens = list(dict.fromkeys(list(old_opens) + list(new_opens)))
                existing["opens_container"] = merged_opens if merged_opens else None

        if item_template:
            if self.known_item_template is None:
                self.known_item_template = dict(item_template)
            elif self.known_item_template.get("template_id") == item_template.get("template_id"):
                known_t_els = {e["id"]: e for e in self.known_item_template.get("elements", []) if e.get("id")}
                for tel in item_template.get("elements", []):
                    if tel.get("id") and tel["id"] not in known_t_els:
                        self.known_item_template.setdefault("elements", []).append(dict(tel))

@dataclass
class ActionEdge:
    source_container_id: str
    element_id: str
    action_type: str                        # e.g., "click", "hover"
    target_container_ids: List[str] = field(default_factory=list)


class ContainerLibrary:
    """
    Compositional Container Graph & Polymorphic Variant Store.

    Responsibilities:
      - Stores ContainerNodes, each supporting multiple geometric/layout ContainerVariants
        that share identical semantic element IDs.
      - Performs fast, illumination- and gradient-invariant CV matching against stored variants
        (using gain compensation, Sobel edge correlation, and +/-1px jitter verification).
      - Retrieves Top-N candidate containers for ElementEngine using Action-Edge priors,
        lexical/semantic description similarity, and geometric aspect-ratio/IoU scoring.
      - Verifies whether newly extracted element coordinates match an existing variant within
        tolerance or require storing a new variant.
      - Generates visually masked T+1 screenshots for incremental LayoutEngine calls.
    """

    def __init__(
        self,
        storage_dir: Optional[str] = None,
        target_resolution: Tuple[int, int] = (854, 480),
        cv_edge_threshold: float = 0.84,
        cv_explained_threshold: float = 0.78,
        comp_threshold: int = 10,
        coord_tolerance_px: float = 6.0,
    ):
        self.storage_dir = Path(storage_dir) if storage_dir else None
        if self.storage_dir:
            (self.storage_dir / "crops").mkdir(parents=True, exist_ok=True)

        self.target_res = target_resolution
        self.cv_edge_threshold = cv_edge_threshold
        self.cv_explained_threshold = cv_explained_threshold
        self.comp_threshold = comp_threshold
        self.coord_tol_px = coord_tolerance_px

        self.nodes: Dict[str, ContainerNode] = {}
        self.edges: Dict[Tuple[str, str, str], ActionEdge] = {}
        self._crops_mem: Dict[str, np.ndarray] = {}  # key: f"{container_id}:{variant_id}" -> BGR crop at target_res

    # =========================================================================
    # 1. Fast CV Matching (Gain + Sobel Edge Correlation + Jitter Verification)
    # =========================================================================
    def match_variant_cv(
        self,
        bgr_t1_target: np.ndarray,
        variant: ContainerVariant,
        search_pad_px: int = 16,
    ) -> Tuple[bool, float, Tuple[int, int]]:
        """
        Checks if `variant` is present in `bgr_t1_target` (resized to target_resolution).
        Uses gain compensation, Sobel gradient structure constancy, and +/-1px jitter verification
        so ambient background gradients (e.g., Spotify purple header shift) do not break matching.

        Returns:
            (is_match, confidence_score, (dx, dy))
        """
        ref_crop = self._load_variant_crop(variant)
        if ref_crop is None or ref_crop.size == 0:
            return False, 0.0, (0, 0)

        tw, th = self.target_res
        x, y, w, h = variant.pixel_bbox
        if w <= 4 or h <= 4 or ref_crop.shape[:2] != (h, w):
            return False, 0.0, (0, 0)

        gray_ref = cv2.cvtColor(ref_crop, cv2.COLOR_BGR2GRAY)
        gray_t1 = cv2.cvtColor(bgr_t1_target, cv2.COLOR_BGR2GRAY)

        # 1. Locate best alignment within a small local search window (absorbs minor window/layout shifts)
        sx1, sy1 = max(0, x - search_pad_px), max(0, y - search_pad_px)
        sx2, sy2 = min(tw, x + w + search_pad_px), min(th, y + h + search_pad_px)
        roi_gray = gray_t1[sy1:sy2, sx1:sx2]
        if roi_gray.shape[0] < h or roi_gray.shape[1] < w:
            return False, 0.0, (0, 0)

        # Match on Sobel gradient magnitudes so smooth background color/gradient changes are ignored
        mag_ref = self._sobel_magnitude(gray_ref)
        mag_roi = self._sobel_magnitude(roi_gray)

        if float(mag_ref.mean()) >= 4.0:
            res = cv2.matchTemplate(mag_roi, mag_ref, cv2.TM_CCOEFF_NORMED)
            _, best_edge_ncc, _, loc = cv2.minMaxLoc(res)
            dx = (sx1 + loc[0]) - x
            dy = (sy1 + loc[1]) - y
            if abs(dx) <= 1 and abs(dy) <= 1:
                dx = dy = 0
        else:
            dx = dy = 0
            best_edge_ncc = 1.0

        nx, ny = x + dx, y + dy
        if nx < 0 or ny < 0 or nx + w > tw or ny + h > th:
            return False, 0.0, (0, 0)

        dst_bgr = bgr_t1_target[ny:ny + h, nx:nx + w]
        dst_gray = gray_t1[ny:ny + h, nx:nx + w]

        # 2. Verify structural edge similarity at the aligned position
        edge_sim = self._edge_similarity_patch(gray_ref, dst_gray)
        if edge_sim < self.cv_edge_threshold:
            return False, round(edge_sim, 3), (dx, dy)

        # 3. Gain-compensated, +/-1px jitter pixel verification
        gain = self._estimate_local_gain(gray_ref, dst_gray)
        comp_ref = np.clip(ref_crop.astype(np.float32) * gain, 0, 255).astype(np.uint8)
        explained_ratio = self._jitter_explained_ratio(comp_ref, dst_bgr)

        # If edge structure is nearly identical (>= 0.90), allow smooth background gradients
        # that otherwise lower raw RGB pixel explanation.
        is_match = (
            (edge_sim >= self.cv_edge_threshold and explained_ratio >= self.cv_explained_threshold)
            or edge_sim >= 0.91
        )
        score = 0.65 * edge_sim + 0.35 * explained_ratio
        return is_match, round(float(score), 4), (dx, dy)

    def find_best_library_match_cv(
        self,
        bgr_t1_target: np.ndarray,
        region_bbox: BBox,
        action_context: Optional[Tuple[str, str, str]] = None,
        min_iou: float = 0.35,
    ) -> Optional[Tuple[ContainerNode, ContainerVariant, float, Tuple[int, int]]]:
        """
        Searches the library for any stored ContainerVariant that visually matches `bgr_t1_target`
        at or near `region_bbox` without calling any LLM.
        """
        candidates: List[Tuple[float, ContainerNode, ContainerVariant]] = []
        prior_targets: Set[str] = set()
        if action_context and action_context in self.edges:
            prior_targets.update(self.edges[action_context].target_container_ids)

        for cid, node in self.nodes.items():
            if node.is_volatile:
                continue
            for var in node.variants.values():
                iou = self._bbox_iou(region_bbox, var.pixel_bbox)
                is_prior = cid in prior_targets
                if iou >= min_iou or is_prior:
                    priority = (1.0 if is_prior else 0.0) + iou
                    candidates.append((priority, node, var))

        candidates.sort(key=lambda item: item[0], reverse=True)

        best_match = None
        best_score = 0.0
        for _, node, var in candidates:
            ok, score, delta = self.match_variant_cv(bgr_t1_target, var)
            if ok and score > best_score:
                best_score = score
                best_match = (node, var, score, delta)

        return best_match

    # =========================================================================
    # 2. Top-N Candidate Selection for ElementEngine
    # =========================================================================
    def get_top_candidates(
        self,
        detected_container: Dict[str, Any],
        action_context: Optional[Tuple[str, str, str]] = None,
        top_n: int = 3,
        min_score: float = 0.22,
    ) -> List[Dict[str, Any]]:
        """
        Selects up to `top_n` candidate containers from the library to pass into `ElementEngine`.
        - Evaluates geometric similarity across ALL stored variants of each node and uses the
          highest-matching variant's bbox.
        - Attaches `node.known_elements` (the union superset of elements across all variants)
          so the LLM has the complete vocabulary of element IDs for subset/superset matching.
        """
        if not self.nodes:
            return []

        det_id = detected_container.get("id", "")
        det_desc = detected_container.get("description", "")
        det_bbox = detected_container.get("bbox", {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0})
        det_tokens = self._tokenize(f"{det_id} {det_desc}")
        ar_det = det_bbox["w"] / max(1e-4, det_bbox["h"])

        prior_targets: Set[str] = set()
        if action_context and action_context in self.edges:
            prior_targets.update(self.edges[action_context].target_container_ids)

        scored: List[Tuple[float, Dict[str, Any]]] = []
        for cid, node in self.nodes.items():
            if not node.variants:
                continue

            # 1. Find the variant with the highest geometric match to the detected container
            best_var: Optional[ContainerVariant] = None
            best_geom_score = -1.0
            for var in node.variants.values():
                iou = self._norm_bbox_iou(det_bbox, var.bbox)
                ar_var = var.bbox["w"] / max(1e-4, var.bbox["h"])
                ar_sim = max(0.0, 1.0 - abs(ar_det - ar_var) / max(ar_det, ar_var))
                g_score = 0.65 * iou + 0.35 * ar_sim
                if g_score > best_geom_score:
                    best_geom_score = g_score
                    best_var = var

            assert best_var is not None

            # 2. Use the union catalog (contains all elements from the largest variant AND best_var)
            catalog_elements = list(node.known_elements.values()) or best_var.elements
            catalog_template = node.known_item_template or best_var.item_template

            # 3. Action-edge history bonus
            edge_score = 1.0 if cid in prior_targets else 0.0

            # 4. Semantic similarity (container id + description + union of all known element texts)
            elem_words = " ".join(
                f"{e.get('id', '')} {e.get('text', '') or ''} {e.get('description', '')}"
                for e in catalog_elements
            )
            node_tokens = self._tokenize(f"{cid} {node.description} {elem_words}")
            sem_score = self._dice_similarity(det_tokens, node_tokens)

            total_score = 0.45 * edge_score + 0.35 * sem_score + 0.20 * best_geom_score
            if total_score >= min_score or edge_score > 0:
                scored.append((
                    total_score,
                    {
                        "container_id": cid,
                        "best_variant_id": best_var.variant_id,
                        "description": node.description,
                        "scrollable": node.scrollable,
                        "bbox": best_var.bbox,
                        "elements": catalog_elements,
                        "item_template": catalog_template,
                        "candidate_score": round(total_score, 3),
                    }
                ))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [item[1] for item in scored[:top_n]]

    # =========================================================================
    # 3. Local Coordinate Verification & Variant Upsert
    # =========================================================================
    def upsert_from_extraction(
        self,
        detected_container: Dict[str, Any],
        extraction_data: Dict[str, Any],
        bgr_t1_target: Optional[np.ndarray] = None,
        action_context: Optional[Tuple[str, str, str]] = None,
    ) -> Tuple[ContainerNode, ContainerVariant, bool]:
        """
        Resolves whether the newly extracted container is:
          - An EXACT match to an existing variant (same element IDs + same coordinates within tolerance)
          - A SUBSET, SUPERSET, or SHIFTED_COORDS variant of an existing ContainerNode
          - A brand-new ContainerNode
        """
        tw, th = self.target_res
        new_elements = extraction_data.get("elements", [])
        new_template = extraction_data.get("item_template")

        # 1. Determine target ContainerNode (from LLM match or local subset/superset + action check)
        matched_id = extraction_data.get("matched_container_id")
        if not matched_id or matched_id not in self.nodes:
            matched_id = self._infer_subset_superset_node_id(new_elements, new_template, action_context)

        cid = matched_id if (matched_id and matched_id in self.nodes) else detected_container["id"]

        norm_bbox = detected_container["bbox"]
        px_bbox = [
            int(round(norm_bbox["x"] * tw)),
            int(round(norm_bbox["y"] * th)),
            max(2, int(round(norm_bbox["w"] * tw))),
            max(2, int(round(norm_bbox["h"] * th))),
        ]

        relation = "INITIAL"
        if cid in self.nodes:
            node = self.nodes[cid]
            if detected_container.get("is_volatile"):
                node.is_volatile = True

            # Check all existing variants of this node
            for var in node.variants.values():
                rel, coords_equal = self._assess_variant_relationship(var, px_bbox, new_elements, new_template)
                if rel == "EXACT_SET" and coords_equal:
                    # Identical element set AND identical coordinates -> reuse existing variant!
                    node.merge_into_catalog(new_elements, new_template)
                    if action_context:
                        self.record_action_edge(*action_context, target_container_id=cid)
                    return node, var, False
                if rel in ("SUBSET", "SUPERSET"):
                    relation = rel
                elif rel == "EXACT_SET" and not coords_equal:
                    relation = "SHIFTED_COORDS"
                elif relation == "INITIAL":
                    relation = rel

            variant_id = f"v{len(node.variants) + 1}"
        else:
            node = ContainerNode(
                container_id=cid,
                description=detected_container.get("description", ""),
                scrollable=detected_container.get("scrollable", "none"),
                is_volatile=bool(detected_container.get("is_volatile", False)),
            )
            self.nodes[cid] = node
            variant_id = "v1"

        node.merge_into_catalog(new_elements, new_template)

        variant = ContainerVariant(
            variant_id=variant_id,
            container_id=cid,
            bbox=norm_bbox,
            pixel_bbox=px_bbox,
            elements=new_elements,
            item_template=new_template,
            relation_to_node=relation,
        )

        if bgr_t1_target is not None:
            self._save_variant_crop(variant, bgr_t1_target)

        node.variants[variant_id] = variant

        if action_context:
            self.record_action_edge(*action_context, target_container_id=cid)

        return node, variant, True

    def _infer_subset_superset_node_id(
        self,
        new_elements: List[Dict[str, Any]],
        new_template: Optional[Dict[str, Any]],
        action_context: Optional[Tuple[str, str, str]],
    ) -> Optional[str]:
        """
        Local fallback: if the image elements are a strict subset/superset of a known node
        (especially one triggered by the same action_context), returns that node's container_id.
        """
        new_ids = {e["id"] for e in new_elements if e.get("id")}
        if not new_ids and not new_template:
            return None

        prior_targets: Set[str] = set()
        if action_context and action_context in self.edges:
            prior_targets.update(self.edges[action_context].target_container_ids)

        best_cid: Optional[str] = None
        best_score = 0.0

        for cid, node in self.nodes.items():
            known_ids = set(node.known_elements.keys())
            if new_ids and known_ids:
                inter = len(new_ids & known_ids)
                min_len = min(len(new_ids), len(known_ids))
                subset_ratio = inter / float(max(1, min_len))
            else:
                subset_ratio = 0.0

            tmpl_match = (
                bool(new_template and node.known_item_template)
                and new_template.get("template_id") == node.known_item_template.get("template_id")
            )

            is_same_action = cid in prior_targets
            # Require either a strict subset/superset (>= 0.80) or moderate overlap (>= 0.60) + same action origin
            if (subset_ratio >= 0.80 and len(new_ids & known_ids) >= 2) or (
                is_same_action and (subset_ratio >= 0.60 or tmpl_match)
            ):
                score = subset_ratio + (0.25 if is_same_action else 0.0) + (0.20 if tmpl_match else 0.0)
                if score > best_score:
                    best_score = score
                    best_cid = cid

        return best_cid

    def _assess_variant_relationship(
        self,
        existing: ContainerVariant,
        new_px_bbox: BBox,
        new_elements: List[Dict[str, Any]],
        new_template: Optional[Dict[str, Any]],
    ) -> Tuple[str, bool]:
        """
        Compares the element sets and shared coordinates between `existing` and `new_elements`.

        Returns:
            (set_relation, shared_coordinates_equal)
            where set_relation is one of: "EXACT_SET", "SUBSET", "SUPERSET", "OVERLAP"
        """
        old_by_id = {e["id"]: e for e in existing.elements if e.get("id")}
        new_by_id = {e["id"]: e for e in new_elements if e.get("id")}
        old_ids = set(old_by_id.keys())
        new_ids = set(new_by_id.keys())

        if new_ids == old_ids:
            set_relation = "EXACT_SET"
        elif new_ids < old_ids:
            set_relation = "SUBSET"
        elif new_ids > old_ids:
            set_relation = "SUPERSET"
        else:
            set_relation = "OVERLAP"

        ox, oy, ow, oh = existing.pixel_bbox
        nx, ny, nw, nh = new_px_bbox

        # Check whether all shared elements have identical coordinates within tolerance
        shared_ids = old_ids & new_ids
        coords_equal = True
        for eid in shared_ids:
            old_el = old_by_id[eid]
            new_el = new_by_id[eid]
            old_px_x = ox + old_el["rel_x"] * ow
            old_px_y = oy + old_el["rel_y"] * oh
            new_px_x = nx + new_el["rel_x"] * nw
            new_px_y = ny + new_el["rel_y"] * nh
            if math.hypot(new_px_x - old_px_x, new_px_y - old_px_y) > self.coord_tol_px:
                coords_equal = False
                break

        # Also verify template bounds if templates are present
        if bool(existing.item_template) != bool(new_template):
            coords_equal = False
        elif existing.item_template and new_template:
            if existing.item_template.get("template_id") != new_template.get("template_id"):
                coords_equal = False
            else:
                ob = existing.item_template["collection_bounds"]
                nb = new_template["collection_bounds"]
                old_cb = [ox + ob["x"] * ow, oy + ob["y"] * oh, ob["w"] * ow, ob["h"] * oh]
                new_cb = [nx + nb["x"] * nw, ny + nb["y"] * nh, nb["w"] * nw, nb["h"] * nh]
                if any(abs(a - b) > self.coord_tol_px for a, b in zip(old_cb, new_cb)):
                    coords_equal = False

        return set_relation, coords_equal

    # =========================================================================
    # 4. Multi-Action Lookahead Resolver (Runtime Execution)
    # =========================================================================
    def resolve_lookahead_click(
        self,
        bgr_t1_full: np.ndarray,
        expected_container_ids: List[str],
        target_element_id: str,
    ) -> Optional[Tuple[int, int, str, str]]:
        """
        Used by the Executor during a multi-step lookahead chain:
          e.g., Click("menu_3") -> spawns one of `expected_container_ids` -> Click(`target_element_id`).

        Runs fast CV matching across all stored variants of `expected_container_ids` on frame T+1.
        If a variant matches above threshold, computes and returns the exact (x, y) pixel coordinates
        on `bgr_t1_full` for `target_element_id`:
            (abs_x_px, abs_y_px, matched_container_id, matched_variant_id)
        Returns None if no stored variant matches (signaling the caller to abort lookahead and map).
        """
        orig_h, orig_w = bgr_t1_full.shape[:2]
        bgr_t1_target = cv2.resize(bgr_t1_full, self.target_res, interpolation=cv2.INTER_AREA)
        tw, th = self.target_res

        best_var: Optional[ContainerVariant] = None
        best_delta: Tuple[int, int] = (0, 0)
        best_score = 0.0

        for cid in expected_container_ids:
            node = self.nodes.get(cid)
            if not node:
                continue
            for var in node.variants.values():
                ok, score, delta = self.match_variant_cv(bgr_t1_target, var)
                if ok and score > best_score:
                    best_score = score
                    best_var = var
                    best_delta = delta

        if best_var is None:
            return None

        for el in best_var.elements:
            if el.get("id") == target_element_id:
                vx, vy, vw, vh = best_var.pixel_bbox
                dx, dy = best_delta
                target_x = (vx + dx) + el["rel_x"] * vw
                target_y = (vy + dy) + el["rel_y"] * vh
                abs_x = int(round((target_x / tw) * orig_w))
                abs_y = int(round((target_y / th) * orig_h))
                return abs_x, abs_y, best_var.container_id, best_var.variant_id

        return None

    # =========================================================================
    # 5. Visual Masking for Incremental LayoutEngine Calls
    # =========================================================================
    def build_masked_image(
        self,
        frame_t1_path: str,
        verified_containers: List[Dict[str, Any]],
        unmasked_holes_norm: Optional[List[Dict[str, float]]] = None,
        output_path: Optional[str] = None,
        mask_alpha: float = 0.78,
    ) -> str:
        """
        Paints a dark semi-transparent `[KNOWN: <id>]` mask over all `verified_containers` on frame T+1,
        while carving out any `unmasked_holes_norm` (e.g., a newly opened dropdown menu floating over
        an otherwise intact container) so LayoutEngine sees ONLY the newly revealed UI regions.
        """
        img = cv2.imread(frame_t1_path)
        if img is None:
            raise ValueError(f"Unable to read screenshot at '{frame_t1_path}'")

        H, W = img.shape[:2]
        mask_bool = np.zeros((H, W), dtype=bool)
        boxes_to_label: List[Tuple[str, int, int, int, int]] = []

        for c in verified_containers:
            b = c.get("bbox")
            if not b:
                continue
            x1 = max(0, min(W, int(round(b["x"] * W))))
            y1 = max(0, min(H, int(round(b["y"] * H))))
            x2 = max(0, min(W, int(round((b["x"] + b["w"]) * W))))
            y2 = max(0, min(H, int(round((b["y"] + b["h"]) * H))))
            if x2 > x1 and y2 > y1:
                mask_bool[y1:y2, x1:x2] = True
                boxes_to_label.append((c.get("id", "known"), x1, y1, x2, y2))

        # Carve out any floating overlay / unexplained surface holes so they remain 100% visible
        if unmasked_holes_norm:
            for hole in unmasked_holes_norm:
                hx1 = max(0, min(W, int(round(hole["x"] * W))))
                hy1 = max(0, min(H, int(round(hole["y"] * H))))
                hx2 = max(0, min(W, int(round((hole["x"] + hole["w"]) * W))))
                hy2 = max(0, min(H, int(round((hole["y"] + hole["h"]) * H))))
                mask_bool[hy1:hy2, hx1:hx2] = False

        canvas = img.copy()
        dark = np.full_like(canvas, 18)
        canvas[mask_bool] = (
            canvas[mask_bool].astype(np.float32) * (1.0 - mask_alpha)
            + dark[mask_bool].astype(np.float32) * mask_alpha
        ).astype(np.uint8)

        for cid, x1, y1, x2, y2 in boxes_to_label:
            # Only draw the label if the center of the box was not carved out by an unmasked hole
            cy, cx = (y1 + y2) // 2, (x1 + x2) // 2
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (70, 160, 70), 1)
            if mask_bool[min(H - 1, max(0, cy)), min(W - 1, max(0, cx))]:
                label = f"[KNOWN: {cid}]"
                (tw_txt, th_txt), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                lx = max(x1 + 4, min(cx - tw_txt // 2, x2 - tw_txt - 4))
                ly = max(y1 + th_txt + 6, min(cy, y2 - 6))
                cv2.rectangle(canvas, (lx - 3, ly - th_txt - 3), (lx + tw_txt + 3, ly + base + 2), (0, 0, 0), -1)
                cv2.putText(canvas, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (110, 220, 110), 1, cv2.LINE_AA)

        if output_path is None:
            out_dir = self.storage_dir if self.storage_dir else Path(".")
            output_path = str(out_dir / "masked_t1_update.png")

        cv2.imwrite(output_path, canvas)
        return output_path

    # =========================================================================
    # 6. Graph Edge & Persistence Helpers
    # =========================================================================
    def record_action_edge(
        self,
        source_container_id: str,
        element_id: str,
        action_type: str,
        target_container_ids: List[str],
    ) -> None:
        """
        Records that interacting with `element_id` inside `source_container_id` opened `target_container_ids`,
        and syncs the `opens_container: Optional[List[str]]` field directly onto the source element.
        """
        if not target_container_ids:
            return

        key = (source_container_id, element_id, action_type)
        edge = self.edges.get(key)
        if edge is None:
            edge = ActionEdge(
                source_container_id=source_container_id,
                element_id=element_id,
                action_type=action_type,
                target_container_ids=list(dict.fromkeys(target_container_ids)),
            )
            self.edges[key] = edge
        else:
            for tid in target_container_ids:
                if tid not in edge.target_container_ids:
                    edge.target_container_ids.append(tid)

        # Sync the updated list back onto the element in the source container's catalog and variants
        source_node = self.nodes.get(source_container_id)
        if source_node:
            if element_id in source_node.known_elements:
                cur = source_node.known_elements[element_id].get("opens_container") or []
                source_node.known_elements[element_id]["opens_container"] = list(
                    dict.fromkeys(list(cur) + list(target_container_ids))
                )
            for var in source_node.variants.values():
                for el in var.elements:
                    if el.get("id") == element_id:
                        cur = el.get("opens_container") or []
                        el["opens_container"] = list(
                            dict.fromkeys(list(cur) + list(target_container_ids))
                        )

    def save_to_disk(self) -> None:
        if not self.storage_dir:
            return
        payload = {
            "nodes": {cid: asdict(node) for cid, node in self.nodes.items()},
            "edges": [asdict(e) for e in self.edges.values()],
        }
        with open(self.storage_dir / "container_library.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    def _save_variant_crop(self, variant: ContainerVariant, bgr_target: np.ndarray) -> None:
        x, y, w, h = variant.pixel_bbox
        crop = bgr_target[y:y + h, x:x + w].copy()
        key = f"{variant.container_id}:{variant.variant_id}"
        self._crops_mem[key] = crop
        if self.storage_dir:
            crop_file = self.storage_dir / "crops" / f"{variant.container_id}_{variant.variant_id}.png"
            cv2.imwrite(str(crop_file), crop)
            variant.crop_path = str(crop_file)

    def _load_variant_crop(self, variant: ContainerVariant) -> Optional[np.ndarray]:
        key = f"{variant.container_id}:{variant.variant_id}"
        if key in self._crops_mem:
            return self._crops_mem[key]
        if variant.crop_path and Path(variant.crop_path).exists():
            img = cv2.imread(variant.crop_path)
            if img is not None:
                self._crops_mem[key] = img
                return img
        return None

    # =========================================================================
    # Low-Level CV & Math Utilities
    # =========================================================================
    @staticmethod
    def _sobel_magnitude(gray: np.ndarray) -> np.ndarray:
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        return cv2.magnitude(gx, gy)

    def _edge_similarity_patch(self, gray_a: np.ndarray, gray_b: np.ndarray) -> float:
        if gray_a.size == 0 or gray_a.shape != gray_b.shape:
            return 0.0
        g0, g1 = self._sobel_magnitude(gray_a), self._sobel_magnitude(gray_b)
        e0, e1 = float(g0.mean()), float(g1.mean())
        if max(e0, e1) < 6.0:
            return 1.0
        if min(e0, e1) < 0.25 * max(e0, e1):
            return 0.0
        a, b = g0 - e0, g1 - e1
        denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
        return float((a * b).sum() / denom) if denom > 1e-6 else 0.0

    @staticmethod
    def _estimate_local_gain(g0: np.ndarray, g1: np.ndarray) -> float:
        lit = g0 > 40
        if np.count_nonzero(lit) < 0.05 * g0.size:
            return 1.0
        ratio = g1[lit].astype(np.float32) / g0[lit].astype(np.float32)
        median = float(np.median(ratio))
        support = float(np.mean(np.abs(ratio - median) < 0.10))
        return median if (support > 0.5 and 0.4 <= median <= 2.2) else 1.0

    def _jitter_explained_ratio(self, ref_bgr: np.ndarray, dst_bgr: np.ndarray) -> float:
        H, W = dst_bgr.shape[:2]
        if H <= 2 or W <= 2:
            return 1.0
        center_dst = dst_bgr[1:H - 1, 1:W - 1]
        best = None
        for jy in (-1, 0, 1):
            for jx in (-1, 0, 1):
                src = ref_bgr[1 + jy:H - 1 + jy, 1 + jx:W - 1 + jx]
                d = cv2.absdiff(src, center_dst).max(axis=2)
                best = d if best is None else np.minimum(best, d)
        return float(np.mean(best <= self.comp_threshold)) if best is not None else 0.0

    @staticmethod
    def _tokenize(text: str) -> Set[str]:
        return {tok for tok in re.split(r"[^a-z0-9]+", text.lower()) if len(tok) > 1}

    @staticmethod
    def _dice_similarity(a: Set[str], b: Set[str]) -> float:
        if not a or not b:
            return 0.0
        return (2.0 * len(a & b)) / float(len(a) + len(b))

    @staticmethod
    def _bbox_iou(a: BBox, b: BBox) -> float:
        ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
        iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
        inter = ix * iy
        union = a[2] * a[3] + b[2] * b[3] - inter
        return inter / float(max(1, union))

    @staticmethod
    def _norm_bbox_iou(a: Dict[str, float], b: Dict[str, float]) -> float:
        ix = max(0.0, min(a["x"] + a["w"], b["x"] + b["w"]) - max(a["x"], b["x"]))
        iy = max(0.0, min(a["y"] + a["h"], b["y"] + b["h"]) - max(a["y"], b["y"]))
        inter = ix * iy
        union = a["w"] * a["h"] + b["w"] * b["h"] - inter
        return inter / union if union > 1e-6 else 0.0