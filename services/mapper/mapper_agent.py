# Inside services/mapper/mapper_agent.py

# 1. Cold-Start (0 Memory): Baseline prompts & schemas are 100% untouched
layout_view, _ = layout_engine.extract_layout(
    image_path=frame_t_path,
    process_name="spotify",
    is_masked_update=False,
)
for container in layout_view["containers"]:
    crop_path = crop_container(frame_t_path, container["bbox"])
    reasoning, extraction, _ = element_engine.extract_elements(
        cropped_image_path=crop_path,
        container_id=container["id"],
        container_description=container["description"],
        candidates=None,
        enable_candidate_matching=False,  # <-- Exact original prompt & schema
    )
    library.upsert_from_extraction(container, extraction, bgr_t_target)

# 2. Incremental Update at T+1 (when StateTransitionTracker reports unexplained regions):
masked_t1_path = library.build_masked_image(
    frame_t1_path=frame_t1_path,
    verified_containers=verified_intact_containers,
    unmasked_holes_norm=[result.target_crop_bbox] if result.target_crop_bbox else None,
)
inc_view, _ = layout_engine.extract_layout(
    image_path=masked_t1_path,
    process_name="spotify",
    is_masked_update=True,
)
for new_c in inc_view["containers"]:
    top_candidates = library.get_top_candidates(new_c, action_context=last_action_ctx, top_n=3)
    crop_path = crop_container(frame_t1_path, new_c["bbox"])
    reasoning, extraction, _ = element_engine.extract_elements(
        cropped_image_path=crop_path,
        container_id=new_c["id"],
        container_description=new_c["description"],
        candidates=top_candidates,
        enable_candidate_matching=bool(top_candidates),  # <-- Activates Rule 6 only if candidates exist
    )
    node, variant, is_new = library.upsert_from_extraction(
        new_c, extraction, bgr_t1_target, action_context=last_action_ctx
    )