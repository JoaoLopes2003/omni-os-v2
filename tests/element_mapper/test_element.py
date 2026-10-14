import cv2
import json
import os
from dotenv import load_dotenv, find_dotenv

from services.mapper.element_agent import ElementEngine

load_dotenv(find_dotenv())

def draw_elements(img, data, w, h, offset_x=0, offset_y=0):
    """Draws standard elements and the template collection mask onto an image."""
    # Draw Standard Elements (Already using relative math, no changes needed here)
    for el in data.get("elements", []):
        cx = offset_x + int(el["rel_x"] * w)
        cy = offset_y + int(el["rel_y"] * h)
        cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)
        cv2.putText(img, el["id"], (cx + 8, cy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    # Draw Template Collection Bounds
    template = data.get("item_template")
    if template and "collection_bounds" in template:
        cb = template["collection_bounds"]
        
        # Convert relative template bounds back to absolute pixels based on the crop size
        bx = offset_x + int(cb["x"] * w)
        by = offset_y + int(cb["y"] * h)
        bw = int(cb["w"] * w)
        bh = int(cb["h"] * h)
        
        # Draw the masking zone in magenta
        cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (255, 0, 255), 2)
        
        # Add a semi-transparent overlay
        overlay = img.copy()
        cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), (255, 0, 255), -1)
        cv2.addWeighted(overlay, 0.15, img, 0.85, 0, img)
        
        label = f"Template Zone: {template.get('template_id', 'collection')}"
        cv2.putText(img, label, (bx + 2, by + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

def run_automated_element_mapping(file_name: str, limit_containers: int = None):
    img_dir = "tests/layout_mapper/inputs"
    json_dir = "tests/layout_mapper/outputs"

    img_path = os.path.join(img_dir, f"{file_name}.png")
    json_path = os.path.join(json_dir, f"{file_name}.json")
    
    output_dir = os.path.join("tests/element_mapper/outputs", file_name)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading Layout Data from {json_path}...")
    with open(json_path, "r", encoding="utf-8") as f:
        layout_payload = json.load(f)
        
    containers = layout_payload.get("layout", {}).get("containers", [])
    if limit_containers is not None:
        containers = containers[:limit_containers]
        
    full_img = cv2.imread(img_path)
    if full_img is None:
        print(f"Error: Could not load image at {img_path}")
        return
        
    img_h, img_w = full_img.shape[:2]  # Get absolute dimensions
    full_img_drawn = full_img.copy()
    engine = ElementEngine()
    
    total_cost = 0.0
    total_tokens = {"input_tokens": 0, "output_tokens": 0}
    unified_container_outputs = []
    
    for i, container in enumerate(containers):
        c_id = container["id"]
        c_desc = container["description"]
        bbox = container["bbox"]
        
        # Convert layout relative coordinates to absolute pixels for cropping
        x = int(bbox["x"] * img_w)
        y = int(bbox["y"] * img_h)
        w = int(bbox["w"] * img_w)
        h = int(bbox["h"] * img_h)
        
        print(f"\n[{i+1}/{len(containers)}] Processing Container: '{c_id}'")
        
        # Crop the container from the full image
        crop_img = full_img[y:y+h, x:x+w]
        temp_crop_path = os.path.join(output_dir, f"temp_{c_id}.png")
        cv2.imwrite(temp_crop_path, crop_img)
        
        # Run Element Extraction
        reasoning, final_data, tokens = engine.extract_elements(
            cropped_image_path=temp_crop_path,
            container_id=c_id,
            container_description=c_desc
        )
        
        cost = (tokens["input_tokens"] / 1_000_000) * 2.00 + (tokens["output_tokens"] / 1_000_000) * 12.00
        total_cost += cost
        total_tokens["input_tokens"] += tokens["input_tokens"]
        total_tokens["output_tokens"] += tokens["output_tokens"]
        
        container_payload = {
            "container_id": c_id,
            "thought_process": reasoning,
            "container_data": final_data,
            "metrics": {"token_usage": tokens, "estimated_cost_usd": cost}
        }
        unified_container_outputs.append(container_payload)
        
        json_out_path = os.path.join(output_dir, f"{c_id}.json")
        with open(json_out_path, "w", encoding="utf-8") as f:
            json.dump(container_payload, f, indent=2)
            
        # Draw on the individual crop and save
        drawn_crop = crop_img.copy()
        draw_elements(drawn_crop, final_data, w, h, offset_x=0, offset_y=0)
        crop_out_path = os.path.join(output_dir, f"{c_id}.png")
        cv2.imwrite(crop_out_path, drawn_crop)
        
        # Draw on the full master image
        draw_elements(full_img_drawn, final_data, w, h, offset_x=x, offset_y=y)
        
        if os.path.exists(temp_crop_path):
            os.remove(temp_crop_path)
            
        print(f"  -> Saved {c_id}.png and {c_id}.json")

    # Save the master compiled image
    master_img_path = os.path.join(output_dir, f"{file_name}_full_mapped.png")
    cv2.imwrite(master_img_path, full_img_drawn)
    
    # Save the unified elements JSON map
    unified_payload = {
        "view_id": file_name,
        "container_outputs": unified_container_outputs,
        "session_metrics": {
            "total_tokens": total_tokens,
            "total_cost_usd": total_cost
        }
    }
    unified_json_path = os.path.join(output_dir, f"{file_name}_elements.json")
    with open(unified_json_path, "w", encoding="utf-8") as f:
        json.dump(unified_payload, f, indent=2)
    
    print("\n==========================================")
    print("Element Mapping Complete!")
    print(f"Master Image saved to: {master_img_path}")
    print(f"Unified Element Map saved to: {unified_json_path}")
    print(f"Total Tokens Used: {total_tokens}")
    print(f"Total Session Cost: ${total_cost:.6f}")
    print("==========================================")

if __name__ == "__main__":
    TARGET_FILE_NAME = "vscode_test"
    LIMIT_CONTAINERS = 15
    run_automated_element_mapping(file_name=TARGET_FILE_NAME, limit_containers=LIMIT_CONTAINERS)