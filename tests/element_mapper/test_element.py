import cv2
import json
import os
from dotenv import load_dotenv, find_dotenv

from services.mapper.element_agent import ElementEngine

load_dotenv(find_dotenv())

def draw_elements(img, data, w, h, offset_x=0, offset_y=0):
    """Draws standard elements and the template collection mask onto an image."""
    # Draw Standard Elements
    for el in data.get("elements", []):
        cx = offset_x + int(el["rel_x"] * w)
        cy = offset_y + int(el["rel_y"] * h)
        cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)
        cv2.putText(img, el["id"], (cx + 8, cy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    # Draw Template Collection Bounds (Masking Zone)
    template = data.get("item_template")
    if template and "collection_bounds" in template:
        cb = template["collection_bounds"]
        bx = offset_x + cb["x"]
        by = offset_y + cb["y"]
        bw = cb["w"]
        bh = cb["h"]
        
        # Draw the masking zone in magenta
        cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (255, 0, 255), 2)
        
        # Add a semi-transparent overlay for visual verification of the mask
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
        
    full_img_drawn = full_img.copy()
    engine = ElementEngine()
    
    total_cost = 0.0
    total_tokens = {"input_tokens": 0, "output_tokens": 0}
    
    for i, container in enumerate(containers):
        c_id = container["id"]
        c_desc = container["description"]
        bbox = container["bbox"]
        x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
        
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
        
        # Save individual container JSON
        container_payload = {
            "container_id": c_id,
            "thought_process": reasoning,
            "container_data": final_data,
            "metrics": {"token_usage": tokens, "estimated_cost_usd": cost}
        }
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
        
        # Clean up the raw temporary crop
        if os.path.exists(temp_crop_path):
            os.remove(temp_crop_path)
            
        print(f"  -> Saved {c_id}.png and {c_id}.json")

    # Save the master compiled image
    master_img_path = os.path.join(output_dir, f"{file_name}_full_mapped.png")
    cv2.imwrite(master_img_path, full_img_drawn)
    
    print("\n==========================================")
    print("Element Mapping Complete!")
    print(f"Master Image saved to: {master_img_path}")
    print(f"Total Tokens Used: {total_tokens}")
    print(f"Total Session Cost: ${total_cost:.6f}")
    print("==========================================")

if __name__ == "__main__":
    # Define your variables here
    TARGET_FILE_NAME = "spotify_test"
    LIMIT_CONTAINERS = 15  # Set to None to process all containers in the JSON
    
    run_automated_element_mapping(
        file_name=TARGET_FILE_NAME, 
        limit_containers=LIMIT_CONTAINERS
    )