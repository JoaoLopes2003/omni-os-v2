import cv2
import json
import os
from dotenv import load_dotenv, find_dotenv

from services.mapper.layout_agent import LayoutEngine

load_dotenv(find_dotenv())

def test_layout_mapping(image_path: str, process_name: str):
    print(f"Testing Layout Mapper on {image_path}...")
    
    engine = LayoutEngine()
    final_view, tokens = engine.extract_layout(image_path, process_name)
    
    # Calculate cost (gemini-3.1-pro-preview rates: $2.00/1M input, $12.00/1M output)
    cost = (tokens["input_tokens"] / 1_000_000) * 2.00 + (tokens["output_tokens"] / 1_000_000) * 12.00
    
    print("\n--- LLM JSON OUTPUT ---")
    print(json.dumps(final_view, indent=2))
    print(f"\nTokens used: {tokens}")
    print(f"Estimated Cost: ${cost:.6f}")
    
    # Draw the results to verify accuracy
    img = cv2.imread(image_path)
    
    for container in final_view["containers"]:
        bbox = container["bbox"]
        x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
        
        # Draw bounding box
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Add label background and text
        label = f"{container['id']} ({container['container_type']})"
        cv2.rectangle(img, (x, y-20), (x + len(label)*10, y), (0, 255, 0), -1)
        cv2.putText(img, label, (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Resolve output directory by replacing 'inputs' with 'outputs' in the path
    output_dir = os.path.dirname(image_path).replace("inputs", "outputs")
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.basename(image_path)
    name_only = os.path.splitext(base_name)[0]
    
    output_img_path = os.path.join(output_dir, base_name)
    output_json_path = os.path.join(output_dir, f"{name_only}.json")
    
    # Save the visual artifact
    cv2.imwrite(output_img_path, img)
    
    # Save the JSON payload including metrics
    full_payload = {
        "layout": final_view,
        "metrics": {
            "token_usage": tokens,
            "estimated_cost_usd": cost
        }
    }
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(full_payload, f, indent=2)
        
    print(f"\nSaved visual artifact to {output_img_path}.")
    print(f"Saved JSON payload to {output_json_path}.")

if __name__ == "__main__":
    # Updated path to match the inputs directory structure
    test_layout_mapping("tests/layout_mapper/inputs/vscode_test_2.png", "spotify.exe")