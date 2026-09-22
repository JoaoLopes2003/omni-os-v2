# Omni-OS

Omni-OS is a multimodal Vision-RPA (Robotic Process Automation) agent designed to autonomously navigate and interact with desktop operating systems. 

By combining optical character recognition (OCR) with Set-of-Marks (SoM) visual prompting and native Vision-Language Model (VLM) point grounding, Omni-OS overcomes the traditional latency and inaccuracy trade-offs of pure coordinate-guessing or pure DOM/accessibility trees.

---

## Key Architectural Highlights

* **Hybrid Perception Pipeline**:
  * **Deterministic Text Grounding**: Uses **EasyOCR** and **Set-of-Marks (SoM)** to extract and label interactable text elements with pixel-accurate bounding boxes.
  * **Probabilistic Icon Pointing (Fallback)**: When interacting with non-text visual elements (dock icons, logos, controls), the planner falls back to normalized $1000 \times 1000$ spatial point grounding (`click_point`).
* **Multi-Monitor Display Server Compensation**:
  * Automatically inspects monitor geometry via `mss` display dictionaries.
  * Maps local monitor crop coordinates directly to the operating system's global Virtual Desktop space ($x_{\text{offset}}, y_{\text{offset}}$).
* **Agentic Loop & Loop-Breaking**:
  * Maintains rolling chronological action memory (`action_history`) to prevent infinite click cycles.
  * Explicit state protocols supporting `click`, `type`, `wait`, `done`, and `abort`.
* **Telemetry & Debugging**:
  * Dumps frame captures, annotated SoM frames, extracted OCR manifests, raw API prompts, and timing breakdowns per step.

---

## Architecture Overview

                    ┌────────────────────────┐
                    │   ScreenCaptureEngine  │ (mss)
                    └───────────┬────────────┘
                                │ Raw Frame
                                ▼
                    ┌────────────────────────┐
                    │  TextPerceptionEngine  │ (EasyOCR)
                    └───────────┬────────────┘
                                │ BoundingBoxes + Text
                                ▼
                    ┌────────────────────────┐
                    │      SomRenderer       │ (OpenCV Set-of-Marks)
                    └───────────┬────────────┘
                                │ Annotated UI + Element Dict
                                ▼
                    ┌────────────────────────┐
                    │    OSPlannerEngine     │ (Gemini 3.1 Pro)
                    └───────────┬────────────┘
                                │ Pydantic ActionPlan
                                ▼
                    ┌────────────────────────┐
                    │      ActionDriver      │ (pyautogui)
                    └────────────────────────┘

---

## Tech Stack

* **Language**: Python 3.12+
* **Reasoning / VLM**: Google Gemini 3.1 Pro (`gemini-3.1-pro-preview` via `google-genai`)
* **Perception**: EasyOCR, OpenCV (`opencv-python`), Pillow (`PIL`)
* **Screen Capture**: `mss`
* **OS Automation**: `pyautogui`
* **Data Validation**: `pydantic` v2

---

## Installation & Setup

### 1. Clone Repository & Setup Virtual Environment

```bash
git clone git@github.com:JoaoLopes2003/omni-os.git
cd omni-os

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
### 2. Configure Environment Variables

Create a .env file in the root directory:

```bash
GEMINI_API_KEY="your-google-gemini-api-key"
```

---

## Project Structure

    omni-os/
    ├── .gitignore
    ├── requirements.txt
    ├── README.md
    └── src/
        ├── core/
        │   ├── __init__.py
        │   ├── capture.py        # mss screen capture & monitor offset detection
        │   └── schemas.py        # Pydantic data schemas (BoundingBox, UIElement)
        ├── perception/
        │   ├── __init__.py
        │   ├── ocr.py            # EasyOCR text extraction & bbox normalization
        │   └── som.py            # Set-of-Marks visual bounding box & tag overlay
        ├── reasoning/
        │   ├── __init__.py
        │   └── llm_planner.py    # VLM interface & prompt orchestration (Gemini)
        ├── execution/
        │   ├── __init__.py
        │   └── driver.py         # OS input execution with failsafe & offsets
        └── main.py               # Autonomous perception-reasoning-action orchestrator

---

## Usage

Run the autonomous loop from the root directory:

```bash
python3 -m src.main
```

When prompted, enter a high-level command (e.g., "Open Firefox and search for GitHub" or "Play a song on Spotify").

Emergency hardware failsafe: To immediately abort agent control, slam your physical mouse cursor into any corner of the screen (pyautogui.FAILSAFE).