# Hardware Port Vision & Metadata Engine

An end-to-end computer vision and multimodal AI platform designed to detect, classify, ground, and annotate physical connection ports and interfaces across telecom equipment (5G Massive MIMO, Macro RRUs, baseband units), networking switches/routers, and consumer computing hardware.

---

## Key Capabilities

- **Zero-Shot & Few-Shot Vision Grounding**: Direct visual connector localization powered by Google Gemini Vision (`gemini-3.5-flash-lite`) and GPT-5.1 Vision with prompt-level domain constraints.
- **Negative Feature Suppression**: Explicit filtering to reject cooling fins, heatsink ribs, ventilation slits, wall mounts, and screws that typically confuse generic object detectors.
- **Telecom & Enterprise Port Vocabulary**: Standardized grounding for weather-sealed optical SFP/eCPRI interfaces, -48V DC power terminals, 100-240V AC mains connectors, 4.3-10 & 7/16 DIN RF coax connectors, AISG/RET control ports, dual-hole chassis ground lugs, RJ45 Ethernet, USB, HDMI, and console ports.
- **Dual Visual Output Formats**:
  - **OpenCV HUD Overlays**: Staggered badges, bounding boxes, and leader lines with collision avoidance.
  - **Technical Marketplace Diagrams**: Orthogonal non-crossing callout lines with clean TrueType typography and specifications.
- **Interactive Annotation Studio**: Web UI with click-and-drag box drawing, real-time draggable boxes, resize handles, and instant dataset re-exporting.
- **Multi-Format Dataset Export**: One-click generation of Ultralytics YOLOv8 datasets, CVAT for Images 1.1 XML, and COCO 1.0 JSON.

---

## System Architecture

```text
               User Image / PDF Upload
                         │
                         ▼
        ┌───────────────────────────────────┐
        │        app.py (Flask 5000)        │  ← REST API Gateway & Web UI
        └──────────────┬────────────────────┘
                       │
      ┌────────────────┼────────────────────┐
      ▼                ▼                    ▼
 pipeline.py     metadata_manager.py  pdf_extractor.py
 (Vision AI &    (Product Specs &     (Datasheet PDF
   Rendering)      Port Catalog)        Extraction)
      │
      ├──► Gemini / Vision LLM API (Structured Bounding Boxes)
      ├──► NMS & Box Deduplication (IoU 0.82)
      ├──► OpenCV & Marketplace Layout Engines
      └──► Exporters (YOLOv8, CVAT 1.1, COCO 1.0)
```

---

## Installation & Setup

### 1. Prerequisites
- Python 3.10+
- Linux / macOS / Windows
- OpenCV, PyMuPDF, Flask, Requests, python-dotenv, Ultralytics

Install required dependencies:
```bash
pip install flask opencv-python numpy requests python-dotenv PyMuPDF ultralytics pillow
```

### 2. Environment Configuration
The system uses environment variables loaded from `.env`. A sanitized template is provided in `.env.example`:

```bash
cp .env.example .env
```

Open `.env` and set your credentials:
```ini
# Google Gemini Vision Configuration
GEMINI_API_KEY=your_actual_gemini_api_key_here
GEMINI_MODEL=gemini-3.5-flash-lite
VISION_PROVIDER=gemini

# Application Port
PORT=5000

# Optional: OpenAI / GPT-5.1 Configuration
OPENAI_API_KEY=
OPENAI_MODEL=gpt-5.1
```

> [!IMPORTANT]
> Never commit your `.env` file to version control. The repository `.gitignore` is pre-configured to exclude all `.env*` files and secret tokens.

---

## Running the Application

Start the local dashboard server:
```bash
python3 app.py --port 5000
```

Once running, navigate to:
```text
http://localhost:5000
```

---

## Detection & Grounding Strategy

The core detection engine in [`pipeline.py`](pipeline.py) conditions multimodal LLMs with high-resolution imagery (1280px max-dimension @ 88% JPEG quality) and strict structured prompting:

1. **Chassis & Perspective Classification**: Evaluates whether the viewing angle physically exposes I/O connection panels (e.g., bottom plate) vs non-port perspectives (top cover, front LED faceplate, antenna radome).
2. **False-Positive Prevention**: Specifically directs the model to disregard parallel metallic grooves (heatsinks, cooling fins, thermal exhaust vents) that heuristic detectors often mistake for rows of ports.
3. **Structured Coordinate Grounding**: Extracts bounding boxes in normalized `[ymin, xmin, ymax, xmax]` range `[0, 1000]`.
4. **NMS Deduplication**: Applies Non-Maximum Suppression with an IoU threshold of 0.82 to merge multi-hit detections and unwraps nested coordinate lists.
5. **Class Canonicalization**: Normalizes labels into standard classes: `SFP`, `Power`, `RF`, `Console`, `Ground`, `RJ45`, `USB`, `Audio`, `HDMI`, `DVI`, `SD_Card`, `Other`.

> [!NOTE]
> Detailed system prompts and LLM instruction templates are maintained in `prompts/` and are protected from public exposure via `.gitignore`.

---

## REST API Reference

| Endpoint | Method | Description |
| :--- | :---: | :--- |
| `GET /api/devices` | GET | List all registered devices with detection status and specs. |
| `GET /api/device/<id>` | GET | Retrieve full JSON annotation and port list for a specific device. |
| `POST /api/upload` | POST | Upload a hardware image (`file`), trigger AI detection, and render diagrams. |
| `POST /api/device/<id>/update` | POST | Update edited port coordinates/types and trigger re-rendering of all diagrams. |
| `POST /api/device/<id>/reanalyze` | POST | Force AI re-analysis on an existing image. |
| `POST /api/upload_pdf` | POST | Upload a PDF datasheet to extract specs and port diagrams. |
| `GET /export/yolo` | GET | Download complete Ultralytics YOLOv8 dataset as a `.zip` archive. |
| `GET /export/cvat` | GET | Download CVAT for Images 1.1 XML annotation file. |
| `GET /export/coco` | GET | Download COCO 1.0 JSON annotation file. |

---

## Dataset Exports

Generated annotation files are continuously synchronized in `data/`:
- **Ultralytics YOLOv8**: `data/yolo_export/` (`data.yaml`, `images/train`, `images/val`, `labels/train`, `labels/val`)
- **CVAT for Images 1.1**: `data/cvat_export/cvat_annotations.xml`
- **COCO 1.0**: `data/cvat_export/coco_annotations.json`

---

## Security & Privacy

- All sensitive keys, tokens, and credentials are kept out of source code and loaded dynamically via `python-dotenv`.
- Proprietary prompt engineering files (`*PROMPT*.md`, `prompts/`) and environment configurations (`.env`) are excluded by `.gitignore`.
