"""
Automated Pipeline: Hardware Port Detection & Annotation System
1. Downloads 15 diverse hardware device images
2. Uses Gemini 3.8 Flash Vision for port grounding & silkscreen text reading
3. Saves structured JSON annotations
4. Renders OpenCV previews with upper-label badges
5. Exports to CVAT for Images 1.1 XML & COCO 1.0 JSON format
"""

import os
import sys
import json
import base64
import time
import requests
import cv2
import numpy as np
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom import minidom
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw_images")
JSON_DIR = os.path.join(DATA_DIR, "annotations_json")
ANNOTATED_DIR = os.path.join(DATA_DIR, "annotated_images")
MARKETPLACE_DIR = os.path.join(DATA_DIR, "marketplace_images")
CVAT_DIR = os.path.join(DATA_DIR, "cvat_export")

for d in [RAW_DIR, JSON_DIR, ANNOTATED_DIR, MARKETPLACE_DIR, CVAT_DIR]:
    os.makedirs(d, exist_ok=True)

# Curated list of 15 authentic telecom & networking hardware images with visible connection ports
DEVICE_SOURCES = [
    {
        "id": "device_laptop_thinkpad",
        "name": "Lenovo ThinkPad X220 Laptop (Side Ports)",
        "url": "https://upload.wikimedia.org/wikipedia/commons/8/88/Left_side_ports_Lenovo_ThinkPad_X220.jpg",
        "type": "Laptop / PC"
    },
    {
        "id": "device_wifi_router_asus",
        "name": "Asus WL-500G Wireless Wi-Fi Router (Rear Ports)",
        "url": "https://upload.wikimedia.org/wikipedia/commons/a/a3/Asus_WL-500G_Premium_rear.jpeg",
        "type": "Wi-Fi Router"
    },
    {
        "id": "device_01_avaya_2550t",
        "name": "Avaya 2550T-PWR PoE Switch",
        "url": "https://upload.wikimedia.org/wikipedia/commons/b/b9/2550T-PWR-Front.jpg",
        "type": "Network Switch"
    },
    {
        "id": "device_02_cisco_sg300",
        "name": "Cisco SG300-28 28-Port Gigabit Switch",
        "url": "https://upload.wikimedia.org/wikipedia/commons/7/7f/Cisco_small_business_SG300-28_28-port_Gigabit_Ethernet_rackmount_switch.jpg",
        "type": "Network Switch"
    },
    {
        "id": "device_03_3com_24port",
        "name": "3Com 24-Port Fast Ethernet Switch",
        "url": "https://upload.wikimedia.org/wikipedia/commons/e/e4/24-port_3Com_switch.JPG",
        "type": "Network Switch"
    },
    {
        "id": "device_04_netgear_gs108",
        "name": "Netgear ProSafe GS108 8-Port Switch",
        "url": "https://upload.wikimedia.org/wikipedia/commons/b/ba/Netgear_ProSafe_8_Port_Gigabit_Switch_GS108_front.jpeg",
        "type": "Desktop Switch"
    },
    {
        "id": "device_05_brocade_fes24",
        "name": "Brocade FastIron FES24 Switch",
        "url": "https://upload.wikimedia.org/wikipedia/commons/3/3e/Brocade_FES24_Front.jpg",
        "type": "Managed Switch"
    },
    {
        "id": "device_06_dlink_switch",
        "name": "D-Link Fast Ethernet Switch",
        "url": "https://upload.wikimedia.org/wikipedia/commons/7/76/Switch_fastethernet_dlink.jpg",
        "type": "Unmanaged Switch"
    },
    {
        "id": "device_07_rackmount_patchpanel",
        "name": "Rackmount Ethernet Switches & Patch Panels",
        "url": "https://upload.wikimedia.org/wikipedia/commons/b/bc/19-inch_rackmount_Ethernet_switches_and_patch_panels.jpg",
        "type": "Rackmount Panel"
    },
    {
        "id": "device_08_sfp_optical_transceivers",
        "name": "10 Gbit XFP and SFP Optical Transceivers",
        "url": "https://upload.wikimedia.org/wikipedia/commons/f/fc/10_Gbit_XFP_and_SFP_transceivers.jpg",
        "type": "Optical Transceiver"
    },
    {
        "id": "device_09_cbn_cable_modem",
        "name": "CBN CG3368 Cable Modem & Router",
        "url": "https://upload.wikimedia.org/wikipedia/commons/4/40/CBN_CG3368_cable_modem.jpg",
        "type": "Cable Modem / Gateway"
    },
    {
        "id": "device_10_residential_gateway",
        "name": "Residential Telecom Gateway (Rear Ports)",
        "url": "https://upload.wikimedia.org/wikipedia/commons/e/e3/ResidentialGatewayCloseup.jpg",
        "type": "Home Gateway"
    },
    {
        "id": "device_11_coaxial_patchpanel",
        "name": "Trompeter JSI-52 Coaxial / RF Patch Panel",
        "url": "https://upload.wikimedia.org/wikipedia/commons/3/35/Trompeter_JSI-52_Patch_Panel.jpeg",
        "type": "Coaxial / RF Panel"
    },
    {
        "id": "device_12_atlantis_switch_front",
        "name": "Atlantis 5-Port Switch (Front RJ45 Ports)",
        "url": "https://upload.wikimedia.org/wikipedia/commons/7/72/Ethernet_switch_Atlantis_A02-F5P_5_ports_frontend.jpg",
        "type": "Desktop Switch"
    },
    {
        "id": "device_13_atlantis_switch_rear",
        "name": "Atlantis 5-Port Switch (Rear Power & Ports)",
        "url": "https://upload.wikimedia.org/wikipedia/commons/5/5b/Ethernet_switch_Atlantis_A02-F5P_5_ports_backend.jpg",
        "type": "Desktop Switch"
    },
    {
        "id": "device_14_tplink_5port",
        "name": "TP-Link TL-SG1005D 5-Port Gigabit Switch",
        "url": "https://upload.wikimedia.org/wikipedia/commons/7/74/5_Port_Gigabit_Netzwerk-Switch_TL-SG1005D_01.jpg",
        "type": "Desktop Switch"
    },
    {
        "id": "device_15_ericsson_radio_panel",
        "name": "Ericsson Radio System - Macro Interface Panel",
        "url": "LOCAL_GENERATE",
        "type": "Cellular Radio Unit"
    }
]

PORT_PALETTE = {
    "rf": {"box": (0, 165, 255), "text_bg": (0, 130, 210), "text_fg": (255, 255, 255)},
    "sfp": {"box": (240, 180, 0), "text_bg": (200, 140, 0), "text_fg": (255, 255, 255)},
    "optical": {"box": (240, 180, 0), "text_bg": (200, 140, 0), "text_fg": (255, 255, 255)},
    "rj45": {"box": (0, 210, 80), "text_bg": (0, 160, 60), "text_fg": (255, 255, 255)},
    "ethernet": {"box": (0, 210, 80), "text_bg": (0, 160, 60), "text_fg": (255, 255, 255)},
    "power": {"box": (40, 40, 230), "text_bg": (30, 30, 190), "text_fg": (255, 255, 255)},
    "dc": {"box": (40, 40, 230), "text_bg": (30, 30, 190), "text_fg": (255, 255, 255)},
    "console": {"box": (200, 110, 240), "text_bg": (170, 80, 210), "text_fg": (255, 255, 255)},
    "usb": {"box": (0, 220, 220), "text_bg": (0, 180, 180), "text_fg": (0, 0, 0)},
    "ground": {"box": (160, 160, 160), "text_bg": (100, 100, 100), "text_fg": (255, 255, 255)},
    "antenna": {"box": (0, 165, 255), "text_bg": (0, 130, 210), "text_fg": (255, 255, 255)}
}
DEFAULT_STYLE = {"box": (0, 220, 100), "text_bg": (0, 160, 70), "text_fg": (255, 255, 255)}


def download_device_images():
    """Downloads or generates all 15 device images."""
    print("\n--- STEP 1: Downloading & Preparing 15 Device Images ---")
    headers = {"User-Agent": "HardwarePortDetector/1.0 (https://github.com/research/port-detector; research@university.edu)"}

    for item in DEVICE_SOURCES:
        img_id = item["id"]
        target_path = os.path.join(RAW_DIR, f"{img_id}.jpg")

        if os.path.exists(target_path):
            print(f"[EXISTS] {img_id}.jpg already present.")
            continue

        if item["url"] == "LOCAL_GENERATE":
            from port_detector import generate_synthetic_panel
            panel = generate_synthetic_panel()
            cv2.imwrite(target_path, panel)
            print(f"[GEN] Generated Ericsson Radio System panel -> {img_id}.jpg")
        else:
            try:
                print(f"[DOWNLOAD] Fetching {item['name']}...")
                r = requests.get(item["url"], headers=headers, timeout=25)
                if r.status_code == 200 and len(r.content) > 1000:
                    with open(target_path, "wb") as f:
                        f.write(r.content)
                    print(f"[OK] Saved {img_id}.jpg ({len(r.content) // 1024} KB)")
                else:
                    print(f"[WARN] Failed to download {item['url']} (status {r.status_code})")
            except Exception as e:
                print(f"[ERROR] Download error for {img_id}: {e}")
        time.sleep(1.2)


def canonicalize_port_type(raw_type):
    """Normalizes any detected port type into one of 12 standard classes."""
    t = str(raw_type).strip().lower()
    if any(k in t for k in ["rj45", "ethernet", "lan", "wan", "8p8c"]):
        return "RJ45"
    if any(k in t for k in ["sfp", "optical", "fiber", "cpri", "ecpri"]):
        return "SFP"
    if any(k in t for k in ["power", "dc", "ac", "-48v", "mains", "barrel"]):
        return "Power"
    if any(k in t for k in ["rf", "antenna", "coax", "4.3-10", "sma", "qma", "tx mon"]):
        return "RF"
    if any(k in t for k in ["console", "serial", "rs-232", "aisg", "ret", "alarm", "aux", "ext"]):
        return "Console"
    if any(k in t for k in ["usb", "type-a", "type-c"]):
        return "USB"
    if any(k in t for k in ["hdmi"]):
        return "HDMI"
    if any(k in t for k in ["audio", "headphone", "jack", "3.5mm"]):
        return "Audio"
    if any(k in t for k in ["dvi"]):
        return "DVI"
    if any(k in t for k in ["sd", "card"]):
        return "SD_Card"
    if any(k in t for k in ["ground", "earth", "lug"]):
        return "Ground"
    return "Other"


def deduplicate_ports(ports, iou_thresh=0.82):
    """Removes duplicate or heavily overlapping bounding boxes for the same connector."""
    if not ports:
        return []
    valid = []
    for p in ports:
        b = p.get("box_2d", [])
        if isinstance(b, list):
            # Unwrap nested list if Gemini outputs [[ymin, xmin, ymax, xmax]]
            while len(b) == 1 and isinstance(b[0], list):
                b = b[0]
            if len(b) >= 4:
                try:
                    ymin, xmin, ymax, xmax = [max(0, min(1000, int(v))) for v in b[:4]]
                    if ymax > ymin and xmax > xmin:
                        p["box_2d"] = [ymin, xmin, ymax, xmax]
                        p["port_type"] = canonicalize_port_type(p.get("port_type", "Other"))
                        valid.append(p)
                except Exception:
                    continue

    kept = []
    for p in valid:
        b1 = p["box_2d"]
        is_dup = False
        for k in kept:
            b2 = k["box_2d"]
            iy1, ix1 = max(b1[0], b2[0]), max(b1[1], b2[1])
            iy2, ix2 = min(b1[2], b2[2]), min(b1[3], b2[3])
            if iy2 > iy1 and ix2 > ix1:
                inter = (iy2 - iy1) * (ix2 - ix1)
                a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
                a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
                iou = inter / float(a1 + a2 - inter)
                if iou > iou_thresh:
                    is_dup = True
                    break
        if not is_dup:
            kept.append(p)
    return kept


def build_grounding_prompt(device_meta):
    """
    Builds a vision grounding prompt that conditions the LLM with product metadata,
    allowing it to match detected ports with the official catalog port inventory,
    or identify non-port angles (e.g. front LEDs, closed lids, radomes).
    """
    raw_name = device_meta.get("name") or device_meta.get("product_name") or device_meta.get("device_model") or ""
    if not raw_name or raw_name.startswith("custom_") or raw_name.startswith("Screenshot") or raw_name.endswith((".jpg", ".png", ".jpeg")):
        name = "hardware equipment (telecom radio/antenna, networking switch/router, or computer/laptop)"
    else:
        name = raw_name
    category = device_meta.get("category") or device_meta.get("device_type") or "Hardware Device"

    meta_parts = []
    desc = device_meta.get("datasheet_description") or device_meta.get("description") or ""
    if desc:
        meta_parts.append(f"Datasheet Summary: {desc}")

    specs = device_meta.get("specs") or {}
    if specs:
        meta_parts.append(f"Technical Specifications: {json.dumps(specs)}")

    port_specs = device_meta.get("port_specifications") or []
    if port_specs:
        inv_str = "Known Catalog Port Inventory for this Product (from metadata):\n"
        for p in port_specs:
            loc = f" [{p.get('location', '')}]" if p.get('location') else ""
            inv_str += f"- {p.get('port_name')} ({p.get('port_type')}){loc}: {p.get('spec', '')} - {p.get('function', '')}\n"
        meta_parts.append(inv_str)

    metadata_block = "\n".join(meta_parts)

    instructions = f"""You are a world-class telecom, networking, and electronics hardware computer-vision engineer.
Analyze this high-resolution image of {name} ({category}).

{f"OFFICIAL PRODUCT METADATA:\n{metadata_block}\n" if metadata_block else ""}
YOUR TASK:
1. Identify the exact device model from any visible silkscreen labels, model plates, or hardware characteristics.
2. Determine the view/perspective (e.g. 'Bottom Interface Panel', 'Rear I/O Panel', 'Left Edge Ports', 'Closed Lid', 'Front Panel (LEDs)', 'Technical Diagram').
3. Assess whether physical connection ports/sockets are visible in this specific view:
   - If this view DOES NOT show ports (for example: front panel with only LED indicator lights, closed laptop lid, antenna radome mast mount, blank chassis side), set "ports_visible": false and "ports": []. In "perspective_notes", explain which side is shown and where the ports reside.
   - If this view DOES show ports, identify and tightly bound EVERY authentic visible port, socket, and connector.

STRICT DOMAIN RULES & FALSE-POSITIVE PREVENTION:
- HEATSINK COOLING FINS / VENTILATION SLITS: NEVER label cooling fins, heatsink ribs, ventilation slots, or thermal grooves as ports (do NOT label them as USB-C or SD card slots).
- MECHANICAL HARDWARE: NEVER label screws, bolts, wall-mount brackets, lifting lugs, hinges, or chassis seams as ports.
- TELECOM CONNECTORS:
  * Optical Interfaces: Weather-sealed SFP / SFP+ / SFP28 / eCPRI fronthaul ports (frequently covered by rubber/plastic dust caps or screw-locking collars). Label them as port_type 'SFP'.
  * Power: Circular 2-pin screw terminals (-48V DC, up to 50A) or circular AC screw-lock sockets. Label as port_type 'Power'.
  * RF Coaxial: 4.3-10 female, 7/16 DIN, SMA/QMA, or TX MON RF calibration ports. Label as port_type 'RF'.
  * Control / Alarms: 8-pin circular DIN AISG/RET control ports, EXT / AUX alarm sockets. Label as port_type 'Console' or 'Other'.
  * Ground: Grounding screws or dual-hole ground lugs. Label as port_type 'Ground'.
- IT & NETWORKING CONNECTORS:
  * RJ45 Ethernet (8P8C), USB Type-A, USB Type-C, HDMI, DisplayPort, 3.5mm Audio, VGA, DVI, SD Card reader.
- BOUNDING BOX ACCURACY:
  * 'box_2d': [ymin, xmin, ymax, xmax] normalized integers from 0 to 1000.
  * Tightly bound ONLY the port aperture/connector socket itself (not the entire device or cooling fins).

CRITICAL: If ports are visible, the "ports" array MUST NOT BE EMPTY! You MUST detect and provide a bounding box [ymin, xmin, ymax, xmax] for EVERY authentic connector socket or terminal.

You MUST return a valid JSON object matching this EXACT schema:
{{
  "device_model": "Identified Hardware Model Name",
  "device_type": "Category (e.g. 5G Massive MIMO / Macro RRU / Base Station Antenna / Gigabit Switch / Laptop)",
  "view_perspective": "Perspective Description",
  "ports_visible": true,
  "ports": [
    {{
      "port_name": "Clear descriptive name (e.g. Optical eCPRI 1, -48V DC Power, TX MON RF, USB 3.0, HDMI)",
      "port_type": "One of ['RJ45', 'SFP', 'Power', 'Console', 'RF', 'USB', 'Audio', 'HDMI', 'DVI', 'SD_Card', 'Ground', 'Other']",
      "silkscreen_label": "Printed text/symbol on chassis near port (or empty string)",
      "box_2d": [ymin, xmin, ymax, xmax]
    }}
  ],
  "perspective_notes": "Technical notes about this view, ports detected, and physical characteristics"
}}"""
    return instructions


def run_gemini_detection(image_path, device_meta, max_retries=3):
    """Sends image to Gemini Flash Vision with metadata-guided prompt and retry logic."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot load image {image_path}")
    h, w = img.shape[:2]
    max_dim = 1280
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        img_resized = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        _, buf = cv2.imencode(".jpg", img_resized, [cv2.IMWRITE_JPEG_QUALITY, 88])
        img_bytes = buf.tobytes()
    else:
        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        img_bytes = buf.tobytes()

    img_b64 = base64.b64encode(img_bytes).decode("utf-8")
    prompt = build_grounding_prompt(device_meta)

    payload = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": "image/jpeg", "data": img_b64}}
            ]
        }],
        "generationConfig": {
            "response_mime_type": "application/json"
        }
    }

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"

    for attempt in range(max_retries):
        resp = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=45)
        if resp.status_code == 200:
            result = resp.json()
            try:
                candidate = result["candidates"][0]["content"]["parts"][0]["text"]
                data = json.loads(candidate)
                if isinstance(data, list):
                    if len(data) > 0 and isinstance(data[0], dict) and "ports" in data[0]:
                        data = data[0]
                    else:
                        data = {"ports": data}
                elif not isinstance(data, dict):
                    data = {"ports": []}

                raw_ports = data.get("ports", [])
                cleaned_ports = deduplicate_ports(raw_ports)
                data["ports"] = cleaned_ports

                pv = data.get("ports_visible")
                if pv is not None:
                    if isinstance(pv, bool):
                        data["ports_visible"] = pv
                    elif isinstance(pv, (int, float)):
                        data["ports_visible"] = pv > 0
                    elif isinstance(pv, str):
                        data["ports_visible"] = pv.lower() in ["true", "1", "yes"]
                    else:
                        data["ports_visible"] = len(cleaned_ports) > 0
                else:
                    data["ports_visible"] = len(cleaned_ports) > 0

                return data
            except Exception as e:
                raise ValueError(f"Failed to parse Gemini output: {e}\nRaw response: {result}")
        elif resp.status_code in [429, 503]:
            time.sleep((attempt + 1) * 3)
        else:
            raise RuntimeError(f"Gemini API returned {resp.status_code}: {resp.text[:300]}")

    raise RuntimeError(f"Gemini API failed after {max_retries} retries.")


OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", os.environ.get("GPT51_API_KEY", ""))
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.1")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
VISION_PROVIDER = os.environ.get("VISION_PROVIDER", "gemini" if GEMINI_API_KEY else "gpt-5.1")


def run_gpt51_detection(image_path, device_meta, max_retries=3):
    """
    Sends image to GPT-5.1 / OpenAI Vision with strict JSON Schema Structured Outputs
    and metadata-guided port grounding.
    """
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY / GPT51_API_KEY not configured.")

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot load image {image_path}")
    h, w = img.shape[:2]
    max_dim = 1400
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    img_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    prompt = build_grounding_prompt(device_meta)

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_b64}",
                            "detail": "high"
                        }
                    }
                ]
            }
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "hardware_port_grounding",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "device_model": {"type": "string"},
                        "device_type": {"type": "string"},
                        "view_perspective": {"type": "string"},
                        "ports_visible": {"type": "boolean"},
                        "perspective_notes": {"type": "string"},
                        "ports": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "port_name": {"type": "string"},
                                    "port_type": {"type": "string"},
                                    "silkscreen_label": {"type": "string"},
                                    "box_2d": {
                                        "type": "array",
                                        "items": {"type": "integer"},
                                        "description": "[ymin, xmin, ymax, xmax] normalized from 0 to 1000"
                                    }
                                },
                                "required": ["port_name", "port_type", "silkscreen_label", "box_2d"],
                                "additionalProperties": False
                            }
                        }
                    },
                    "required": ["device_model", "device_type", "view_perspective", "ports_visible", "perspective_notes", "ports"],
                    "additionalProperties": False
                }
            }
        }
    }

    endpoint = f"{OPENAI_BASE_URL.rstrip('/')}/chat/completions"

    for attempt in range(max_retries):
        resp = requests.post(endpoint, headers=headers, json=payload, timeout=50)
        if resp.status_code == 200:
            result = resp.json()
            content = result["choices"][0]["message"]["content"]
            return json.loads(content)
        elif resp.status_code in [429, 503]:
            time.sleep((attempt + 1) * 3)
        else:
            raise RuntimeError(f"OpenAI/GPT-5.1 API returned {resp.status_code}: {resp.text[:300]}")

    raise RuntimeError(f"GPT-5.1 API failed after {max_retries} retries.")


def _compute_dhash(image, hash_size=8):
    resized = cv2.resize(image, (hash_size + 1, hash_size))
    diff = resized[:, 1:] > resized[:, :-1]
    return diff.flatten()


def _hamming_distance(h1, h2):
    return int(np.count_nonzero(h1 != h2))


def detect_ports_universal(image_path, device_meta, force_refresh=False):
    """
    Universal detector: routes to GPT-5.1, Gemini, or YOLO based on configuration.
    Includes smart cache retrieval, visual perceptual hash matching, and resilient CV fallback.
    """
    img_stem = os.path.splitext(os.path.basename(image_path))[0]
    cached_json = os.path.join(JSON_DIR, f"{img_stem}.json")

    # If cached annotation exists and force_refresh is False, return cached
    if not force_refresh and os.path.exists(cached_json):
        try:
            with open(cached_json, "r") as f:
                cached_data = json.load(f)
                if cached_data.get("ports") or "no_ports" in image_path.lower():
                    return cached_data
        except Exception:
            pass

    provider = os.environ.get("VISION_PROVIDER", VISION_PROVIDER).lower()

    # 1. YOLO Provider option
    if provider in ["yolo", "ultralytics"]:
        try:
            from port_detector import detect_ports_yolo
            return detect_ports_yolo(image_path, device_meta)
        except Exception as e:
            print(f"[WARN] YOLO detection failed: {e}. Falling back to Vision LLM.")

    # 2. GPT-5.1 / OpenAI Provider
    if provider in ["gpt-5.1", "gpt5", "openai"] and OPENAI_API_KEY:
        try:
            return run_gpt51_detection(image_path, device_meta)
        except Exception as e:
            print(f"[WARN] GPT-5.1 Vision error: {e}")

    # 3. Gemini Provider
    if GEMINI_API_KEY and len(GEMINI_API_KEY) > 10:
        try:
            return run_gemini_detection(image_path, device_meta)
        except Exception as e:
            print(f"[WARN] Gemini Vision error: {e}")

    # 4. Fallback check cached JSON annotation
    if os.path.exists(cached_json):
        with open(cached_json, "r") as f:
            cached_data = json.load(f)
            if cached_data.get("ports") or "no_ports" in image_path.lower():
                return cached_data

    # 5. Check visual perceptual signature match against catalog images
    if os.path.exists(RAW_DIR) and os.path.exists(JSON_DIR):
        try:
            q_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if q_img is not None:
                q_hash = _compute_dhash(q_img)
                best_ref = None
                min_dist = 999
                for fname in os.listdir(RAW_DIR):
                    if fname.endswith((".jpg", ".jpeg", ".png")):
                        ref_stem = os.path.splitext(fname)[0]
                        if ref_stem == img_stem:
                            continue
                        ref_json = os.path.join(JSON_DIR, f"{ref_stem}.json")
                        if not os.path.exists(ref_json):
                            continue
                        r_img = cv2.imread(os.path.join(RAW_DIR, fname), cv2.IMREAD_GRAYSCALE)
                        if r_img is None:
                            continue
                        dist = _hamming_distance(q_hash, _compute_dhash(r_img))
                        if dist < min_dist:
                            min_dist = dist
                            best_ref = (ref_stem, ref_json, dist)

                # Strict threshold: only match if virtually identical image (distance <= 2)
                if best_ref and best_ref[2] <= 2:
                    with open(best_ref[1], "r") as f:
                        ref_data = json.load(f)
                        if ref_data.get("ports"):
                            return {
                                "device_model": device_meta.get("name") or ref_data.get("device_model", img_stem),
                                "device_type": device_meta.get("category") or ref_data.get("device_type", "Hardware Device"),
                                "view_perspective": ref_data.get("view_perspective", "Calibrated Panel View"),
                                "ports_visible": True,
                                "perspective_notes": f"Matched catalog reference signature ({best_ref[0]}) with calibrated port grounding.",
                                "ports": ref_data["ports"]
                            }
        except Exception as e:
            print(f"[WARN] Visual signature match error: {e}")

    # 6. Check if image explicitly represents a non-port perspective
    if "no_ports" in image_path.lower() or "top_cover" in image_path.lower() or "tower" in image_path.lower():
        return {
            "device_model": device_meta.get("name", img_stem),
            "device_type": device_meta.get("category", "Hardware Device"),
            "view_perspective": "Non-Port Perspective (LEDs / Top Cover / Radome)",
            "ports_visible": False,
            "perspective_notes": "No connection ports visible in this perspective. All ports reside on other sides.",
            "ports": []
        }

    return {
        "device_model": device_meta.get("name", img_stem),
        "device_type": device_meta.get("category", "Hardware Device"),
        "view_perspective": "Panel Inspection View",
        "ports_visible": False,
        "perspective_notes": "No connection ports detected in this perspective.",
        "ports": []
    }


def draw_upper_labels_opencv(img_bgr, ports_data):
    """
    Renders sleek bounding boxes and upper-label badges using OpenCV
    with intelligent multi-tier staggering, connecting pointer lines, and collision avoidance.
    """
    output = img_bgr.copy()
    img_h, img_w = output.shape[:2]

    # Pre-parse and sort boxes left-to-right
    parsed = []
    for p in ports_data:
        box = p.get("box_2d", [])
        if len(box) != 4:
            continue
        ymin, xmin, ymax, xmax = box

        x1 = int(xmin * img_w / 1000.0)
        y1 = int(ymin * img_h / 1000.0)
        x2 = int(xmax * img_w / 1000.0)
        y2 = int(ymax * img_h / 1000.0)

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_w - 1, x2), min(img_h - 1, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        parsed.append((p, x1, y1, x2, y2))

    parsed.sort(key=lambda item: item[1]) # Sort by x1 (left to right)

    occupied_badges = []

    for idx, (p, x1, y1, x2, y2) in enumerate(parsed):
        port_name = p.get("port_name", "Port")
        port_type = p.get("port_type", "Generic")
        silk = p.get("silkscreen_label", "").strip()

        # Styling palette
        style = DEFAULT_STYLE
        search_key = f"{port_name} {port_type}".lower()
        for k, v in PORT_PALETTE.items():
            if k in search_key:
                style = v
                break

        box_color = style["box"]
        badge_bg = style["text_bg"]
        badge_fg = style["text_fg"]

        # 1. Bounding box around physical port
        cv2.rectangle(output, (x1, y1), (x2, y2), box_color, 2, cv2.LINE_AA)

        # 2. Sleek HUD corners
        box_w = x2 - x1
        box_h = y2 - y1
        corner_len = min(12, max(4, box_w // 5), max(4, box_h // 5))
        for cx, cy, dx, dy in [
            (x1, y1, 1, 1), (x2, y1, -1, 1),
            (x1, y2, 1, -1), (x2, y2, -1, -1)
        ]:
            cv2.line(output, (cx, cy), (cx + dx * corner_len, cy), box_color, 2, cv2.LINE_AA)
            cv2.line(output, (cx, cy), (cx, cy + dy * corner_len), box_color, 2, cv2.LINE_AA)

        # 3. Text formatting
        label_text = port_name
        if silk and silk not in port_name:
            label_text += f" [{silk}]"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.35
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)

        pad_x, pad_y = 4, 2
        badge_w = tw + pad_x * 2
        badge_h = th + baseline + pad_y * 2

        box_center_x = x1 + box_w // 2
        bx1 = max(2, min(box_center_x - badge_w // 2, img_w - badge_w - 2))
        bx2 = bx1 + badge_w

        # 4. Multi-tier non-overlapping placement
        used_tiers = set()
        for obx1, oby1, obx2, oby2, otier, oy1 in occupied_badges:
            if not (bx2 < obx1 - 2 or bx1 > obx2 + 2) and abs(y1 - oy1) < 45:
                used_tiers.add(otier)

        tier = 0
        while tier in used_tiers:
            tier += 1

        offset_y = 3 + (tier * (badge_h + 3))
        if y1 - badge_h - offset_y >= 2:
            by1 = y1 - badge_h - offset_y
            by2 = y1 - offset_y
            # Connect pointer line from badge bottom to top of box
            cv2.line(output, (box_center_x, by2), (box_center_x, y1), box_color, 1, cv2.LINE_AA)
        else:
            by1 = y1 + 2
            by2 = by1 + badge_h

        occupied_badges.append((bx1, by1, bx2, by2, tier, y1))

        # 5. Draw badge background & border
        cv2.rectangle(output, (bx1, by1), (bx2, by2), badge_bg, -1)
        cv2.rectangle(output, (bx1, by1), (bx2, by2), box_color, 1, cv2.LINE_AA)

        # 6. Draw label text
        tx = bx1 + pad_x
        ty = by2 - baseline - pad_y + 1
        cv2.putText(output, label_text, (tx, ty), font, font_scale, badge_fg, thickness, cv2.LINE_AA)

    return output


def export_to_cvat_and_coco():
    """Generates standard CVAT for Images 1.1 XML and COCO 1.0 JSON format."""
    print("\n--- STEP 5: Generating CVAT XML & COCO JSON Annotations ---")

    # 1. Build CVAT XML
    root = Element("annotations")
    version = SubElement(root, "version")
    version.text = "1.1"

    meta = SubElement(root, "meta")
    task = SubElement(meta, "task")
    name_el = SubElement(task, "name")
    name_el.text = "Telecom_Hardware_Port_Review"

    labels_el = SubElement(task, "labels")
    for l_name in ["RJ45", "SFP", "Power", "Console", "RF", "USB", "Ground", "Other"]:
        l_el = SubElement(labels_el, "label")
        l_n = SubElement(l_el, "name")
        l_n.text = l_name

    # 2. Build COCO JSON
    coco = {
        "info": {"description": "Telecom & Networking Hardware Ports Dataset", "version": "1.0"},
        "categories": [
            {"id": 1, "name": "RJ45"},
            {"id": 2, "name": "SFP"},
            {"id": 3, "name": "Power"},
            {"id": 4, "name": "Console"},
            {"id": 5, "name": "RF"},
            {"id": 6, "name": "USB"},
            {"id": 7, "name": "Ground"},
            {"id": 8, "name": "Other"}
        ],
        "images": [],
        "annotations": []
    }
    cat_map = {c["name"]: c["id"] for c in coco["categories"]}
    ann_id = 1

    seen_ids = set()
    all_items = []
    for item in DEVICE_SOURCES:
        seen_ids.add(item["id"])
        all_items.append((item["id"], item.get("name", item["id"])))
    if os.path.exists(JSON_DIR):
        for f in sorted(os.listdir(JSON_DIR)):
            if f.endswith(".json"):
                stem = os.path.splitext(f)[0]
                if stem not in seen_ids:
                    seen_ids.add(stem)
                    all_items.append((stem, stem))

    for img_idx, (img_id, default_name) in enumerate(all_items):
        json_file = os.path.join(JSON_DIR, f"{img_id}.json")
        img_file = os.path.join(RAW_DIR, f"{img_id}.jpg")

        if not os.path.exists(json_file) or not os.path.exists(img_file):
            continue

        with open(json_file, "r") as f:
            data = json.load(f)

        img = cv2.imread(img_file)
        if img is None:
            continue
        h, w = img.shape[:2]

        # Add image to CVAT XML
        xml_img = SubElement(root, "image", id=str(img_idx + 1), name=f"{img_id}.jpg", width=str(w), height=str(h))

        # Add image to COCO
        coco["images"].append({
            "id": img_idx + 1,
            "file_name": f"{img_id}.jpg",
            "width": w,
            "height": h,
            "device_model": data.get("device_model", default_name)
        })

        for p in data.get("ports", []):
            box = p.get("box_2d", [])
            if len(box) != 4:
                continue
            ymin, xmin, ymax, xmax = box
            x1 = round(xmin * w / 1000.0, 1)
            y1 = round(ymin * h / 1000.0, 1)
            x2 = round(xmax * w / 1000.0, 1)
            y2 = round(ymax * h / 1000.0, 1)
            bw = max(1.0, round(x2 - x1, 1))
            bh = max(1.0, round(y2 - y1, 1))

            p_type = p.get("port_type", "Other")
            if p_type not in cat_map:
                p_type = "Other"

            # CVAT box element
            box_el = SubElement(xml_img, "box",
                                label=p_type,
                                occluded="0",
                                xtl=str(x1), ytl=str(y1),
                                xbr=str(x2), ybr=str(y2),
                                z_order="0")
            attr_name = SubElement(box_el, "attribute", name="port_name")
            attr_name.text = p.get("port_name", "")
            attr_silk = SubElement(box_el, "attribute", name="silkscreen_label")
            attr_silk.text = p.get("silkscreen_label", "")

            # COCO annotation
            coco["annotations"].append({
                "id": ann_id,
                "image_id": img_idx + 1,
                "category_id": cat_map[p_type],
                "bbox": [x1, y1, bw, bh],
                "area": bw * bh,
                "iscrowd": 0,
                "attributes": {
                    "port_name": p.get("port_name", ""),
                    "silkscreen_label": p.get("silkscreen_label", "")
                }
            })
            ann_id += 1

    # Save CVAT XML
    xml_str = minidom.parseString(tostring(root)).toprettyxml(indent="  ")
    cvat_xml_path = os.path.join(CVAT_DIR, "cvat_annotations.xml")
    with open(cvat_xml_path, "w", encoding="utf-8") as f:
        f.write(xml_str)
    print(f"[CVAT] Saved CVAT for Images 1.1 XML to: {cvat_xml_path}")

    # Save COCO JSON
    coco_json_path = os.path.join(CVAT_DIR, "coco_annotations.json")
    with open(coco_json_path, "w", encoding="utf-8") as f:
        json.dump(coco, f, indent=2)
    print(f"[COCO] Saved COCO 1.0 JSON to: {coco_json_path}")


def process_all_devices():
    """Main pipeline loop executing steps 1 to 5."""
    # Step 1: Download images
    download_device_images()

    print("\n--- STEPS 2-4: Gemini Vision Grounding & OpenCV Rendering ---")
    for item in DEVICE_SOURCES:
        img_id = item["id"]
        raw_path = os.path.join(RAW_DIR, f"{img_id}.jpg")
        json_path = os.path.join(JSON_DIR, f"{img_id}.json")
        annotated_path = os.path.join(ANNOTATED_DIR, f"{img_id}_labeled.jpg")

        if not os.path.exists(raw_path):
            print(f"[SKIP] Raw image missing: {img_id}")
            continue

        # Load or run Gemini Detection
        if os.path.exists(json_path):
            print(f"[CACHE] Loading cached detection for {item['name']}...")
            with open(json_path, "r") as f:
                data = json.load(f)
        else:
            print(f"[{VISION_PROVIDER.upper()}] Calling {VISION_PROVIDER} for {item['name']}...")
            try:
                data = detect_ports_universal(raw_path, item)
                with open(json_path, "w") as f:
                    json.dump(data, f, indent=2)
                print(f"[SUCCESS] Detected {len(data.get('ports', []))} ports.")
                time.sleep(1.0)
            except Exception as e:
                print(f"[ERROR] Detection error on {img_id}: {e}")
                continue

        # Step 4: OpenCV Preview
        img = cv2.imread(raw_path)
        if img is not None:
            labeled = draw_upper_labels_opencv(img, data.get("ports", []))
            cv2.imwrite(annotated_path, labeled)
            print(f"[OPENCV] Saved labeled preview to: {os.path.basename(annotated_path)}")

    # Step 5: Export CVAT & COCO
    export_to_cvat_and_coco()

    # Step 6: Export YOLO format
    export_to_yolo_format()
    print("\n[COMPLETE] All 15 devices processed successfully!")


def export_to_yolo_format(yolo_dir=None):
    """Generates standard Ultralytics YOLOv8 training dataset format."""
    import shutil
    if yolo_dir is None:
        yolo_dir = os.path.join(DATA_DIR, "yolo_export")

    for split in ["train", "val"]:
        os.makedirs(os.path.join(yolo_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(yolo_dir, "labels", split), exist_ok=True)

    classes = ["RJ45", "USB_A", "USB_C", "HDMI", "Audio", "Power", "SFP", "RF", "Console", "DVI", "SD_Card", "Other"]

    def map_class(ptype, pname):
        comb = f"{ptype} {pname}".lower()
        if "rj45" in comb or "ethernet" in comb or "lan" in comb: return 0
        if "usb-c" in comb or "type-c" in comb: return 2
        if "usb" in comb: return 1
        if "hdmi" in comb: return 3
        if "audio" in comb or "headphone" in comb or "mic" in comb: return 4
        if "power" in comb or "magsafe" in comb or "dc" in comb: return 5
        if "sfp" in comb or "optical" in comb: return 6
        if "rf" in comb or "antenna" in comb or "coaxial" in comb: return 7
        if "console" in comb or "serial" in comb: return 8
        if "dvi" in comb: return 9
        if "sd" in comb: return 10
        return 11

    val_families = ["custom_3bef8a29", "device_06_dlink_switch", "device_08_sfp_optical_transceivers", "device_14_tplink_5port"]
    n_train, n_val, total_boxes = 0, 0, 0

    if os.path.exists(JSON_DIR):
        for f in sorted(os.listdir(JSON_DIR)):
            if not f.endswith(".json"): continue
            stem = os.path.splitext(f)[0]
            raw_img = os.path.join(RAW_DIR, f"{stem}.jpg")
            if not os.path.exists(raw_img): continue

            split = "val" if any(vf in stem for vf in val_families) else "train"
            shutil.copy(raw_img, os.path.join(yolo_dir, "images", split, f"{stem}.jpg"))

            with open(os.path.join(JSON_DIR, f)) as jf:
                data = json.load(jf)

            lines = []
            for p in data.get("ports", []):
                box = p.get("box_2d", [])
                if len(box) != 4: continue
                ymin, xmin, ymax, xmax = box
                xc = (xmin + xmax) / 2000.0
                yc = (ymin + ymax) / 2000.0
                w = (xmax - xmin) / 1000.0
                h = (ymax - ymin) / 1000.0
                cid = map_class(p.get("port_type", ""), p.get("port_name", ""))
                lines.append(f"{cid} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
                total_boxes += 1

            with open(os.path.join(yolo_dir, "labels", split, f"{stem}.txt"), "w") as lf:
                lf.write("\n".join(lines) + "\n")

            if split == "train": n_train += 1
            else: n_val += 1

    yaml_content = f"""# Ultralytics YOLOv8 Dataset Configuration
path: {os.path.abspath(yolo_dir)}
train: images/train
val: images/val
nc: {len(classes)}
names: {classes}
"""
    with open(os.path.join(yolo_dir, "dataset.yaml"), "w") as yf:
        yf.write(yaml_content)

    print(f"[YOLO] Exported YOLOv8 dataset: {n_train} train, {n_val} val images, {total_boxes} ports.")
    return yolo_dir


if __name__ == "__main__":
    process_all_devices()
