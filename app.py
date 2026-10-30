"""
Flask Web Application for Hardware Port Detection & Human Review
Features:
1. Image & Photo Detection (Gemini Vision + OpenCV)
2. PDF Datasheet Data Extraction (pypdf + pdftoppm + OpenCV Upper-Labeling)
3. CVAT for Images 1.1 XML & COCO 1.0 JSON Export
4. MongoDB-Style Product Metadata & Non-Port Image Lineage Auditor
"""

import os
import sys
import json
import base64
import uuid
import cv2
from flask import Flask, render_template, request, jsonify, send_from_directory, send_file
from dotenv import load_dotenv
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))
DATA_DIR = os.path.join(BASE_DIR, "data")

os.environ.setdefault("GEMINI_MODEL", "gemini-3.5-flash-lite")
RAW_DIR = os.path.join(DATA_DIR, "raw_images")
JSON_DIR = os.path.join(DATA_DIR, "annotations_json")
ANNOTATED_DIR = os.path.join(DATA_DIR, "annotated_images")
MARKETPLACE_DIR = os.path.join(DATA_DIR, "marketplace_images")
CVAT_DIR = os.path.join(DATA_DIR, "cvat_export")
UPLOADS_DIR = os.path.join(DATA_DIR, "uploads")
PDF_DIR = os.path.join(DATA_DIR, "pdf_uploads")
SAMPLE_DIR = os.path.join(BASE_DIR, "sample_datasheets")
TEST_IMG_DIR = os.path.join(DATA_DIR, "metadata_test_images")

for d in [RAW_DIR, JSON_DIR, ANNOTATED_DIR, MARKETPLACE_DIR, CVAT_DIR, UPLOADS_DIR, PDF_DIR, SAMPLE_DIR, TEST_IMG_DIR]:
    os.makedirs(d, exist_ok=True)

app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"), static_folder=os.path.join(BASE_DIR, "static"))

from pipeline import detect_ports_universal, draw_upper_labels_opencv, export_to_cvat_and_coco, export_to_yolo_format, DEVICE_SOURCES
from marketplace_layout import render_marketplace_diagram
from pdf_extractor import process_pdf_datasheet
from metadata_manager import ProductMetadataManager

metadata_mgr = ProductMetadataManager()


@app.route("/")
def index():
    return render_template("index.html")


# --- METADATA & NON-PORT IMAGE AUDIT ENDPOINTS ---

@app.route("/api/metadata/products")
def get_metadata_products():
    """Returns all products in the MongoDB catalog with associated image relations."""
    products = metadata_mgr.get_all_products()
    return jsonify(products)


@app.route("/api/metadata/audit/<prod_id>/<image_id>")
def get_metadata_audit(prod_id, image_id):
    """Returns deep reconciliation between an image and its product metadata."""
    audit = metadata_mgr.audit_image_relation(prod_id, image_id)
    return jsonify(audit)


@app.route("/images/test/<filename>")
def serve_test_image(filename):
    """Serves non-port, blurry, or inverted hardware images."""
    path = os.path.join(TEST_IMG_DIR, filename)
    if not os.path.exists(path):
        return "Test image not found", 404
    return send_file(path, mimetype="image/jpeg")


# --- STANDARD DEVICE & CVAT ENDPOINTS ---

@app.route("/api/devices")
def get_devices():
    """Returns list of all devices with metadata, specs, and detection status."""
    devices = []
    seen_ids = set()
    all_json_files = sorted([f for f in os.listdir(JSON_DIR) if f.endswith(".json")])

    for jf in all_json_files:
        dev_id = os.path.splitext(jf)[0]
        seen_ids.add(dev_id)
        json_file = os.path.join(JSON_DIR, jf)
        raw_file = os.path.join(RAW_DIR, f"{dev_id}.jpg")
        annotated_file = os.path.join(ANNOTATED_DIR, f"{dev_id}_labeled.jpg")
        marketplace_file = os.path.join(MARKETPLACE_DIR, f"{dev_id}_marketplace.png")
        if not os.path.exists(marketplace_file):
            marketplace_file = os.path.join(ANNOTATED_DIR, f"{dev_id}_marketplace.png")

        data = {}
        if os.path.exists(json_file):
            with open(json_file, "r") as f:
                data = json.load(f)

        ports = data.get("ports", [])
        port_types = list(set([p.get("port_type", "Other") for p in ports]))

        matched_source = next((item for item in DEVICE_SOURCES if item["id"] == dev_id), None)
        category = matched_source["type"] if matched_source else ("PDF Datasheet" if "specs" in data else "Hardware")

        devices.append({
            "id": dev_id,
            "name": data.get("device_model", matched_source["name"] if matched_source else dev_id),
            "category": category,
            "port_count": len(ports),
            "port_types": port_types,
            "raw_available": os.path.exists(raw_file),
            "annotated_available": os.path.exists(annotated_file),
            "marketplace_available": os.path.exists(marketplace_file),
            "marketplace_image": f"/images/marketplace/{dev_id}_marketplace.png" if os.path.exists(marketplace_file) else None,
            "specs": data.get("specs", {}),
            "ports": ports
        })

    return jsonify(devices)


@app.route("/api/device/<dev_id>")
def get_device_detail(dev_id):
    """Returns detailed port information and extracted specs for a specific device."""
    json_file = os.path.join(JSON_DIR, f"{dev_id}.json")
    if not os.path.exists(json_file):
        return jsonify({"error": "Device annotations not found"}), 404

    with open(json_file, "r") as f:
        data = json.load(f)

    return jsonify({
        "id": dev_id,
        "device_model": data.get("device_model", dev_id),
        "device_type": data.get("device_type", "Hardware"),
        "specs": data.get("specs", {}),
        "ports": data.get("ports", [])
    })


@app.route("/api/device/<dev_id>/update", methods=["POST"])
def update_device_annotations(dev_id):
    """Allows human review and editing of port annotations."""
    json_file = os.path.join(JSON_DIR, f"{dev_id}.json")
    raw_file = os.path.join(RAW_DIR, f"{dev_id}.jpg")
    annotated_file = os.path.join(ANNOTATED_DIR, f"{dev_id}_labeled.jpg")
    marketplace_file1 = os.path.join(ANNOTATED_DIR, f"{dev_id}_marketplace.png")
    marketplace_file2 = os.path.join(MARKETPLACE_DIR, f"{dev_id}_marketplace.png")

    req_data = request.json
    ports = req_data.get("ports", [])

    existing_data = {}
    if os.path.exists(json_file):
        with open(json_file, "r") as f:
            existing_data = json.load(f)

    existing_data["device_model"] = req_data.get("device_model", dev_id)
    existing_data["ports"] = ports
    existing_data["ports_visible"] = len(ports) > 0

    with open(json_file, "w") as f:
        json.dump(existing_data, f, indent=2)

    # Re-render OpenCV preview and Marketplace diagram
    if os.path.exists(raw_file):
        img = cv2.imread(raw_file)
        if img is not None:
            labeled = draw_upper_labels_opencv(img, ports)
            cv2.imwrite(annotated_file, labeled)
            try:
                render_marketplace_diagram(raw_file, ports, output_path=marketplace_file1)
                render_marketplace_diagram(raw_file, ports, output_path=marketplace_file2)
            except Exception as e:
                print(f"[WARN] Marketplace re-render error: {e}")

    export_to_cvat_and_coco()
    export_to_yolo_format()
    return jsonify({"status": "success", "message": "Annotations updated, OpenCV & Marketplace diagrams re-rendered, datasets synced."})


@app.route("/api/device/<dev_id>/reanalyze", methods=["POST"])
def reanalyze_device(dev_id):
    """Forces AI re-detection on an existing image with updated domain heuristics."""
    raw_file = os.path.join(RAW_DIR, f"{dev_id}.jpg")
    if not os.path.exists(raw_file):
        return jsonify({"error": "Device raw image not found"}), 404

    json_file = os.path.join(JSON_DIR, f"{dev_id}.json")
    existing_data = {}
    if os.path.exists(json_file):
        with open(json_file, "r") as f:
            existing_data = json.load(f)

    req_data = request.json or {}
    custom_desc = req_data.get("metadata_context", "")

    device_meta = {
        "name": req_data.get("device_model", existing_data.get("device_model", dev_id)),
        "category": req_data.get("category", existing_data.get("device_type", "Hardware Device")),
        "datasheet_description": custom_desc or existing_data.get("perspective_notes", "")
    }

    try:
        new_data = detect_ports_universal(raw_file, device_meta, force_refresh=True)
        new_ports = new_data.get("ports", [])

        existing_data.update(new_data)
        with open(json_file, "w") as f:
            json.dump(existing_data, f, indent=2)

        # Re-render OpenCV and Marketplace images
        img = cv2.imread(raw_file)
        if img is not None:
            labeled = draw_upper_labels_opencv(img, new_ports)
            cv2.imwrite(os.path.join(ANNOTATED_DIR, f"{dev_id}_labeled.jpg"), labeled)
            try:
                render_marketplace_diagram(raw_file, new_ports, output_path=os.path.join(ANNOTATED_DIR, f"{dev_id}_marketplace.png"))
                render_marketplace_diagram(raw_file, new_ports, output_path=os.path.join(MARKETPLACE_DIR, f"{dev_id}_marketplace.png"))
            except Exception as e:
                print(f"[WARN] Marketplace render error: {e}")

        export_to_cvat_and_coco()
        export_to_yolo_format()

        return jsonify({
            "status": "success",
            "message": f"Successfully reanalyzed {dev_id}. Detected {len(new_ports)} ports.",
            "device_model": existing_data.get("device_model", dev_id),
            "ports": new_ports,
            "port_count": len(new_ports)
        })
    except Exception as e:
        return jsonify({"error": f"Reanalysis failed: {str(e)}"}), 500


@app.route("/api/upload_pdf", methods=["POST"])
def upload_pdf():
    """Extracts specs, renders diagram, and labels ports from an uploaded PDF datasheet."""
    if "file" not in request.files:
        return jsonify({"error": "No PDF file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    save_name = f"pdf_{uuid.uuid4().hex[:8]}_{file.filename}"
    pdf_path = os.path.join(PDF_DIR, save_name)
    file.save(pdf_path)

    try:
        result = process_pdf_datasheet(pdf_path)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"PDF Processing failed: {str(e)}"}), 500


@app.route("/api/sample_pdf")
def process_sample_pdf():
    """Quick demo trigger for the CommScope 10-port antenna PDF datasheet."""
    sample_path = os.path.join(SAMPLE_DIR, "Commscope_Antenna_10Port_EGRZV-65D-R5N43.pdf")
    if not os.path.exists(sample_path):
        return jsonify({"error": "Sample PDF not found"}), 404

    try:
        result = process_pdf_datasheet(sample_path)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Sample PDF processing failed: {str(e)}"}), 500


@app.route("/api/upload", methods=["POST"])
def upload_and_detect():
    """Allows uploading any hardware device photo (2D, 3D, or diagram) with optional metadata grounding."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    upload_id = f"custom_{uuid.uuid4().hex[:8]}"
    raw_path = os.path.join(RAW_DIR, f"{upload_id}.jpg")
    file.save(raw_path)

    # Check for metadata passed via form
    product_id = request.form.get("product_id")
    custom_desc = request.form.get("metadata_context", "")

    device_meta = {"name": file.filename, "category": "Hardware Device"}
    if product_id:
        prod = metadata_mgr.get_product(product_id)
        if prod:
            device_meta["name"] = prod.get("product_name", file.filename)
            device_meta["category"] = prod.get("category", "Hardware Device")
            device_meta["datasheet_description"] = prod.get("datasheet_description", "")
            device_meta["port_specifications"] = prod.get("port_specifications", [])
    elif custom_desc:
        device_meta["datasheet_description"] = custom_desc

    try:
        data = detect_ports_universal(raw_path, device_meta)
        json_path = os.path.join(JSON_DIR, f"{upload_id}.json")
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)

        ports = data.get("ports", [])
        img = cv2.imread(raw_path)
        if img is not None:
            labeled = draw_upper_labels_opencv(img, ports)
            annotated_path = os.path.join(ANNOTATED_DIR, f"{upload_id}_labeled.jpg")
            cv2.imwrite(annotated_path, labeled)

        # Generate Marketplace Technical Diagram with Orthogonal Lines
        marketplace_path1 = os.path.join(ANNOTATED_DIR, f"{upload_id}_marketplace.png")
        marketplace_path2 = os.path.join(MARKETPLACE_DIR, f"{upload_id}_marketplace.png")
        try:
            render_marketplace_diagram(raw_path, ports, output_path=marketplace_path1)
            render_marketplace_diagram(raw_path, ports, output_path=marketplace_path2)
        except Exception as e:
            print(f"[WARN] Marketplace render error: {e}")

        export_to_cvat_and_coco()
        export_to_yolo_format()

        return jsonify({
            "id": upload_id,
            "device_model": data.get("device_model", file.filename),
            "view_perspective": data.get("view_perspective", "Standard View"),
            "ports_visible": data.get("ports_visible", len(ports) > 0),
            "perspective_notes": data.get("perspective_notes", ""),
            "port_count": len(ports),
            "ports": ports,
            "annotated_image": f"/images/annotated/{upload_id}_labeled.jpg",
            "marketplace_image": f"/images/marketplace/{upload_id}_marketplace.png",
            "raw_image": f"/images/raw/{upload_id}.jpg"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/images/raw/<path:filename>")
def serve_raw_image(filename):
    if not filename.endswith(".jpg"):
        filename = f"{filename}.jpg"
    path = os.path.join(RAW_DIR, filename)
    if not os.path.exists(path):
        path = os.path.join(UPLOADS_DIR, filename)
    if not os.path.exists(path):
        return "Image not found", 404
    return send_file(path, mimetype="image/jpeg")


@app.route("/images/annotated/<path:filename>")
def serve_annotated_image(filename):
    if not filename.endswith(".jpg"):
        filename = f"{filename}_labeled.jpg"
    path = os.path.join(ANNOTATED_DIR, filename)
    if not os.path.exists(path):
        alt = os.path.join(ANNOTATED_DIR, filename.replace("_labeled.jpg", ".jpg"))
        if os.path.exists(alt):
            path = alt
        else:
            return "Annotated image not found", 404
    return send_file(path, mimetype="image/jpeg")


@app.route("/images/marketplace/<path:filename>")
def serve_marketplace_image(filename):
    if not filename.endswith(".png"):
        filename = f"{filename}_marketplace.png"
    path = os.path.join(MARKETPLACE_DIR, filename)
    if not os.path.exists(path):
        alt = os.path.join(ANNOTATED_DIR, filename)
        if os.path.exists(alt):
            path = alt
        else:
            return "Marketplace diagram not found", 404
    return send_file(path, mimetype="image/png")


@app.route("/export/cvat")
def download_cvat():
    xml_path = os.path.join(CVAT_DIR, "cvat_annotations.xml")
    if os.path.exists(xml_path):
        return send_file(xml_path, as_attachment=True, download_name="cvat_annotations.xml")
    return "CVAT XML not generated yet", 404


@app.route("/export/coco")
def download_coco():
    json_path = os.path.join(CVAT_DIR, "coco_annotations.json")
    if os.path.exists(json_path):
        return send_file(json_path, as_attachment=True, download_name="coco_annotations.json")
    return "COCO JSON not generated yet", 404


@app.route("/export/yolo")
def download_yolo():
    import shutil
    yolo_dir = os.path.join(DATA_DIR, "yolo_export")
    zip_path = os.path.join(DATA_DIR, "yolo_dataset")
    shutil.make_archive(zip_path, 'zip', yolo_dir)
    return send_file(f"{zip_path}.zip", as_attachment=True, download_name="yolo_dataset.zip")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Hardware Port Vision & Metadata Dashboard")
    parser.add_argument("--port", "-p", type=int, default=int(os.environ.get("PORT", 5000)), help="Port to run server on (default: 5000)")
    args = parser.parse_args()

    port = args.port
    print(f"\n[SERVER] Hardware Port Vision & Metadata Dashboard running on: http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
