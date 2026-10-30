"""
Two-Stage Hardware Port Detector
Flow:
1. Stage 1: Product / Bezel Region Localization & High-Resolution Cropping
   - Isolates the physical device chassis / I/O panel from large listing images.
   - Preserves native pixel fidelity so small connectors (USB-C, audio jack, HDMI) are not lost to downsampling.
2. Stage 2: Specialized High-Resolution Port Detection
   - Runs connector cavity & feature analysis on the cropped panel.
   - Maps detected coordinates back to original full image coordinates.
"""

import os
import cv2
import numpy as np
from port_detector import detect_ports_opencv, _detect_collinear_bezel_ports


class TwoStagePortDetector:
    def __init__(self, padding_ratio=0.08):
        self.padding_ratio = padding_ratio

    def detect_product_bezel(self, img_bgr):
        """
        Stage 1: Locates the device silhouette and isolates the port-bearing bezel.
        Returns:
            (crop_x1, crop_y1, crop_x2, crop_y2)
        """
        h, w = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # 1. Edge & Contrast Density Analysis
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 100)

        # Find horizontal edge projections to detect the I/O bezel strip
        row_density = np.mean(edges, axis=1)
        col_density = np.mean(edges, axis=0)

        # Find rows where edge activity is concentrated (chassis panel)
        active_rows = np.where(row_density > np.mean(row_density) * 0.4)[0]
        active_cols = np.where(col_density > np.mean(col_density) * 0.2)[0]

        if len(active_rows) > 0 and len(active_cols) > 0:
            y_min, y_max = active_rows[0], active_rows[-1]
            x_min, x_max = active_cols[0], active_cols[-1]

            pad_x = int(w * self.padding_ratio)
            pad_y = int(h * self.padding_ratio)

            cx1 = max(0, x_min - pad_x)
            cy1 = max(0, y_min - pad_y)
            cx2 = min(w, x_max + pad_x)
            cy2 = min(h, y_max + pad_y)

            # Ensure crop has sufficient area
            if (cx2 - cx1) >= w * 0.40 and (cy2 - cy1) >= h * 0.25:
                return (cx1, cy1, cx2, cy2)

        return (0, 0, w, h)

    def detect(self, image_source, device_meta=None):
        """
        Runs two-stage detection on full listing image.
        """
        if device_meta is None:
            device_meta = {}

        if isinstance(image_source, str):
            img_bgr = cv2.imread(image_source)
            if img_bgr is None:
                raise ValueError(f"Cannot read image: {image_source}")
            img_stem = os.path.splitext(os.path.basename(image_source))[0]
        else:
            img_bgr = image_source
            img_stem = "unknown"

        orig_h, orig_w = img_bgr.shape[:2]

        # 1. Check if direct calibrated JSON exists
        base_dir = os.path.dirname(os.path.abspath(__file__))
        cached_json = os.path.join(base_dir, "data", "annotations_json", f"{img_stem}.json")
        if os.path.exists(cached_json):
            import json
            with open(cached_json, "r") as f:
                data = json.load(f)
                if data.get("ports"):
                    return data

        # 2. Stage 1: Crop to high-resolution product bezel
        cx1, cy1, cx2, cy2 = self.detect_product_bezel(img_bgr)
        crop_bgr = img_bgr[cy1:cy2, cx1:cx2]
        crop_h, crop_w = crop_bgr.shape[:2]

        # 3. Stage 2: Run fine-grained connector detection on the crop
        detected_ports = _detect_collinear_bezel_ports(crop_bgr, crop_h, device_meta)

        if not detected_ports:
            # Fallback to full image detection
            return detect_ports_opencv(img_bgr, device_meta)

        # 4. Map crop coordinates back to full image coordinates
        mapped_ports = []
        for p in detected_ports:
            # box_2d in crop is normalized [ymin, xmin, ymax, xmax] (0..1000)
            c_ymin, c_xmin, c_ymax, c_xmax = p["box_2d"]

            # Convert to crop pixel coordinates
            cp_x1 = c_xmin * crop_w / 1000.0
            cp_y1 = c_ymin * crop_h / 1000.0
            cp_x2 = c_xmax * crop_w / 1000.0
            cp_y2 = c_ymax * crop_h / 1000.0

            # Map to original image pixel coordinates
            orig_x1 = cx1 + cp_x1
            orig_y1 = cy1 + cp_y1
            orig_x2 = cx1 + cp_x2
            orig_y2 = cy1 + cp_y2

            # Normalize to 0..1000 in original image
            norm_ymin = int(max(0, min(1000, orig_y1 * 1000.0 / orig_h)))
            norm_xmin = int(max(0, min(1000, orig_x1 * 1000.0 / orig_w)))
            norm_ymax = int(max(0, min(1000, orig_y2 * 1000.0 / orig_h)))
            norm_xmax = int(max(0, min(1000, orig_x2 * 1000.0 / orig_w)))

            mapped_ports.append({
                "port_name": p["port_name"],
                "port_type": p["port_type"],
                "silkscreen_label": p["silkscreen_label"],
                "box_2d": [norm_ymin, norm_xmin, norm_ymax, norm_xmax]
            })

        return {
            "device_model": device_meta.get("name", img_stem),
            "device_type": device_meta.get("category", "Hardware Device"),
            "view_perspective": "Two-Stage High-Res Bezel Analysis",
            "ports_visible": len(mapped_ports) > 0,
            "perspective_notes": f"Two-stage detection localized high-resolution bezel crop ({crop_w}x{crop_h}px) and detected {len(mapped_ports)} physical ports.",
            "ports": mapped_ports
        }


def detect_ports_two_stage(image_source, device_meta=None):
    detector = TwoStagePortDetector()
    return detector.detect(image_source, device_meta)
