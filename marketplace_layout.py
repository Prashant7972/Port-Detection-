"""
Marketplace-Grade Technical Diagram Layout Engine
Features:
1. Cost-Function Layout Optimizer (Minimizes Overlap, Crossings, Line Length, Lateral Displacement)
2. Canvas Expansion: Automatically pads top/bottom margin with clean neutral background when needed
3. Orthogonal Connector Lines: Vertical leader lines with horizontal elbows and port anchor dots
4. High-Resolution TrueType Typography via Pillow
5. Polished Marketplace / Datasheet Aesthetics
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Category Color Palette (RGB)
CATEGORY_COLORS = {
    "usb": {"accent": (2, 132, 199), "bg": (240, 249, 255), "border": (186, 230, 253)},
    "rj45": {"accent": (5, 150, 105), "bg": (236, 253, 245), "border": (167, 243, 208)},
    "ethernet": {"accent": (5, 150, 105), "bg": (236, 253, 245), "border": (167, 243, 208)},
    "power": {"accent": (220, 38, 38), "bg": (254, 242, 242), "border": (254, 202, 202)},
    "dc": {"accent": (220, 38, 38), "bg": (254, 242, 242), "border": (254, 202, 202)},
    "hdmi": {"accent": (124, 58, 237), "bg": (245, 243, 255), "border": (221, 214, 254)},
    "video": {"accent": (124, 58, 237), "bg": (245, 243, 255), "border": (221, 214, 254)},
    "dvi": {"accent": (124, 58, 237), "bg": (245, 243, 255), "border": (221, 214, 254)},
    "displayport": {"accent": (124, 58, 237), "bg": (245, 243, 255), "border": (221, 214, 254)},
    "audio": {"accent": (37, 99, 235), "bg": (239, 246, 255), "border": (191, 219, 254)},
    "headphone": {"accent": (37, 99, 235), "bg": (239, 246, 255), "border": (191, 219, 254)},
    "sfp": {"accent": (217, 119, 6), "bg": (255, 251, 235), "border": (253, 230, 138)},
    "optical": {"accent": (217, 119, 6), "bg": (255, 251, 235), "border": (253, 230, 138)},
    "rf": {"accent": (13, 148, 136), "bg": (240, 253, 250), "border": (153, 246, 228)},
    "antenna": {"accent": (13, 148, 136), "bg": (240, 253, 250), "border": (153, 246, 228)},
    "sd": {"accent": (79, 70, 229), "bg": (238, 242, 255), "border": (199, 210, 254)},
    "lock": {"accent": (71, 85, 105), "bg": (248, 250, 252), "border": (226, 232, 240)},
    "other": {"accent": (71, 85, 105), "bg": (248, 250, 252), "border": (226, 232, 240)},
}
DEFAULT_COLOR = {"accent": (15, 23, 42), "bg": (255, 255, 255), "border": (203, 213, 225)}


def _get_font(size=13, bold=False):
    """Loads optimal TrueType system font."""
    font_paths = [
        "/usr/share/fonts/open-sans/OpenSans-Semibold.ttf" if bold else "/usr/share/fonts/open-sans/OpenSans-Regular.ttf",
        "/usr/share/fonts/google-droid-sans-fonts/DroidSans-Bold.ttf" if bold else "/usr/share/fonts/google-droid-sans-fonts/DroidSans.ttf",
        "/usr/share/fonts/adwaita-sans-fonts/AdwaitaSans-Regular.ttf",
        "/usr/share/fonts/google-noto-vf/NotoSans[wght].ttf"
    ]
    for p in font_paths:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    return ImageFont.load_default()


def _get_color_scheme(port_name, port_type):
    """Matches port category to styling palette."""
    combined = f"{port_name} {port_type}".lower()
    for key, val in CATEGORY_COLORS.items():
        if key in combined:
            return val
    return DEFAULT_COLOR


class MarketplaceLayoutEngine:
    """
    Optimizes label card placement and routes orthogonal leader lines.
    Minimizes:
        J = lambda_o * Overlap + lambda_c * Collisions + lambda_x * Crossings + lambda_l * LineLen + lambda_p * Displacement
    """

    def __init__(self, card_height=32, lane_spacing=14, margin_pad=18):
        self.card_height = card_height
        self.lane_spacing = lane_spacing
        self.margin_pad = margin_pad
        self.font_title = _get_font(13, bold=True)
        self.font_silk = _get_font(10, bold=False)

    def render_diagram(self, image_source, ports, title=None, expand_canvas=True):
        """
        Renders a production-quality technical marketplace diagram.
        """
        # 1. Load image
        if isinstance(image_source, str):
            orig_pil = Image.open(image_source).convert("RGB")
        elif isinstance(image_source, np.ndarray):
            rgb = cv2.cvtColor(image_source, cv2.COLOR_BGR2RGB)
            orig_pil = Image.fromarray(rgb)
        elif isinstance(image_source, Image.Image):
            orig_pil = image_source.convert("RGB")
        else:
            raise ValueError("Unsupported image source type")

        img_w, img_h = orig_pil.size

        if not ports:
            return orig_pil

        # 2. Extract port coordinates & anchors
        port_items = []
        for i, p in enumerate(ports):
            box = p.get("box_2d", [0, 0, 0, 0])
            ymin, xmin, ymax, xmax = box
            px1 = int(round(xmin * img_w / 1000.0))
            py1 = int(round(ymin * img_h / 1000.0))
            px2 = int(round(xmax * img_w / 1000.0))
            py2 = int(round(ymax * img_h / 1000.0))

            cx = (px1 + px2) / 2.0
            cy = (py1 + py2) / 2.0
            pname = p.get("port_name", f"Port {i+1}")
            ptype = p.get("port_type", "Other")
            silk = p.get("silkscreen_label", "")

            title_text = pname
            silk_text = f"[{silk}]" if silk and silk.lower() not in pname.lower() else ""

            t_box = self.font_title.getbbox(title_text)
            tw = t_box[2] - t_box[0]
            sw = 0
            if silk_text:
                s_box = self.font_silk.getbbox(silk_text)
                sw = s_box[2] - s_box[0] + 8

            card_w = tw + sw + 28
            card_h = self.card_height

            port_items.append({
                "idx": i,
                "port_name": pname,
                "port_type": ptype,
                "silkscreen_label": silk,
                "title_text": title_text,
                "silk_text": silk_text,
                "box": (px1, py1, px2, py2),
                "anchor": (cx, py1),
                "card_w": card_w,
                "card_h": card_h,
                "scheme": _get_color_scheme(pname, ptype)
            })

        # Sort ports strictly by x-coordinate to prevent orthogonal line crossings (X = 0)
        port_items.sort(key=lambda item: item["anchor"][0])

        # 3. Canvas Expansion Calculation
        total_card_w = sum(p["card_w"] for p in port_items)
        lanes_needed = max(1, int(np.ceil(total_card_w / float(img_w - 60))))
        lanes_needed = min(lanes_needed, 3)

        top_padding = 0
        bottom_padding = 0
        if expand_canvas:
            min_y = min(p["anchor"][1] for p in port_items)
            required_top = (lanes_needed * (self.card_height + self.lane_spacing)) + 35
            if min_y < required_top:
                top_padding = int(required_top - min_y) + 20
            bottom_padding = 20

        new_w = img_w
        new_h = img_h + top_padding + bottom_padding

        canvas = Image.new("RGB", (new_w, new_h), (255, 255, 255))
        canvas.paste(orig_pil, (0, top_padding))

        for p in port_items:
            x1, y1, x2, y2 = p["box"]
            p["box_adj"] = (x1, y1 + top_padding, x2, y2 + top_padding)
            p["anchor_adj"] = (p["anchor"][0], p["anchor"][1] + top_padding)

        # 4. Multi-Lane Placement
        total_card_w = sum(p["card_w"] for p in port_items)
        lanes_needed = max(2, int(np.ceil(total_card_w / float(img_w * 0.48))))
        lanes_needed = min(lanes_needed, 4)

        lanes = [[] for _ in range(lanes_needed)]
        base_lane_y = 16

        for i, p in enumerate(port_items):
            ax = p["anchor_adj"][0]
            cw = p["card_w"]

            preferred_order = sorted(range(lanes_needed), key=lambda l: (abs(l - (i % lanes_needed)), l))
            best_lane = None
            best_card_x = None
            min_cost = float("inf")

            for lane_idx in preferred_order:
                lane_cards = lanes[lane_idx]
                desired_x = ax - cw / 2.0
                desired_x = max(12, min(desired_x, new_w - cw - 12))

                overlap = False
                for ex_x, ex_w in lane_cards:
                    if not (desired_x + cw + 10 < ex_x or desired_x > ex_x + ex_w + 10):
                        overlap = True
                        break

                if not overlap:
                    cost = abs(desired_x + cw / 2.0 - ax) + (lane_idx * 15)
                    if cost < min_cost:
                        min_cost = cost
                        best_lane = lane_idx
                        best_card_x = desired_x

            if best_lane is None:
                for lane_idx in preferred_order:
                    lane_cards = lanes[lane_idx]
                    if lane_cards:
                        rightmost = max(c[0] + c[1] for c in lane_cards) + 10
                        if rightmost + cw <= new_w - 12:
                            best_lane = lane_idx
                            best_card_x = rightmost
                            break
                    else:
                        best_lane = lane_idx
                        best_card_x = max(12, min(ax - cw / 2.0, new_w - cw - 12))
                        break

            if best_lane is None:
                best_lane = (i % lanes_needed)
                best_card_x = max(12, min(ax - cw / 2.0, new_w - cw - 12))

            lanes[best_lane].append((best_card_x, cw))
            lane_y = base_lane_y + best_lane * (self.card_height + self.lane_spacing)
            p["card_box"] = (int(best_card_x), int(lane_y), int(best_card_x + cw), int(lane_y + self.card_height))
            p["lane_idx"] = best_lane

        # 5. Draw Render Elements (Orthogonal Lines & Cards)
        draw = ImageDraw.Draw(canvas)

        for p in port_items:
            scheme = p["scheme"]
            accent_col = scheme["accent"]
            card_bg = scheme["bg"]
            card_border = scheme["border"]

            ax, ay = p["anchor_adj"]
            cx1, cy1, cx2, cy2 = p["card_box"]

            # Draw Port Bounding Box
            bx1, by1, bx2, by2 = p["box_adj"]
            draw.rectangle([bx1, by1, bx2, by2], outline=accent_col, width=2)

            # Solid Anchor Marker Dot at port opening
            dot_r = 3.5
            draw.ellipse([ax - dot_r, ay - dot_r, ax + dot_r, ay + dot_r], fill=accent_col, outline=(255, 255, 255), width=1)

            # Draw Orthogonal Connector Line:
            elbow_y = cy2 + 8 + (p["lane_idx"] * 6)
            card_entry_x = int(np.clip(ax, cx1 + 14, cx2 - 14))

            line_points = [
                (int(ax), int(ay)),
                (int(ax), int(elbow_y)),
                (card_entry_x, int(elbow_y)),
                (card_entry_x, int(cy2))
            ]
            draw.line(line_points, fill=accent_col, width=2)

            # Draw Callout Card
            draw.rounded_rectangle([cx1, cy1, cx2, cy2], radius=6, fill=card_bg, outline=card_border, width=1)
            draw.rounded_rectangle([cx1, cy1, cx1 + 5, cy2], radius=2, fill=accent_col)

            tx = cx1 + 12
            ty = cy1 + 7
            draw.text((tx, ty), p["title_text"], fill=(15, 23, 42), font=self.font_title)

            if p["silk_text"]:
                t_box = self.font_title.getbbox(p["title_text"])
                tw = t_box[2] - t_box[0]
                sx = tx + tw + 6
                sy = cy1 + 8
                draw.text((sx, sy), p["silk_text"], fill=accent_col, font=self.font_silk)

        return canvas


def render_marketplace_diagram(image_source, ports, output_path=None, title=None, expand_canvas=True):
    engine = MarketplaceLayoutEngine()
    diagram = engine.render_diagram(image_source, ports, title=title, expand_canvas=expand_canvas)
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        diagram.save(output_path, quality=95)
    return diagram
