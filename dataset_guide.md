# Guide: Dataset Collection & Training for Ericsson Port Detection

## 1. Can You Scrape `ericsson.com`?
While scraping is technically possible in web automation, the URL provided (`https://www.ericsson.com/en/portfolio/networks/ericsson-radio-system`) has two major blockers for training a YOLO model:

1. **Anti-Bot Security (HTTP 403 / Cloudflare / Akamai):** Direct scraping requests to Ericsson's marketing site return `403 Forbidden` unless handled with complex headless browser session headers.
2. **Lack of Hardware Port Close-ups:** Marketing product pages feature stylized 3D promotional renders, lifestyle shots, and distant tower views. They rarely show high-resolution, head-on photos of the bottom interface panels where ports are located.
3. **Missing Annotations:** Even if raw images were downloaded, YOLO is a **supervised object detection** model—it requires bounding box coordinates `(x_center, y_center, width, height)` for each port.

---

## 2. Where to Get High-Quality Ericsson Port Images

### Option A: FCC ID Filings (The Best Free Source for Hardware)
Every radio unit deployed in the United States must submit public certification filings to the Federal Communications Commission (FCC).
- Ericsson's official FCC Grantee Code is **`TA8`**.
- Go to [fccid.io](https://fccid.io/TA8) and search for Ericsson radio models (e.g., *Radio 4415, Radio 8843, Radio 2219, Baseband 6630, AIR 32*).
- Click on the filing and download the **"External Photos"** PDF exhibit.
- These exhibits contain pristine, high-resolution, uncompressed photographs of the bottom connector plate, showing exact port geometry and silkscreen text!

### Option B: Technical Product Descriptions (TPDs) & Installation Manuals
Search Google with the `filetype:pdf` operator:
```text
"Ericsson Radio System" "Hardware Highlights" filetype:pdf
"Radio 4415" "User Description" "interfaces" filetype:pdf
"Baseband 6630" "External Interfaces" filetype:pdf
```
Sites like Scribd, SlideShare, and telecom equipment manuals host these documents, which contain clear front/bottom panel connector layouts.

### Option C: Real Site Photos / Smartphone Video (50–100 Images)
If you or your team have access to physical Ericsson equipment, take a 30-second 4K video moving around the bottom connection panel under different lighting conditions. You can extract frames using OpenCV:
```python
import cv2

cap = cv2.VideoCapture("radio_panel_video.mp4")
frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    if frame_idx % 15 == 0:  # Save 2 frames per second
        cv2.imwrite(f"dataset/raw_frames/img_{frame_idx:04d}.jpg", frame)
    frame_idx += 1
cap.release()
```

---

## 3. Recommended Port Classes
Standardize your annotation classes across Ericsson Radio System hardware:

| Class Name | Physical Connector Type | Typical Function | Typical Silkscreen Text |
| :--- | :--- | :--- | :--- |
| `RF_Port` | 4.3-10 or 7-16 coaxial DIN | Antenna Transmit / Receive | `ANT A`, `ANT B`, `TX/RX` |
| `Optical_SFP` | SFP+ / SFP28 cage (FullAXS) | Fronthaul eCPRI / CPRI fiber | `DATA 1`, `DATA 2`, `1`, `2` |
| `DC_Power` | 2-wire screw-lock DC connector | -48V DC power feed | `-48V`, `DC IN`, `+ -` |
| `ALD_RET` | 8-pin circular DIN connector | Antenna Line Device / Tilt | `ALD`, `RET` |
| `External_Alarm`| 14-pin circular DIN connector | Site sensors & alarm input | `ALARM`, `EXT ALARM` |
| `Ground_Lug` | 2-hole bolted terminal lug | Chassis earth ground | `GND` / Ground symbol |

---

## 4. How to Label the Dataset (Roboflow in 15 Minutes)
1. Sign up for a free account at [Roboflow](https://roboflow.com).
2. Create a new project -> Project Type: **Object Detection**.
3. Upload 50–100 collected images.
4. Draw tight bounding boxes around each port and assign the class name.
5. In the **Generate** step:
   - Add augmentations: Slight Rotation (`-15° to +15°`), Brightness (`-25% to +25%`), Blur.
6. Export the dataset choosing the **YOLOv8** format.
7. Unzip into `ericsson_port_detector/dataset/`.

---

## 5. Training and Inference
Run the training script:
```bash
python3 train_yolo.py
```
Run real-time inference on a test photo or live webcam with labels rendered cleanly on top:
```bash
# On an image:
python3 port_detector.py --model runs/detect/train/weights/best.pt --image site_photo.jpg

# Live on webcam:
python3 port_detector.py --model runs/detect/train/weights/best.pt --webcam
```
