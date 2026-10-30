"""
YOLO Model Training Script for Custom Hardware Port Detection
Supports YOLOv8 and YOLO11 via the Ultralytics framework.
"""

import os
import sys
from ultralytics import YOLO


def create_sample_dataset_yaml(yaml_path="dataset.yaml"):
    """Creates a sample dataset configuration YAML if one doesn't exist."""
    content = """# Dataset configuration for Ericsson Hardware Port Detection
path: ./dataset  # Dataset root directory
train: images/train  # Train images (relative to 'path')
val: images/val      # Validation images (relative to 'path')

# Class definitions
names:
  0: RF_Port_4.3-10
  1: SFP_Optical_Fronthaul
  2: DC_Power_Input
  3: ALD_RET_Control
  4: External_Alarm
  5: Ground_Lug
"""
    if not os.path.exists(yaml_path):
        with open(yaml_path, "w") as f:
            f.write(content)
        print(f"[INFO] Created template dataset configuration at: {yaml_path}")


def train_model(
    data_yaml="dataset.yaml",
    base_model="yolov8n.pt",  # Nano model: fast, light, great for real-time edge devices
    epochs=60,
    imgsz=640,
    batch_size=16,
    project_name="ericsson_ports",
    run_name="yolov8n_ports_exp1"
):
    """
    Trains YOLO on the annotated port dataset.
    """
    print(f"[INFO] Initializing base model: {base_model}")
    model = YOLO(base_model)

    print(f"[INFO] Starting training on {data_yaml} for {epochs} epochs...")
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        project=project_name,
        name=run_name,
        patience=20,          # Early stopping patience
        save=True,
        device=0 if os.system("nvidia-smi > /dev/null 2>&1") == 0 else "cpu"
    )

    best_weights = os.path.join(project_name, run_name, "weights", "best.pt")
    print(f"\n[SUCCESS] Training finished! Best model weights saved to:\n{best_weights}")
    print(f"\nTo run inference with your newly trained model:")
    print(f"python3 port_detector.py --model {best_weights} --image test.jpg")


if __name__ == "__main__":
    yaml_file = "dataset.yaml"
    create_sample_dataset_yaml(yaml_file)

    if not os.path.exists("./dataset/images/train"):
        print("\n[NOTE] No dataset directory found yet.")
        print("1. Label ~50-100 hardware photos on Roboflow (https://roboflow.com) or CVAT.")
        print("2. Export in 'YOLOv8' format and unzip into the './dataset' folder.")
        print("3. Run this script: python3 train_yolo.py")
    else:
        train_model()
