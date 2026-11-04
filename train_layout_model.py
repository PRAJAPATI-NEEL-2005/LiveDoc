"""
LiveDoc Model Training Pipeline
Trains or fine-tunes a YOLOv11 deep-learning model for Document Layout Analysis
(Title, Heading, Paragraph, Table, Image, Diagram).

Usage:
  # 1. Quick test training on generated document dataset (10 epochs):
  python train_layout_model.py --generate-sample-data --epochs 10

  # 2. Train on your own DocLayNet / custom dataset:
  python train_layout_model.py --data path/to/dataset.yaml --epochs 50
"""

import os
import sys
import shutil
import argparse
import random
import cv2
import numpy as np
from ultralytics import YOLO

CLASSES = ["Title", "Heading", "Paragraph", "Table", "Image", "Diagram"]


def generate_synthetic_document_dataset(dataset_dir="dataset", num_train=40, num_val=10):
    """
    Generates a synthetic document layout dataset with realistic document components
    (Titles, Headings, Multi-column Paragraphs, Tables, and Mock Figures)
    and saves them in standard YOLO format.
    """
    print(f"\n[1/3] Generating synthetic document layout dataset in '{dataset_dir}'...")

    for split in ["train", "val"]:
        os.makedirs(os.path.join(dataset_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(dataset_dir, "labels", split), exist_ok=True)

    counts = {"train": num_train, "val": num_val}

    for split, count in counts.items():
        for idx in range(count):
            img_w, img_h = 800, 1050
            # Paper background (slight off-white realistic paper texture)
            doc_img = np.ones((img_h, img_w, 3), dtype=np.uint8) * random.randint(245, 255)
            # Add subtle texture variation
            noise = np.random.randint(-5, 5, (img_h, img_w, 3), dtype=np.int16)
            doc_img = np.clip(doc_img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

            boxes = []  # list of (class_id, x1, y1, x2, y2)
            curr_y = 50

            # 1. Title (Class 0)
            t_h = 40
            t_w = random.randint(400, 650)
            t_x1 = (img_w - t_w) // 2
            cv2.rectangle(doc_img, (t_x1, curr_y), (t_x1 + t_w, curr_y + t_h), (30, 30, 30), -1)
            cv2.putText(doc_img, "ANNUAL PROGRESS REPORT", (t_x1 + 20, curr_y + 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            boxes.append((0, t_x1, curr_y, t_x1 + t_w, curr_y + t_h))
            curr_y += t_h + 30

            # 2. Heading (Class 1)
            h_h = 28
            h_w = random.randint(250, 450)
            cv2.rectangle(doc_img, (50, curr_y), (50 + h_w, curr_y + h_h), (60, 60, 60), -1)
            cv2.putText(doc_img, "1. Executive Summary & Overview", (55, curr_y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            boxes.append((1, 50, curr_y, 50 + h_w, curr_y + h_h))
            curr_y += h_h + 20

            # 3. Paragraph (Class 2)
            p_h = random.randint(70, 110)
            cv2.rectangle(doc_img, (50, curr_y), (750, curr_y + p_h), (220, 220, 220), 1)
            # Draw mock text lines
            for ly in range(curr_y + 15, curr_y + p_h - 10, 18):
                lw = random.randint(550, 680)
                cv2.line(doc_img, (60, ly), (60 + lw, ly), (80, 80, 80), 2)
            boxes.append((2, 50, curr_y, 750, curr_y + p_h))
            curr_y += p_h + 25

            # 4. Table (Class 3)
            tbl_h = random.randint(140, 190)
            tbl_w = 700
            tbl_x1 = 50
            cv2.rectangle(doc_img, (tbl_x1, curr_y), (tbl_x1 + tbl_w, curr_y + tbl_h), (40, 40, 40), 2)
            # Table horizontal grid lines
            num_rows = 4
            row_step = tbl_h // num_rows
            for r in range(1, num_rows):
                cv2.line(doc_img, (tbl_x1, curr_y + r * row_step), (tbl_x1 + tbl_w, curr_y + r * row_step), (100, 100, 100), 1)
            # Table vertical grid lines
            num_cols = 4
            col_step = tbl_w // num_cols
            for c in range(1, num_cols):
                cv2.line(doc_img, (tbl_x1 + c * col_step, curr_y), (tbl_x1 + c * col_step, curr_y + tbl_h), (100, 100, 100), 1)
            boxes.append((3, tbl_x1, curr_y, tbl_x1 + tbl_w, curr_y + tbl_h))
            curr_y += tbl_h + 25

            # 5. Image or Diagram (Class 4 / 5)
            fig_h = min(random.randint(120, 180), img_h - curr_y - 40)
            if fig_h > 60:
                fig_w = random.randint(300, 500)
                fig_x = 50
                cv2.rectangle(doc_img, (fig_x, curr_y), (fig_x + fig_w, curr_y + fig_h), (180, 150, 120), -1)
                cv2.putText(doc_img, "Figure 1: Performance Metric", (fig_x + 20, curr_y + fig_h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                boxes.append((4, fig_x, curr_y, fig_x + fig_w, curr_y + fig_h))

            # Save image
            img_filename = f"doc_{split}_{idx:04d}.jpg"
            img_path = os.path.join(dataset_dir, "images", split, img_filename)
            cv2.imwrite(img_path, doc_img)

            # Save YOLO format labels (class x_center y_center width height normalized)
            label_filename = f"doc_{split}_{idx:04d}.txt"
            label_path = os.path.join(dataset_dir, "labels", split, label_filename)
            with open(label_path, "w") as lf:
                for cls_id, x1, y1, x2, y2 in boxes:
                    xc = ((x1 + x2) / 2.0) / img_w
                    yc = ((y1 + y2) / 2.0) / img_h
                    bw = (x2 - x1) / float(img_w)
                    bh = (y2 - y1) / float(img_h)
                    lf.write(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

    # Generate data.yaml
    yaml_path = os.path.join(dataset_dir, "data.yaml")
    abs_dataset_dir = os.path.abspath(dataset_dir)
    with open(yaml_path, "w") as yf:
        yf.write(f"path: {abs_dataset_dir}\n")
        yf.write("train: images/train\n")
        yf.write("val: images/val\n")
        yf.write(f"nc: {len(CLASSES)}\n")
        yf.write(f"names: {CLASSES}\n")

    print(f"✓ Synthetic dataset created at: {yaml_path}")
    return yaml_path


def train_yolo_model(data_yaml_path, epochs=10, imgsz=640, base_model="yolo11n.pt"):
    """
    Trains YOLOv11 on the document layout dataset and outputs to models/yolo11_doc.pt.
    """
    print(f"\n[2/3] Loading base model '{base_model}'...")
    model = YOLO(base_model)

    print(f"\n[3/3] Starting training for {epochs} epoch(s)...")
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        project="runs/layout_train",
        name="yolo11_doc",
        exist_ok=True
    )

    # Locate trained weights
    best_weights = os.path.join("runs", "layout_train", "yolo11_doc", "weights", "best.pt")
    target_weights = os.path.join("models", "yolo11_doc.pt")

    os.makedirs("models", exist_ok=True)
    if os.path.exists(best_weights):
        shutil.copy(best_weights, target_weights)
        print(f"\n🎉 SUCCESS: Trained model weights saved directly to:\n   -> {target_weights}")
        print("LiveDoc will now automatically load and use this trained YOLO model!")
    else:
        print(f"\nNote: Training finished. If best.pt is located elsewhere, copy it to {target_weights}.")


def main():
    parser = argparse.ArgumentParser(description="LiveDoc YOLO Layout Model Training Script")
    parser.add_argument("--data", type=str, default=None, help="Path to data.yaml dataset file")
    parser.add_argument("--generate-sample-data", action="store_true", help="Generate synthetic document dataset")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs (default 10)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution for training (default 640)")
    parser.add_argument("--base-model", type=str, default="yolo11n.pt", help="Pretrained YOLO base model")

    args = parser.parse_args()

    if args.data is None or args.generate_sample_data:
        data_yaml = generate_synthetic_document_dataset()
    else:
        data_yaml = args.data

    train_yolo_model(data_yaml, epochs=args.epochs, imgsz=args.imgsz, base_model=args.base_model)


if __name__ == "__main__":
    main()
