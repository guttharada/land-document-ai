"""Export best.pt and ocr_thai.pt to ONNX for the browser demo in docs/.

    .venv/bin/pip install onnx==1.23.2 onnxscript==0.7.2 onnxslim==0.1.99 onnxruntime==1.32.0
    .venv/bin/python export_onnx.py
"""
import os, shutil, tempfile
from pathlib import Path

os.environ.update(YOLO_OFFLINE="true", YOLO_AUTOINSTALL="false")
import torch
from ultralytics import YOLO

from train_ocr import load

HERE = Path(__file__).parent
OUT = HERE / "docs" / "models"
OUT.mkdir(parents=True, exist_ok=True)

with tempfile.TemporaryDirectory() as d:  # ultralytics writes next to the weights
    pt = shutil.copy(HERE / "best.pt", d)
    onnx = YOLO(pt).export(format="onnx", imgsz=960, opset=17, dynamic=False, simplify=True, max_det=600, device="cpu")
    shutil.move(onnx, OUT / "yolo.onnx")

torch.onnx.export(load(HERE / "ocr_thai.pt"), (torch.zeros(1, 1, 48, 192),), str(OUT / "reader.onnx"),
                  input_names=["x"], output_names=["y"], opset_version=17, dynamo=False)
for p in sorted(OUT.iterdir()):
    print(p.name, f"{p.stat().st_size / 1e6:.1f} MB")
