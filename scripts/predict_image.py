#!/usr/bin/env python3
"""YOLO26n pretrained smoke test on a local image (no training)."""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import torch
from ultralytics import YOLO

PROJECT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True,
                        help="Path to an input image; e.g. data/bus.jpg")
    parser.add_argument("--weights", default="yolo26n.pt",
                        help="Checkpoint; standard name downloads automatically")
    parser.add_argument("--output", default="outputs/prediction.jpg",
                        help="Annotated JPEG path relative to project")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="auto", help="auto, cpu, or 0")
    args = parser.parse_args()

    source = Path(args.source).expanduser()
    if not source.is_file():
        parser.error(f"Input image not found: {source}")
    output = Path(args.output)
    if not output.is_absolute():
        output = PROJECT / output
    output.parent.mkdir(parents=True, exist_ok=True)

    device = ("0" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    weights = Path(args.weights)
    if weights.is_file():
        model_name = str(weights.resolve())
    elif args.weights == "yolo26n.pt":
        model_name = args.weights
    else:
        parser.error(f"Missing weights: {weights}")

    model = YOLO(model_name)
    t0 = time.perf_counter()
    result = model.predict(
        source=str(source), device=device,
        imgsz=args.imgsz, conf=args.conf,
        verbose=False, save=False
    )[0]
    wall_ms = 1000 * (time.perf_counter() - t0)
    result.save(filename=str(output))

    detections = []
    for box in result.boxes:
        cls = int(box.cls[0].item())
        detections.append({
            "class": result.names[cls],
            "confidence": round(float(box.conf[0]), 4),
            "xyxy": [round(float(v), 1) for v in box.xyxy[0].tolist()],
        })
    info = {
        "status": "success",
        "model": "YOLO26n pretrained",
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device != "cpu" and torch.cuda.is_available() else None,
        "cuda": torch.version.cuda,
        "image": str(source),
        "num_detections": len(detections),
        "classes": dict(Counter(x["class"] for x in detections)),
        "inference_time_ms": round(float(result.speed["inference"]), 3),
        "end_to_end_call_ms": round(wall_ms, 3),
        "predictions": detections,
        "note": "Demonstration only; not an accuracy benchmark or fine-tuned model.",
    }
    metadata = output.with_suffix(".json")
    metadata.write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: info[k] for k in ("status", "device", "gpu", "num_detections", "classes", "inference_time_ms")}, ensure_ascii=False))
    print(f"Saved image: {output}")
    print(f"Saved metadata: {metadata}")


if __name__ == "__main__":
    main()
