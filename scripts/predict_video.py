#!/usr/bin/env python3
"""YOLO26n pretrained vehicle detection on a short video or camera stream."""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

PROJECT = Path(__file__).resolve().parents[1]
COCO_VEHICLES = [2, 3, 5, 7]  # car, motorcycle, bus, truck


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True,
                        help="Path to MP4, RTSP URL, or camera index")
    parser.add_argument("--weights", default="yolo26n.pt")
    parser.add_argument("--output", default="outputs/vehicle_detection.mp4")
    parser.add_argument("--seconds", type=float, default=10,
                        help="Maximum time from source to process")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="auto", help="auto, cpu, or 0")
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("--seconds must be positive")

    device = ("0" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    weights = Path(args.weights)
    if weights.is_file():
        model_name = str(weights.resolve())
    elif args.weights == "yolo26n.pt":
        model_name = args.weights
    else:
        parser.error(f"Missing model weights: {weights}")

    source = int(args.source) if args.source.isdecimal() else args.source
    if isinstance(source, str) and "://" not in source and not Path(source).is_file():
        parser.error(f"Input video not found: {source}")
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV cannot open video source: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 240:
        fps = 20.0  # fallback for RTSP sources without reliable fps metadata
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError("Invalid video frame dimensions")

    output = Path(args.output)
    if not output.is_absolute():
        output = PROJECT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Unable to open MP4 output writer")

    max_frames = max(1, round(fps * args.seconds))
    model = YOLO(model_name)
    counts = Counter()
    prediction_time = []
    frames = 0
    start = time.perf_counter()
    try:
        while frames < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            result = model.predict(frame, device=device, imgsz=args.imgsz,
                                   conf=args.conf, classes=COCO_VEHICLES,
                                   verbose=False)[0]
            annotated = result.plot(line_width=2)
            frame_counts = Counter(result.names[int(i)] for i in result.boxes.cls.tolist())
            counts.update(frame_counts)
            cv2.rectangle(annotated, (0, 0), (width, 60), (15, 27, 38), -1)
            cv2.putText(annotated, "YOLO26n | Pretrained Vehicle Detection",
                        (20, 27), cv2.FONT_HERSHEY_SIMPLEX, .78,
                        (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(annotated,
                        "car: {}  motorcycle: {}  bus: {}  truck: {}".format(
                            *(frame_counts.get(s, 0) for s in ("car", "motorcycle", "bus", "truck"))
                        ),
                        (20, 51), cv2.FONT_HERSHEY_SIMPLEX, .52,
                        (130, 230, 195), 1, cv2.LINE_AA)
            writer.write(annotated)
            prediction_time.append(float(result.speed["inference"]))
            frames += 1
    finally:
        cap.release()
        writer.release()

    elapsed = time.perf_counter() - start
    if frames == 0:
        raise RuntimeError("Source contained no readable frames")
    info = {
        "status": "success",
        "model": "YOLO26n pretrained",
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device != "cpu" and torch.cuda.is_available() else None,
        "width": width, "height": height, "source_fps": fps,
        "frames_processed": frames,
        "playback_duration_sec": round(frames / fps, 3),
        "end_to_end_throughput_fps": round(frames / elapsed, 3),
        "avg_inference_ms": round(sum(prediction_time) / frames, 3),
        "detections_summed_across_frames": dict(counts),
        "warning": "Framewise box counts are NOT unique vehicle counts. No accuracy metrics.",
    }
    metadata = output.with_suffix(".json")
    metadata.write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(info, ensure_ascii=False, indent=2))
    print(f"Saved video: {output}")


if __name__ == "__main__":
    main()
