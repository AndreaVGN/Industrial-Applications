import subprocess
import requests
import os
import uuid
import time
import argparse
import json
import cv2

from ultralytics import YOLO

# =========================
# CONFIG
# =========================
PC_IP = "172.20.10.12"
PC_PORT = 9000

MODE = "edge_video"        # "edge_video" | "edge_inference"

VIDEO_SECONDS = 5
WIDTH = 640
HEIGHT = 480
FPS = 10

OUT_DIR = "/home/pi/videos"
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE_ID = "raspberry_pi_cam_v3"

# =========================
# YOLO EDGE CONFIG
# =========================
USE_EDGE_INFERENCE = (MODE == "edge_inference")
MODEL_VARIANT = "small"                     # "nano" | "small"
YOLO_MODEL_PATH = "yolo_small_weights.pt"   # path locale sul Raspberry
IMG_SIZE = 320                              # OBBLIGATORIO per small
CONF_THRES = 0.4
SAMPLE_EVERY_N = 3                          # processa 1 frame ogni N

# =========================
# ENDPOINTS
# =========================
API_PROCESS = f"http://{PC_IP}:{PC_PORT}/api/process"
API_PUSH = f"http://{PC_IP}:{PC_PORT}/api/push"

# =========================
# LOAD YOLO MODEL (ONCE)
# =========================
yolo_model = None

if USE_EDGE_INFERENCE:
    print(f"[INFO] Loading YOLOv8-{MODEL_VARIANT} model on Raspberry...")
    yolo_model = YOLO(YOLO_MODEL_PATH)
    print("[INFO] YOLO model loaded.")

# =========================
# RECORD VIDEO
# =========================
def record_video(path):
    cmd = [
        "rpicam-vid",
        "-t", str(VIDEO_SECONDS * 1000),
        "--width", str(WIDTH),
        "--height", str(HEIGHT),
        "--framerate", str(FPS),
        "--codec", "h264",
        "--profile", "baseline",
        "--nopreview",
        "-o", path
    ]
    subprocess.run(cmd, check=True)

# =========================
# LOCAL YOLO INFERENCE
# =========================
def run_local_inference(video_path):
    cap = cv2.VideoCapture(video_path)

    violence = False
    max_conf = 0.0
    frame_id = 0

    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        if frame_id % SAMPLE_EVERY_N != 0:
            continue

        results = yolo_model(
            frame,
            imgsz=IMG_SIZE,
            conf=CONF_THRES,
            device="cpu",
            verbose=False
        )

        for det in results[0].boxes:
            cls = int(det.cls)
            conf = float(det.conf)

            if cls == 1:  # Violence
                violence = True
                max_conf = max(max_conf, conf)

    cap.release()

    t1 = time.time()

    return {
        "prediction": "Violence" if violence else "NoViolence",
        "confidence": round(max_conf, 3),
        "inference_ms": int((t1 - t0) * 1000)
    }

# =========================
# MAIN
# =========================
clip_id = str(uuid.uuid4())
video_path = os.path.join(OUT_DIR, f"clip_{clip_id}.mp4")

print("[INFO] Recording video...")
ts_start_capture = int(time.time() * 1000)
record_video(video_path)
ts_end_capture = int(time.time() * 1000)

print(f"[INFO] Video saved: {video_path}")

# =========================
# MODE A: EDGE VIDEO
# =========================
if MODE == "edge_video":
    print("[INFO] Uploading video to central server...")

    with open(video_path, "rb") as f:
        r = requests.post(
            API_PROCESS,
            files={"file": (os.path.basename(video_path), f, "video/mp4")},
            data={
                "device_id": DEVICE_ID,
                "clip_id": clip_id,
                "ts_start_capture": ts_start_capture
            },
            timeout=120
        )

    print("[INFO] Server response:", r.status_code)
    print(r.text)

# =========================
# MODE B: EDGE INFERENCE
# =========================
elif MODE == "edge_inference":
    print("[INFO] Running local YOLO inference on Raspberry...")

    ts_start_inf = int(time.time() * 1000)
    result = run_local_inference(video_path)
    ts_end_inf = int(time.time() * 1000)

    payload = {
        "clip_id": clip_id,
        "prediction": result["prediction"],
        "confidence": result["confidence"],
        "confirmed": False,
        "mode": "edge",
        "device_id": DEVICE_ID,
        "ts_utc_ms": ts_end_inf,
        "timings_ms": {
            "capture": ts_end_capture - ts_start_capture,
            "inference": result["inference_ms"],
            "edge_total": ts_end_inf - ts_start_capture
        },
        "model": f"yolo_{MODEL_VARIANT}"
    }

    print("[INFO] Sending inference result to server...")
    r = requests.post(API_PUSH, json=payload, timeout=30)

    print("[INFO] Server response:", r.status_code)
    print(r.text)

else:
    raise ValueError("Invalid MODE")
