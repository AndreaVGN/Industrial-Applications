import subprocess
import requests
import os
import uuid
import time
import cv2
import numpy as np
import onnxruntime as ort

# =========================
# CONFIG
# =========================
PC_IP = "192.168.1.71"
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
# ONNX YOLO CONFIG
# =========================
USE_EDGE_INFERENCE = (MODE == "edge_inference")

ONNX_MODEL_PATH = "yolo_small_weights.onnx"
IMG_SIZE = 320
CONF_THRES = 0.4
SAMPLE_EVERY_N = 3   # processa 1 frame ogni N

# =========================
# ENDPOINTS
# =========================
API_PROCESS = f"http://{PC_IP}:{PC_PORT}/api/process"
API_PUSH = f"http://{PC_IP}:{PC_PORT}/api/push"

# =========================
# LOAD ONNX MODEL (ONCE)
# =========================
session = None
input_name = None

if USE_EDGE_INFERENCE:
    print("[INFO] Loading YOLOv8-small ONNX model on Raspberry...")
    session = ort.InferenceSession(
        ONNX_MODEL_PATH,
        providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    print("[INFO] ONNX model loaded.")

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
# PREPROCESS FRAME (YOLO ONNX)
# =========================
def preprocess(frame):
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))   # HWC -> CHW
    img = np.expand_dims(img, axis=0)    # CHW -> BCHW
    return img

# =========================
# LOCAL ONNX INFERENCE
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

        inp = preprocess(frame)
        outputs = session.run(None, {input_name: inp})[0][0]
        # output shape: (300, 6)
        # [x1, y1, x2, y2, conf, class]

        for x1, y1, x2, y2, conf, cls in outputs:
            if conf < CONF_THRES:
                continue
            if int(cls) == 1:  # Violence
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
    print("[INFO] Running local ONNX inference on Raspberry...")

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
        "model": "yolo_small_onnx"
    }

    print("[INFO] Sending inference result to server...")
    r = requests.post(API_PUSH, json=payload, timeout=30)

    print("[INFO] Server response:", r.status_code)
    print(r.text)

else:
    raise ValueError("Invalid MODE")
