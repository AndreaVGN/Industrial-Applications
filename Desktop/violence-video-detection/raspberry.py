import subprocess
import requests
import os
import uuid
import time

# =========================
# CONFIG
# =========================
PC_IP = "192.168.1.71"
PC_PORT = 9000

VIDEO_SECONDS = 5
WIDTH = 640
HEIGHT = 480
FPS = 10

OUT_DIR = "/home/pi/videos"
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE_ID = "raspberry_pi_cam_v3"

API_PROCESS = f"http://{PC_IP}:{PC_PORT}/api/process"

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
# SEND VIDEO TO SERVER
# =========================
print("[INFO] Uploading video to central server...")

with open(video_path, "rb") as f:
    r = requests.post(
        API_PROCESS,
        files={
            "file": (os.path.basename(video_path), f, "video/mp4")
        },
        data={
            "device_id": DEVICE_ID,
            "clip_id": clip_id,
            "ts_start_capture": ts_start_capture,
            "ts_end_capture": ts_end_capture,
            "format": "h264_640x480_10fps"
        },
        timeout=120
    )

print("[INFO] Server response:", r.status_code)
print(r.text)
