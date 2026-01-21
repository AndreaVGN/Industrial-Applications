from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import time
import uuid
import os
import json
import shutil
import cv2
import numpy as np
import onnxruntime as ort

# ===============================
# CONFIG
# ===============================
ONNX_MODEL_PATH = "yolo_small_weights.onnx"
INPUT_SIZE = 640
VIOLENCE_CLASS_ID = 1

VIDEO_IN = "videos/input"
VIDEO_OUT = "videos/output"
LOG_PATH = "logs/timings.json"

os.makedirs(VIDEO_IN, exist_ok=True)
os.makedirs(VIDEO_OUT, exist_ok=True)
os.makedirs("logs", exist_ok=True)

# ===============================
# APP
# ===============================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
def ui():
    with open("static/index.html", encoding="utf-8") as f:
        return f.read()


# ===============================
# ONNX SESSION (LOAD ONCE)
# ===============================
session = ort.InferenceSession(
    ONNX_MODEL_PATH,
    providers=["CPUExecutionProvider"]  # CUDAExecutionProvider se disponibile
)
input_name = session.get_inputs()[0].name

LAST_EVENT = None

# ===============================
# HELPERS
# ===============================
def preprocess(frame):
    img = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img


def log_timing(entry):
    data = []
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH) as f:
            data = json.load(f)
    data.append(entry)
    with open(LOG_PATH, "w") as f:
        json.dump(data, f, indent=2)

# ===============================
# ENDPOINTS
# ===============================

@app.post("/api/process")
async def process_video(file: UploadFile = File(...)):
    global LAST_EVENT

    clip_id = str(uuid.uuid4())
    in_path = f"{VIDEO_IN}/{clip_id}.mp4"
    out_path = f"{VIDEO_OUT}/{clip_id}.mp4"

    with open(in_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    cap = cv2.VideoCapture(in_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    scale_x = w / INPUT_SIZE
    scale_y = h / INPUT_SIZE

    out = cv2.VideoWriter(
    out_path,
    cv2.VideoWriter_fourcc(*"avc1"),
    fps,
    (w, h)
    )


    violence = False
    frames = 0
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frames += 1
        input_tensor = preprocess(frame)

        # output: (1, N, 6)
        outputs = session.run(None, {input_name: input_tensor})[0][0]

        for x1, y1, x2, y2, conf, cls in outputs:
            if int(cls) == VIOLENCE_CLASS_ID and conf >= 0.4:
                violence = True

                # scale bbox to original frame
                x1 = int(x1 * scale_x)
                y1 = int(y1 * scale_y)
                x2 = int(x2 * scale_x)
                y2 = int(y2 * scale_y)

                cv2.rectangle(frame, (x1, y1), (x2, y2),
                              (0, 0, 255), 2)

        out.write(frame)

    cap.release()
    out.release()

    elapsed_ms = (time.time() - t0) * 1000

    LAST_EVENT = {
        "clip_id": clip_id,
        "prediction": "Violence" if violence else "NoViolence",
        "prob": None,
        "probs": None,
        "confirmed": False,
        "mode": "video",
        "device_id": "PC",
        "ts_utc_ms": int(time.time() * 1000),
        "timings_ms": {
            "total": elapsed_ms,
            "per_frame": elapsed_ms / max(frames, 1)
        }
    }

    log_timing({
        "clip_id": clip_id,
        "prediction": LAST_EVENT["prediction"],
        "frames": frames,
        "time_ms": elapsed_ms,
        "device": "PC"
    })

    return JSONResponse(LAST_EVENT)


@app.get("/api/state")
def state():
    return {
        "server_mode": "PC-YOLO-ONNX",
        "last_event": LAST_EVENT
    }


@app.post("/api/confirm")
def confirm(data: dict):
    global LAST_EVENT
    if LAST_EVENT and data["clip_id"] == LAST_EVENT["clip_id"]:
        LAST_EVENT["confirmed"] = True
        LAST_EVENT["label"] = data["label"]
        return {"ok": True}
    return {"ok": False, "error": "clip_id mismatch"}


@app.get("/api/view/{clip_id}")
def view_video(clip_id: str):
    path = f"{VIDEO_OUT}/{clip_id}.mp4"
    return FileResponse(path, media_type="video/mp4")
