# Video Violence Detection — Edge–Server System
 
Industrial Applications project — M.S. in Artificial Intelligence and Data Engineering, University of Pisa (A.Y. 2025/2026).
Authors: F. Galardi, M. Meazzini, A. Vagnoli.
 
An end-to-end system for automated violence detection in video streams, designed for constrained environments such as commercial transport vehicles. A low-cost embedded device captures the video, a server runs neural-network inference on the GPU, and a web interface lets a human operator confirm or correct each detection.
 
## Architecture
 
```
Raspberry Pi 3 B+ + Camera v2  ──HTTP──▶  FastAPI server (PC)  ──▶  Web UI (human-in-the-loop)
  rpicam-vid capture                       ONNX Runtime (GPU)
  ffmpeg frame sampling                    OpenCV preprocessing
  UUID + timing metadata                   JSON logging, FFmpeg annotation
```
 
- **Edge client (Raspberry Pi 3 B+).** Records short clips with `rpicam-vid` and sends either the full video or frames sampled locally with `ffmpeg` (1, 2 or 5 FPS). Each request carries metadata (clip UUID, device ID, timestamps, sampling parameters) so that capture, upload and inference latency can be measured.
- **Server (FastAPI).** Receives the uploads and runs inference with a YOLO-based violence detection network (~11.2M parameters, from an existing open-source implementation) exported to **ONNX** and run on **ONNX Runtime** with GPU acceleration. A clip is labeled *Violence* if at least one detection of that class exceeds the confidence threshold. Predictions, timings and metadata are logged as JSON.
- **Web UI.** Shows the annotated clip, with bounding boxes on the people involved, and provides buttons for the operator to confirm or correct the label.
## Design choices
 
- **Server-side inference.** We first tried running several public models directly on the Raspberry Pi 3 B+. We dropped this approach because of library and architecture incompatibilities and because of the board's RAM and latency limits.
- **Modular design.** The model can be swapped without changing the rest of the pipeline.
## Experiments
 
- **Dataset.** A custom dataset of 60 clips (30 *Violence*, 30 *NoViolence*), recorded in a setup that simulates two passengers inside a vehicle. The recordings are not released for privacy reasons.
- **Modes tested.** Frame-based inference at 1, 2 and 5 FPS, and video-based inference at 10 FPS, each evaluated at multiple decision thresholds.
- **Metrics.** Accuracy, precision, recall, F1, confusion matrices, and upload and inference latency.
### Key findings
 
- **Frame rate.** A higher FPS improves recall and reduces false negatives, but increases end-to-end latency. At low FPS the upload time dominates; at high FPS the inference time does.
- **Threshold.** Higher thresholds increase precision but make recall drop sharply.
- **Selected operating point.** **5 FPS with a 0.5 threshold** gave no false negatives on the evaluated dataset, with acceptable latency.
## Repository structure
 
```
violence-video-detection/
├── app.py          # FastAPI inference server + web UI
├── raspberry.py    # acquisition and upload script (Raspberry Pi)
└── ...             # model, demo video, requirements
```
 
## Running the demo
 
1. Start the server on the PC: `python app.py`
2. Connect to the Raspberry Pi over SSH and run `python raspberry.py`. This records a 4-second clip and sends it to the server.
3. Open the web UI to view the result and confirm or correct the detection.
## Future work
 
- Longer and more varied datasets.
- On-device inference on more powerful edge hardware.
- Temporal-consistency logic across consecutive frames.
 
