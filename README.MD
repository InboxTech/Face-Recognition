# HelloQ — Face Recognition Microservice

<p align="center">
  <strong>🧠 Identity Verification via Multi-Angle Face Comparison</strong><br/>
  Python FastAPI service with InsightFace ArcFace + MediaPipe liveness detection
</p>

---

## Overview

Standalone microservice that verifies user identity by comparing 4 guided selfies against profile photos. Uses InsightFace ArcFace for face embedding extraction and MediaPipe FaceMesh for head pose estimation and liveness detection.

### Verification Pipeline

```
4 Selfies + Reference Photos
    ↓
Decode & Validate Images
    ↓
Extract Face Embeddings (InsightFace buffalo_sc)
    ↓
Slot Validation (front, left, right, smile poses)
    ↓
Liveness Check (yaw variation ≥ 25° across selfies)
    ↓
Duplicate Detection (SHA-256 + cosine dist < 0.02)
    ↓
Cross-Selfie Consistency (same person? dist < 0.50)
    ↓
Pairwise Comparison (each ref × each selfie, dist < 0.55)
    ↓
Match Ratio ≥ 50% → ✅ VERIFIED | < 50% → ❌ REJECTED
```

---

## Tech Stack

| Category | Technology | Version |
|---|---|---|
| **Framework** | FastAPI | 0.135.x |
| **Server** | Uvicorn | 0.42.x |
| **Face Detection** | InsightFace (buffalo_sc, buffalo_l) | 0.7.3 |
| **Pose Estimation** | MediaPipe FaceMesh | 0.10.x |
| **Computer Vision** | OpenCV | 4.8.x |
| **Math/ML** | NumPy, SciPy, scikit-learn | — |
| **Validation** | Pydantic | 2.x |

---

## API Endpoints

### `POST /api/v1/verify` — Full Multi-Selfie Verification

Accepts 4 selfies + 1-2 reference photos as `multipart/form-data`. Returns verification result with confidence score.

### `POST /api/v1/pose` — Real-Time Head Pose Estimation

Accepts a single camera frame. Returns yaw/pitch/roll angles and slot validity for real-time UI guidance.

```
Response: {
  detected: true,
  yaw: -35.2, pitch: 5.1, roll: 2.3,
  slots: { front: false, left: true, right: false, smile: false },
  thresholds: { ... }
}
```

### `POST /api/v1/verify/quick` — Quick 1-vs-1 Comparison

Fast single-photo comparison for profile photo updates.

### `GET /health` — Health Check

Returns service status and model load state.

---

## Key Parameters

| Parameter | Value | Description |
|---|---|---|
| `MATCH_THRESHOLD` | 0.55 | Max cosine distance for a match |
| `MIN_MATCH_RATIO` | 0.50 | Min fraction of matching pairs |
| `CROSS_SELFIE_THRESHOLD` | 0.50 | Max distance between selfie pairs |
| `NEAR_DUPLICATE_THRESHOLD` | 0.02 | Min distance to not be "same photo" |
| `MIN_YAW_VARIATION` | 25.0° | Min yaw range across all selfies |

### Pose Slot Thresholds

| Slot | Yaw | Pitch | Roll |
|---|---|---|---|
| Front | \|yaw\| ≤ 12° | \|pitch\| ≤ 30° | \|roll\| ≤ 15° |
| Left Turn | 20° ≤ yaw ≤ 60° | — | \|roll\| ≤ 18° |
| Right Turn | −60° ≤ yaw ≤ −20° | — | \|roll\| ≤ 18° |
| Smile | \|yaw\| ≤ 12° | \|pitch\| ≤ 30° | + smile detection |

---

## Getting Started

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate
# Activate (macOS/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start server
python main.py
# ✅ Running on http://localhost:8000
```

### Verify
```bash
curl http://localhost:8000/health
```

### Interactive Test UI
Open `test_face_recognition.html` in a browser for a visual testing interface.

---

## Project Structure

```
Face-Recognition/
├── main.py                  # FastAPI app, lifespan, health routes
├── config.py                # Pydantic settings: thresholds, pose limits
├── routes.py                # API endpoints: /verify, /pose, /verify/quick
├── service.py               # Core verification pipeline (23KB)
├── pose_estimation.py       # MediaPipe FaceMesh + solvePnP head pose
├── schemas.py               # Pydantic response models
├── requirements.txt         # Python dependencies
├── Dockerfile               # Container image
├── Caddyfile                # Reverse proxy config
└── test_face_recognition.html  # Interactive test UI
```

---

## Related Repositories

| Repository | Description |
|---|---|
| [frontend](../frontend) | React Native mobile app (selfie capture UI) |
| [backend](../backend) | Node.js API (orchestrates verification flow) |

---

*HelloQ Engineering — Built with 🏳️‍🌈 Pride*