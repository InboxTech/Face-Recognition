"""
Face Recognition & Verification Service
========================================
(see prior docstring — unchanged in spirit)
"""

import hashlib
import cv2
import numpy as np
from PIL import Image
import io
import insightface
from insightface.app import FaceAnalysis
from fastapi import HTTPException
import logging
from config import settings
from schemas import PairResult, VerificationResponse, VerificationStatus

from pose_estimation import HeadPoseEstimator

logger = logging.getLogger(__name__)
pose_estimator = HeadPoseEstimator(static_image_mode=True)


# ── Model Loading ─────────────────────────────────────────────────────────────

def _load_model(name: str, det_size=(640, 640)):
    app = FaceAnalysis(name=name, providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=det_size)
    logger.info(f"InsightFace model '{name}' loaded.")
    return app


try:
    face_app = _load_model("buffalo_sc")
except Exception as _e:
    logger.error(f"buffalo_sc failed to load: {_e}")
    face_app = None

try:
    face_app_lm = _load_model("buffalo_l")
    _HAS_LM_MODEL = True
    logger.info("buffalo_l loaded — full smile detection enabled.")
except Exception as _e:
    face_app_lm = None
    _HAS_LM_MODEL = False
    logger.warning(f"buffalo_l not available ({_e}). Smile slot will use 5-pt heuristic.")


# ── Global Thresholds — all read from config, grouped by slot ────────────────

MIN_MATCH_RATIO          = settings.MIN_MATCH_RATIO
MATCH_THRESHOLD          = settings.MATCH_THRESHOLD
CROSS_SELFIE_THRESHOLD   = 0.50
NEAR_DUPLICATE_THRESHOLD = 0.02

# Slot 0 — front: yaw + pitch + roll
FRONT_YAW_MAX   = settings.FRONT_YAW_MAX
FRONT_PITCH_MAX = settings.FRONT_PITCH_MAX
FRONT_ROLL_MAX  = settings.FRONT_ROLL_MAX

# Slot 1 — left: yaw + roll only (pitch unconstrained)
LEFT_YAW_MIN  = settings.LEFT_YAW_MIN
LEFT_YAW_MAX  = settings.LEFT_YAW_MAX
LEFT_ROLL_MAX = settings.LEFT_ROLL_MAX

# Slot 2 — right: yaw + roll only (pitch unconstrained)
RIGHT_YAW_MIN  = settings.RIGHT_YAW_MIN
RIGHT_YAW_MAX  = settings.RIGHT_YAW_MAX
RIGHT_ROLL_MAX = settings.RIGHT_ROLL_MAX

# Slot 3 — smile: yaw + pitch + roll (same axes as front)
SMILE_YAW_MAX   = settings.SMILE_YAW_MAX
SMILE_PITCH_MAX = settings.SMILE_PITCH_MAX
SMILE_ROLL_MAX  = settings.SMILE_ROLL_MAX

# Smile landmark thresholds
SMILE_68PT_INNER_OPEN_RATIO = 0.020
SMILE_5PT_MOUTH_WIDTH_RATIO = 0.340


# ── Image Decoding ────────────────────────────────────────────────────────────

def decode_image(image_bytes: bytes, filename: str) -> np.ndarray:
    ext = filename.rsplit(".", 1)[-1].lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"'{filename}' has unsupported format. Allowed: {settings.ALLOWED_EXTENSIONS}",
        )
    max_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024
    if len(image_bytes) > max_bytes:
        raise HTTPException(
            status_code=400,
            detail=f"'{filename}' exceeds {settings.MAX_FILE_SIZE_MB} MB limit.",
        )
    try:
        arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is not None:
            return img
    except Exception:
        pass
    try:
        pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return np.array(pil)[:, :, ::-1]
    except Exception:
        pass
    raise HTTPException(
        status_code=400,
        detail=f"Could not read '{filename}'. Please try a different photo.",
    )


# ── Standalone pose estimation for /pose endpoint ────────────────────────────

def estimate_pose_only(img_bgr: np.ndarray) -> dict:
    """
    Lightweight pose estimation used by the live preview endpoint.
    Returns yaw/pitch/roll + per-slot validity flags using per-slot axis rules:
      front / smile : yaw + pitch + roll
      left  / right : yaw + roll only (pitch ignored)
    Frontend uses these flags to color the oval green/yellow/red in real time.
    """
    pose = pose_estimator.estimate_pose(img_bgr)
    if pose is None:
        return {
            "detected": False,
            "yaw": None,
            "pitch": None,
            "roll": None,
            "slots": {"front": False, "left": False, "right": False, "smile": False},
            "thresholds": _pose_thresholds_payload(),
        }

    yaw, pitch, roll = pose["yaw"], pose["pitch"], pose["roll"]

    # Each slot enforces only its relevant axes
    front_ok = (
        abs(roll)  <= FRONT_ROLL_MAX and
        abs(yaw)   <= FRONT_YAW_MAX  and
        abs(pitch) <= FRONT_PITCH_MAX
    )
    left_ok = (
        abs(roll) <= LEFT_ROLL_MAX and
        LEFT_YAW_MIN <= yaw <= LEFT_YAW_MAX
        # pitch NOT checked for side slots
    )
    right_ok = (
        abs(roll) <= RIGHT_ROLL_MAX and
        -RIGHT_YAW_MAX <= yaw <= -RIGHT_YAW_MIN
        # pitch NOT checked for side slots
    )
    smile_ok = (
        abs(roll)  <= SMILE_ROLL_MAX  and
        abs(yaw)   <= SMILE_YAW_MAX   and
        abs(pitch) <= SMILE_PITCH_MAX
        # smile gesture itself verified server-side at capture
    )

    return {
        "detected": True,
        "yaw":   yaw,
        "pitch": pitch,
        "roll":  roll,
        "slots": {
            "front": front_ok,
            "left":  left_ok,
            "right": right_ok,
            "smile": smile_ok,
        },
        "thresholds": _pose_thresholds_payload(),
    }


def _pose_thresholds_payload() -> dict:
    """
    Nested per-slot thresholds sent to the frontend so JS gauge/hint logic
    reads the same values the backend enforces — no duplication.
    """
    return {
        "front": {
            "yaw_max":   FRONT_YAW_MAX,
            "pitch_max": FRONT_PITCH_MAX,
            "roll_max":  FRONT_ROLL_MAX,
        },
        "left": {
            "yaw_min": LEFT_YAW_MIN,
            "yaw_max": LEFT_YAW_MAX,
            "roll_max": LEFT_ROLL_MAX,
        },
        "right": {
            "yaw_min": RIGHT_YAW_MIN,
            "yaw_max": RIGHT_YAW_MAX,
            "roll_max": RIGHT_ROLL_MAX,
        },
        "smile": {
            "yaw_max":   SMILE_YAW_MAX,
            "pitch_max": SMILE_PITCH_MAX,
            "roll_max":  SMILE_ROLL_MAX,
        },
    }


# ── Face Detection ────────────────────────────────────────────────────────────

def _bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _largest_face(faces):
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


def _detect(img_bgr: np.ndarray, filename: str, app=None):
    app = app or face_app
    if app is None:
        raise HTTPException(status_code=500, detail="Face recognition model not loaded.")
    try:
        faces = app.get(_bgr_to_rgb(img_bgr))
    except Exception as e:
        logger.error(f"InsightFace error on '{filename}': {e}")
        raise HTTPException(status_code=422, detail=f"Face detection failed for '{filename}': {e}")
    if not faces:
        raise HTTPException(
            status_code=422,
            detail=f"No face detected in '{filename}'. Ensure the face is clearly visible and well-lit.",
        )
    if len(faces) > 1:
        logger.warning(f"Multiple faces in '{filename}' — using largest.")
    return [_largest_face(faces) if len(faces) > 1 else faces[0]]


# ── Duplicate detection ───────────────────────────────────────────────────────

def _sha256(img: np.ndarray) -> str:
    return hashlib.sha256(img.tobytes()).hexdigest()


def _check_no_duplicates(selfie_images, selfie_filenames, selfie_embeddings):
    n = len(selfie_images)
    hashes = [_sha256(img) for img in selfie_images]
    for i in range(n):
        for j in range(i + 1, n):
            if hashes[i] == hashes[j]:
                raise HTTPException(
                    status_code=422,
                    detail=f"Duplicate image: '{selfie_filenames[i]}' and '{selfie_filenames[j]}' are identical.",
                )
            dist = cosine_distance(selfie_embeddings[i], selfie_embeddings[j])
            if dist < NEAR_DUPLICATE_THRESHOLD:
                raise HTTPException(
                    status_code=422,
                    detail=f"Near-duplicate: '{selfie_filenames[i]}' vs '{selfie_filenames[j]}' (dist={dist:.4f}).",
                )


def _check_selfie_consistency(selfie_embeddings, selfie_filenames):
    n = len(selfie_embeddings)
    failures = []
    for i in range(n):
        for j in range(i + 1, n):
            d = cosine_distance(selfie_embeddings[i], selfie_embeddings[j])
            if d > CROSS_SELFIE_THRESHOLD:
                failures.append((selfie_filenames[i], selfie_filenames[j], d))
    total_pairs = n * (n - 1) // 2
    if len(failures) > total_pairs // 2:
        worst = sorted(failures, key=lambda x: -x[2])[:3]
        detail = "; ".join(f"'{a}' vs '{b}' (dist={d:.3f})" for a, b, d in worst)
        raise HTTPException(
            status_code=422,
            detail=f"Cross-selfie identity mismatch — submitted selfies appear to be different people. {detail}",
        )


# ── Pose helpers ──────────────────────────────────────────────────────────────

def _pose(img_bgr: np.ndarray, filename: str):
    pose = pose_estimator.estimate_pose(img_bgr)
    if pose is None:
        logger.warning(f"No pose detected for '{filename}'")
        return None, None, None
    yaw, pitch, roll = pose["yaw"], pose["pitch"], pose["roll"]
    logger.info(f"[POSE] {filename} | yaw={yaw:.2f} pitch={pitch:.2f} roll={roll:.2f}")
    return yaw, pitch, roll


# ── Slot checkers ─────────────────────────────────────────────────────────────

def _check_left(face, img_bgr, filename: str) -> None:
    """Slot 1 — left turn. Checks: yaw + roll. Pitch ignored."""
    yaw, _, roll = _pose(img_bgr, filename)
    if yaw is None:
        return
    if roll is not None and abs(roll) > LEFT_ROLL_MAX:
        raise HTTPException(
            status_code=422,
            detail=f"[Slot 1 — left] '{filename}': Head tilted (roll={roll:.1f}°, max ±{LEFT_ROLL_MAX}°). Keep head upright.",
        )
    if LEFT_YAW_MIN <= yaw <= LEFT_YAW_MAX:
        return
    if yaw < LEFT_YAW_MIN:
        hint = f"Not turned enough (yaw={yaw:.1f}°). Turn further LEFT to reach {LEFT_YAW_MIN}°–{LEFT_YAW_MAX}°."
    else:
        hint = f"Turned too far (yaw={yaw:.1f}°, max={LEFT_YAW_MAX}°). Turn back slightly."
    raise HTTPException(status_code=422, detail=f"[Slot 1 — left] '{filename}': {hint}")


def _check_right(face, img_bgr, filename: str) -> None:
    """Slot 2 — right turn. Checks: yaw + roll. Pitch ignored."""
    yaw, _, roll = _pose(img_bgr, filename)
    if yaw is None:
        return
    if roll is not None and abs(roll) > RIGHT_ROLL_MAX:
        raise HTTPException(
            status_code=422,
            detail=f"[Slot 2 — right] '{filename}': Head tilted (roll={roll:.1f}°, max ±{RIGHT_ROLL_MAX}°). Keep head upright.",
        )
    if -RIGHT_YAW_MAX <= yaw <= -RIGHT_YAW_MIN:
        return
    if yaw > -RIGHT_YAW_MIN:
        hint = f"Not turned enough (yaw={yaw:.1f}°). Turn further RIGHT to reach -{RIGHT_YAW_MIN}° to -{RIGHT_YAW_MAX}°."
    else:
        hint = f"Turned too far (yaw={yaw:.1f}°, max=-{RIGHT_YAW_MAX}°). Turn back slightly."
    raise HTTPException(status_code=422, detail=f"[Slot 2 — right] '{filename}': {hint}")


def _check_frontal(face, img_bgr, filename: str, slot_label: str) -> None:
    """
    Front / smile pose gate. Checks: yaw + pitch + roll.
    Uses smile-specific thresholds when slot_label contains 'smile'.
    """
    yaw, pitch, roll = _pose(img_bgr, filename)
    if yaw is None:
        return
    is_smile = "smile" in slot_label.lower()
    yaw_max   = SMILE_YAW_MAX   if is_smile else FRONT_YAW_MAX
    pitch_max = SMILE_PITCH_MAX if is_smile else FRONT_PITCH_MAX
    roll_max  = SMILE_ROLL_MAX  if is_smile else FRONT_ROLL_MAX

    if roll is not None and abs(roll) > roll_max:
        raise HTTPException(
            status_code=422,
            detail=f"[{slot_label}] '{filename}': Head tilted (roll={roll:.1f}°, max ±{roll_max}°). Keep head upright.",
        )
    if abs(yaw) > yaw_max:
        raise HTTPException(
            status_code=422,
            detail=f"[{slot_label}] '{filename}': Not frontal (yaw={yaw:.1f}°, max ±{yaw_max}°).",
        )
    if abs(pitch) > pitch_max:
        raise HTTPException(
            status_code=422,
            detail=f"[{slot_label}] '{filename}': Head up/down (pitch={pitch:.1f}°, max ±{pitch_max}°).",
        )


def _check_straight(face, img_bgr, filename: str) -> None:
    _check_frontal(face, img_bgr, filename, "Slot 0 — straight")


# ── Smile detection ───────────────────────────────────────────────────────────

def _smile_via_68pt(face_lm, filename: str) -> bool:
    lm = getattr(face_lm, "landmark_3d_68", None)
    if lm is None or len(lm) < 68:
        lm = getattr(face_lm, "landmark_2d_106", None)
        if lm is None:
            return False
        try:
            upper = lm[86]; lower = lm[102]
            eye_y = (lm[33][1] + lm[87][1]) / 2.0
            chin_y = lm[16][1]
            face_h = abs(chin_y - eye_y) or 1.0
            ratio = abs(lower[1] - upper[1]) / face_h
            logger.info(f"'{filename}' smile (106pt) ratio={ratio:.4f}")
            return ratio >= SMILE_68PT_INNER_OPEN_RATIO
        except Exception:
            return False
    try:
        upper = lm[61]; lower = lm[67]
        eye_y = (lm[36][1] + lm[45][1]) / 2.0
        chin_y = lm[8][1]
        face_h = abs(chin_y - eye_y) or 1.0
        ratio = abs(lower[1] - upper[1]) / face_h
        logger.info(f"'{filename}' smile (68pt) ratio={ratio:.4f}")
        return ratio >= SMILE_68PT_INNER_OPEN_RATIO
    except Exception:
        return False


def _smile_via_5pt(face, filename: str) -> bool:
    lm = getattr(face, "landmark_2d_5", None)
    if lm is None or len(lm) < 5:
        return True
    try:
        eye_dist = np.linalg.norm(np.array(lm[1]) - np.array(lm[0]))
        mouth_width = np.linalg.norm(np.array(lm[4]) - np.array(lm[3]))
        ratio = mouth_width / (eye_dist or 1.0)
        logger.info(f"'{filename}' smile (5pt) ratio={ratio:.4f}")
        return ratio >= SMILE_5PT_MOUTH_WIDTH_RATIO
    except Exception:
        return True


def _check_smile(face_sc, img_bgr: np.ndarray, filename: str) -> None:
    _check_frontal(face_sc, img_bgr, filename, "Slot 3 — smile")
    if _HAS_LM_MODEL and face_app_lm is not None:
        try:
            lm_faces = face_app_lm.get(_bgr_to_rgb(img_bgr))
            if lm_faces:
                face_lm = _largest_face(lm_faces) if len(lm_faces) > 1 else lm_faces[0]
                if not _smile_via_68pt(face_lm, filename):
                    raise HTTPException(
                        status_code=422,
                        detail=f"[Slot 3 — smile] '{filename}': No smile detected.",
                    )
                return
        except HTTPException:
            raise
        except Exception as e:
            logger.warning(f"'{filename}' buffalo_l failed: {e}")
    if not _smile_via_5pt(face_sc, filename):
        raise HTTPException(status_code=422, detail=f"[Slot 3 — smile] '{filename}': No smile detected.")


# ── Slot registry ─────────────────────────────────────────────────────────────

SELFIE_SLOTS = {
    0: {"label": "straight / neutral selfie",
        "instruction": "Look straight into the camera with neutral expression",
        "pose_check": lambda f, i, n: _check_straight(f, i, n)},
    1: {"label": "left-turn selfie",
        "instruction": "Turn your head to YOUR LEFT",
        "pose_check": lambda f, i, n: _check_left(f, i, n)},
    2: {"label": "right-turn selfie",
        "instruction": "Turn your head to YOUR RIGHT",
        "pose_check": lambda f, i, n: _check_right(f, i, n)},
    3: {"label": "smile selfie",
        "instruction": "Look straight and smile naturally",
        "pose_check": _check_smile},
}
REQUIRED_SELFIE_COUNT = len(SELFIE_SLOTS)


# ── Embedding helpers ─────────────────────────────────────────────────────────

def get_face_embedding(img: np.ndarray, filename: str) -> np.ndarray:
    return _detect(img, filename)[0].embedding


def get_face_embedding_with_pose_check(img, filename, slot_index):
    if slot_index not in SELFIE_SLOTS:
        raise HTTPException(status_code=400, detail=f"Unknown slot {slot_index}.")
    slot = SELFIE_SLOTS[slot_index]
    faces = _detect(img, filename)
    face = faces[0]
    # Pose check is now a SOFT validation — log warnings, don't reject
    try:
        slot["pose_check"](face, img, filename)
        logger.info(f"[POSE-OK] '{filename}' passed pose check for slot {slot_index} ({slot['label']})")
    except HTTPException as e:
        # Log the pose issue but DON'T reject — let face matching decide
        logger.warning(f"[POSE-SOFT] '{filename}' pose issue (slot {slot_index}): {e.detail}")
    return face.embedding


# Minimum yaw range across all selfies to prove head movement (liveness)
# 25° means at least one selfie must have a noticeably different head angle
MIN_YAW_VARIATION = 25.0

def _check_yaw_variation(selfie_images, selfie_filenames):
    """
    Liveness check: ensure the user actually moved their head across selfies.
    Instead of requiring exact angles per slot, we check that the YAW values
    across all 4 selfies span at least MIN_YAW_VARIATION degrees.
    
    Example:
      selfie_0 yaw=2°, selfie_1 yaw=25°, selfie_2 yaw=-18°, selfie_3 yaw=1°
      range = 25 - (-18) = 43° → PASS (>15°)
    
      selfie_0 yaw=1°, selfie_1 yaw=3°, selfie_2 yaw=-2°, selfie_3 yaw=0°
      range = 3 - (-2) = 5° → FAIL (all basically front-facing)
    """
    yaw_values = []
    for img, fn in zip(selfie_images, selfie_filenames):
        yaw, _, _ = _pose(img, fn)
        if yaw is not None:
            yaw_values.append(yaw)
    
    if len(yaw_values) < 2:
        logger.warning("[LIVENESS] Could not extract enough yaw values for variation check")
        return  # Can't check, let it pass
    
    yaw_range = max(yaw_values) - min(yaw_values)
    logger.info(f"[LIVENESS] Yaw values: {[round(y, 1) for y in yaw_values]}, range={yaw_range:.1f}°")
    
    if yaw_range < MIN_YAW_VARIATION:
        raise HTTPException(
            status_code=422,
            detail=f"Liveness check failed: Your selfies all look the same direction. "
                   f"Please turn your head left and right as instructed.",
        )


def cosine_distance(a, b) -> float:
    a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 1.0
    return float(1.0 - np.dot(a, b) / (na * nb))


def distance_to_confidence(distance, threshold) -> float:
    return float(max(0.0, min(1.0, 1.0 - distance / (threshold * 2.0))))


# ── Verification pipeline ─────────────────────────────────────────────────────

def verify_faces(reference_images, selfie_images, reference_filenames, selfie_filenames):
    if len(selfie_images) != REQUIRED_SELFIE_COUNT:
        raise HTTPException(
            status_code=400,
            detail=f"Exactly {REQUIRED_SELFIE_COUNT} selfies required (got {len(selfie_images)}).",
        )

    ref_embeddings = [get_face_embedding(img, fn) for img, fn in zip(reference_images, reference_filenames)]

    selfie_embeddings = []
    for i, (img, fn) in enumerate(zip(selfie_images, selfie_filenames)):
        selfie_embeddings.append(get_face_embedding_with_pose_check(img, fn, slot_index=i))

    # Liveness: ensure user actually turned their head (not all same angle)
    _check_yaw_variation(selfie_images, selfie_filenames)

    _check_no_duplicates(selfie_images, selfie_filenames, selfie_embeddings)
    _check_selfie_consistency(selfie_embeddings, selfie_filenames)

    pair_results, matched = [], 0
    for ri, re in enumerate(ref_embeddings):
        for si, se in enumerate(selfie_embeddings):
            d = cosine_distance(re, se)
            is_match = d < MATCH_THRESHOLD
            conf = distance_to_confidence(d, MATCH_THRESHOLD)
            if is_match:
                matched += 1
            pair_results.append(PairResult(
                reference_index=ri, selfie_index=si, is_match=is_match,
                confidence_score=round(conf, 4), distance=round(d, 4),
            ))

    total = len(pair_results)
    ratio = matched / total if total else 0.0
    overall = round(
        (sum(p.confidence_score for p in pair_results if p.is_match) / matched) if matched
        else max((p.confidence_score for p in pair_results), default=0.0),
        4,
    )
    is_verified = ratio >= MIN_MATCH_RATIO
    status = VerificationStatus.VERIFIED if is_verified else VerificationStatus.REJECTED
    msg = (
        f"Identity {'verified' if is_verified else 'rejected'}. "
        f"{matched}/{total} pairs matched ({ratio*100:.1f}%)."
    )
    return VerificationResponse(
        status=status, confidence_score=overall, matched_pairs=matched,
        total_pairs=total, match_ratio=round(ratio, 4),
        pair_results=pair_results, message=msg,
    )