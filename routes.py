from fastapi import APIRouter, File, UploadFile, HTTPException
from typing import Annotated
import logging

from config import settings
from schemas import VerificationResponse, ErrorResponse
from service import decode_image, verify_faces, estimate_pose_only

logger = logging.getLogger(__name__)
router = APIRouter()


# ── MAIN MULTI-SELFIE VERIFICATION ───────────────────────────────────────

@router.post(
    "/verify",
    response_model=VerificationResponse,
    responses={
        400: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    summary="Verify identity using multi-angle selfies",
)
async def verify_identity(
    reference_images: Annotated[list[UploadFile], File(description="1–2 reference photos")],
    selfie_images:    Annotated[list[UploadFile], File(description="Exactly 4 selfies: front, left, right, smile")],
):
    if not (settings.MIN_REFERENCE_IMAGES <= len(reference_images) <= settings.MAX_REFERENCE_IMAGES):
        raise HTTPException(
            status_code=400,
            detail=f"Provide between {settings.MIN_REFERENCE_IMAGES} and {settings.MAX_REFERENCE_IMAGES} reference images.",
        )
    if len(selfie_images) != settings.MAX_SELFIE_IMAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Exactly {settings.MAX_SELFIE_IMAGES} selfie images required.",
        )

    ref_arrays, ref_filenames = [], []
    for upload in reference_images:
        raw = await upload.read()
        ref_arrays.append(decode_image(raw, upload.filename or "reference.jpg"))
        ref_filenames.append(upload.filename or "reference.jpg")

    selfie_arrays, selfie_filenames = [], []
    for upload in selfie_images:
        raw = await upload.read()
        selfie_arrays.append(decode_image(raw, upload.filename or "selfie.jpg"))
        selfie_filenames.append(upload.filename or "selfie.jpg")

    result = verify_faces(
        reference_images=ref_arrays,
        selfie_images=selfie_arrays,
        reference_filenames=ref_filenames,
        selfie_filenames=selfie_filenames,
    )
    logger.info(f"Result: {result.status} | conf={result.confidence_score} | ratio={result.match_ratio}")
    return result


# ── NEW: LIVE POSE ENDPOINT ──────────────────────────────────────────────
# Frontend calls this every ~500 ms with one camera frame, gets back the
# real yaw/pitch/roll plus per-slot validity flags. No face detection
# (InsightFace) is run here — MediaPipe FaceMesh only — so it stays fast.

@router.post(
    "/pose",
    summary="Real-time head-pose estimation",
    description=(
        "Send a single image frame; receive yaw / pitch / roll in degrees "
        "plus boolean validity flags for each selfie slot (front/left/right/smile). "
        "Designed to be called repeatedly from a webcam preview."
    ),
)
async def get_pose(
    image: Annotated[UploadFile, File(description="Single frame (JPEG/PNG)")],
):
    raw = await image.read()
    img = decode_image(raw, image.filename or "frame.jpg")
    pose = estimate_pose_only(img)
    return pose


# ── QUICK 1-vs-1 (unchanged) ─────────────────────────────────────────────

@router.post(
    "/verify/quick",
    response_model=VerificationResponse,
    summary="Quick 1-vs-1 face comparison",
)
async def verify_quick(
    reference_image: Annotated[UploadFile, File(description="Single reference photo")],
    selfie_image:    Annotated[UploadFile, File(description="Single selfie photo")],
):
    ref_raw = await reference_image.read()
    ref_img = decode_image(ref_raw, reference_image.filename or "reference.jpg")
    selfie_raw = await selfie_image.read()
    selfie_img = decode_image(selfie_raw, selfie_image.filename or "selfie.jpg")
    return verify_faces(
        reference_images=[ref_img],
        selfie_images=[selfie_img, selfie_img, selfie_img, selfie_img],
        reference_filenames=[reference_image.filename or "reference.jpg"],
        selfie_filenames=["front.jpg", "left.jpg", "right.jpg", "smile.jpg"],
    )