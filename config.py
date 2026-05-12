from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── Recognition ──────────────────────────────────────────────────────
    RECOGNITION_MODEL: str = "ArcFace"
    DISTANCE_METRIC: str = "cosine"

    # ── Verification Thresholds ─────────────────────────────────────────
    MATCH_THRESHOLD: float = 0.55
    MIN_MATCH_RATIO: float = 0.50
    
    # ── SLOT 0 — FRONT ──────────────────────────────────────
    # User looks straight at camera, neutral expression
    FRONT_YAW_MAX: float   = 12.0  # ±° horizontal turn allowed
    FRONT_PITCH_MAX: float = 30.0  # ±° up/down tilt allowed
    FRONT_ROLL_MAX: float  = 15.0  # ±° head-tilt (ear toward shoulder)

    # ── SLOT 1 — LEFT ───────────────────────────────────────
    # User turns head to THEIR left (yaw goes positive)
    LEFT_YAW_MIN: float  = 20.0  # must turn at least this far
    LEFT_YAW_MAX: float  = 60.0  # don't go so far face detector loses face
    LEFT_ROLL_MAX: float = 18.0  # ±° head-tilt tolerance

    # ── SLOT 2 — RIGHT ──────────────────────────────────────
    # User turns head to THEIR right (yaw goes negative)
    RIGHT_YAW_MIN: float  = 20.0  # must turn at least this far (absolute value)
    RIGHT_YAW_MAX: float  = 60.0  # don't go so far face detector loses face
    RIGHT_ROLL_MAX: float = 18.0  # ±° head-tilt tolerance

    # ── SLOT 3 — SMILE ──────────────────────────────────────
    SMILE_YAW_MAX: float   = 12.0   # ±° horizontal turn allowed
    SMILE_PITCH_MAX: float = 30.0   # ±° up/down tilt allowed
    SMILE_ROLL_MAX: float  = 18.0   # ±° head-tilt tolerance

    # ── Image Constraints ───────────────────────────────────────────────
    MAX_FILE_SIZE_MB: int = 10
    ALLOWED_EXTENSIONS: list[str] = ["jpg", "jpeg", "png"]
    MIN_REFERENCE_IMAGES: int = 1
    MAX_REFERENCE_IMAGES: int = 2
    MIN_SELFIE_IMAGES: int = 4
    MAX_SELFIE_IMAGES: int = 4

    # Order: 0=front, 1=left, 2=right, 3=smile
    SELFIE_ORDER: list[str] = ["front", "left", "right", "smile"]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()