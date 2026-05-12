import cv2
import numpy as np
import mediapipe as mp


mp_face_mesh = mp.solutions.face_mesh


class HeadPoseEstimator:
    """
    Real head-pose estimation using MediaPipe FaceMesh + OpenCV solvePnP.

    Sign conventions (after correction):
        yaw   > 0  →  user turned head to THEIR LEFT  (left ear toward camera)
        yaw   < 0  →  user turned head to THEIR RIGHT (right ear toward camera)
        pitch > 0  →  looking UP
        pitch < 0  →  looking DOWN
        roll  > 0  →  head tilted to user's RIGHT shoulder
        roll  < 0  →  head tilted to user's LEFT shoulder
    """

    def __init__(self, static_image_mode: bool = True):

        # static_image_mode=True is correct for API: each call is an independent image,
        # NOT a tracked video stream. Using False causes flicker / dropped detections
        # because FaceMesh assumes temporal continuity.
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        )

    def estimate_pose(self, image):

        if image is None or image.size == 0:
            return None

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = self.face_mesh.process(image_rgb)

        if not results.multi_face_landmarks:
            return None

        face_landmarks = results.multi_face_landmarks[0]

        img_h, img_w, _ = image.shape

        # ── 2D image points ──────────────────────────────────────────────
        # 33  = left eye outer
        # 263 = right eye outer
        # 1   = nose tip
        # 61  = mouth left corner
        # 291 = mouth right corner
        # 199 = chin
        landmark_ids = [33, 263, 1, 61, 291, 199]
        face_2d = []

        for idx in landmark_ids:
            lm = face_landmarks.landmark[idx]
            face_2d.append([lm.x * img_w, lm.y * img_h])

        face_2d = np.array(face_2d, dtype=np.float64)

        # ── Canonical 3D model points (mm-ish, generic face) ─────────────
        face_3d = np.array([
            [-30.0,   0.0, -30.0],   # left eye
            [ 30.0,   0.0, -30.0],   # right eye
            [  0.0,   0.0,   0.0],   # nose tip
            [-25.0, -30.0, -20.0],   # left mouth
            [ 25.0, -30.0, -20.0],   # right mouth
            [  0.0, -65.0,  -5.0],   # chin
        ], dtype=np.float64)

        # ── Camera intrinsics ────────────────────────────────────────────
        focal_length = img_w
        cam_matrix = np.array([
            [focal_length, 0,            img_w / 2],
            [0,            focal_length, img_h / 2],
            [0,            0,            1]
        ], dtype=np.float64)
        dist_matrix = np.zeros((4, 1), dtype=np.float64)

        # ── solvePnP ─────────────────────────────────────────────────────
        success, rot_vec, _ = cv2.solvePnP(
            face_3d,
            face_2d,
            cam_matrix,
            dist_matrix,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return None

        rmat, _ = cv2.Rodrigues(rot_vec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

        # ── SIGN CORRECTION ──────────────────────────────────────────────
        # RQDecomp3x3 returns Euler angles in degrees as (pitch, yaw, roll)
        # but with sign conventions that don't match human intuition.
        # After empirical testing with mirrored (selfie) webcam input:
        #
        #   raw angles[0] → pitch (up = +, down = -)  ✓ keep
        #   raw angles[1] → yaw   (turn left = +, turn right = -)  ✓ keep
        #   raw angles[2] → roll  (tilt right = +, tilt left = -)  ✓ keep
        #
        # Previous code negated yaw which broke the left/right slot logic.
        pitch = float(angles[0])
        yaw   = float(angles[1])
        roll  = float(angles[2])

        # ── Normalize wrapped angles ─────────────────────────────

        def normalize_angle(angle):
            while angle > 90:
                angle -= 180
            while angle < -90:
                angle += 180
            return angle

        pitch = normalize_angle(pitch)
        roll  = normalize_angle(roll)

        return {
            "yaw":   round(yaw, 2),
            "pitch": round(pitch, 2),
            "roll":  round(roll, 2),
        }