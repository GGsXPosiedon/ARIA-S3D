"""
ARIA-S3D
PHASE 4.6 - TRACKING-LOSS RECOVERY

Purpose:
    Detect tracking failure during real-time monocular reconstruction
    and attempt recovery using the last reliable frame/keyframe.

Pipeline:
    Live video
        ↓
    SIFT feature detection
        ↓
    Feature matching
        ↓
    Tracking quality evaluation
        ↓
    Tracking-loss detection
        ↓
    Keyframe-based recovery
        ↓
    Recovered / Lost state
        ↓
    Statistics + trajectory output

Controls:
    Q / ESC : Stop
    R       : Force recovery
    K       : Save current frame as recovery keyframe
"""

import os
import cv2
import time
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

VIDEO_FILE = os.path.join(
    BASE_DIR,
    "data",
    "raw_video",
    "video.mp4"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "output"
)

TRAJECTORY_FILE = os.path.join(
    OUTPUT_DIR,
    "realtime_recovery_trajectory.txt"
)

STATS_FILE = os.path.join(
    OUTPUT_DIR,
    "tracking_recovery_stats.txt"
)

WINDOW_NAME = "ARIA-S3D | Tracking-Loss Recovery"


# ============================================================
# PARAMETERS
# ============================================================

SIFT_FEATURES = 2500

MIN_GOOD_MATCHES = 25
RECOVERY_MATCHES = 20

RATIO_TEST = 0.75

MIN_TRACKING_INLIERS = 15

LOSS_THRESHOLD_FRAMES = 3

MAX_RECOVERY_ATTEMPTS = 5

RECOVERY_DISTANCE_THRESHOLD = 1.5


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def print_header(title):
    print("=" * 70)
    print(title)
    print("=" * 70)


def ensure_output_directory():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_video():
    if not os.path.exists(VIDEO_FILE):
        print("[ERROR] Video file not found:")
        print(VIDEO_FILE)
        return None

    cap = cv2.VideoCapture(VIDEO_FILE)

    if not cap.isOpened():
        print("[ERROR] Could not open video.")
        return None

    return cap


def create_sift():
    return cv2.SIFT_create(
        nfeatures=SIFT_FEATURES
    )


def create_matcher():
    return cv2.BFMatcher(
        cv2.NORM_L2,
        crossCheck=False
    )


# ============================================================
# FEATURE DETECTION
# ============================================================

def detect_features(sift, frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    keypoints, descriptors = sift.detectAndCompute(
        gray,
        None
    )

    if descriptors is None:
        return [], None

    return keypoints, descriptors


# ============================================================
# FEATURE MATCHING
# ============================================================

def match_features(
    matcher,
    descriptors1,
    descriptors2
):
    if descriptors1 is None or descriptors2 is None:
        return []

    if len(descriptors1) < 2 or len(descriptors2) < 2:
        return []

    try:
        knn_matches = matcher.knnMatch(
            descriptors1,
            descriptors2,
            k=2
        )
    except cv2.error:
        return []

    good_matches = []

    for pair in knn_matches:

        if len(pair) != 2:
            continue

        m, n = pair

        if m.distance < RATIO_TEST * n.distance:
            good_matches.append(m)

    return good_matches


# ============================================================
# HOMOGRAPHY / TRACKING QUALITY
# ============================================================

def calculate_inliers(
    keypoints1,
    keypoints2,
    matches
):

    if len(matches) < 4:
        return 0, None

    points1 = np.float32(
        [keypoints1[m.queryIdx].pt for m in matches]
    ).reshape(-1, 1, 2)

    points2 = np.float32(
        [keypoints2[m.trainIdx].pt for m in matches]
    ).reshape(-1, 1, 2)

    try:
        H, mask = cv2.findHomography(
            points1,
            points2,
            cv2.RANSAC,
            5.0
        )
    except cv2.error:
        return 0, None

    if mask is None:
        return 0, H

    inliers = int(np.sum(mask))

    return inliers, H


# ============================================================
# CAMERA MOTION ESTIMATION
# ============================================================

def estimate_motion(
    keypoints1,
    keypoints2,
    matches
):

    if len(matches) < 5:
        return np.zeros(3, dtype=np.float64)

    pts1 = np.float32(
        [keypoints1[m.queryIdx].pt for m in matches]
    )

    pts2 = np.float32(
        [keypoints2[m.trainIdx].pt for m in matches]
    )

    displacement = pts2 - pts1

    dx = float(np.mean(displacement[:, 0]))
    dy = float(np.mean(displacement[:, 1]))

    magnitude = float(
        np.mean(
            np.linalg.norm(
                displacement,
                axis=1
            )
        )
    )

    return np.array(
        [dx, dy, magnitude],
        dtype=np.float64
    )


# ============================================================
# TRACKING STATE
# ============================================================

class TrackingState:

    def __init__(self):

        self.position = np.zeros(
            3,
            dtype=np.float64
        )

        self.last_position = np.zeros(
            3,
            dtype=np.float64
        )

        self.trajectory = []

        self.previous_frame = None
        self.previous_keypoints = None
        self.previous_descriptors = None

        self.keyframe = None
        self.keyframe_keypoints = None
        self.keyframe_descriptors = None

        self.frame_index = 0

        self.tracking_lost = False

        self.consecutive_failures = 0

        self.recovery_attempts = 0

        self.total_recoveries = 0

        self.successful_frames = 0

        self.failed_frames = 0

        self.total_matches = 0

        self.total_inliers = 0

        self.recovery_successes = 0

        self.recovery_failures = 0

        self.motion_history = []


# ============================================================
# SAVE KEYFRAME
# ============================================================

def save_keyframe(
    state,
    frame,
    keypoints,
    descriptors
):

    state.keyframe = frame.copy()

    state.keyframe_keypoints = keypoints

    state.keyframe_descriptors = descriptors

    print(
        f"[KEYFRAME] Recovery keyframe saved "
        f"at frame {state.frame_index}"
    )


# ============================================================
# RECOVERY
# ============================================================

def attempt_recovery(
    state,
    frame,
    sift,
    matcher
):

    if state.keyframe is None:
        return False, 0, 0

    if state.keyframe_descriptors is None:
        return False, 0, 0

    keypoints, descriptors = detect_features(
        sift,
        frame
    )

    if descriptors is None:
        return False, 0, 0

    matches = match_features(
        matcher,
        state.keyframe_descriptors,
        descriptors
    )

    match_count = len(matches)

    if match_count < RECOVERY_MATCHES:
        return False, match_count, 0

    inliers, H = calculate_inliers(
        state.keyframe_keypoints,
        keypoints,
        matches
    )

    if inliers < MIN_TRACKING_INLIERS:
        return False, match_count, inliers

    # --------------------------------------------------------
    # Recovery successful
    # --------------------------------------------------------

    motion = estimate_motion(
        state.keyframe_keypoints,
        keypoints,
        matches
    )

    state.position[0] += motion[0] * 0.005
    state.position[1] += motion[1] * 0.005
    state.position[2] += motion[2] * 0.01

    state.last_position = state.position.copy()

    state.consecutive_failures = 0
    state.tracking_lost = False

    state.total_recoveries += 1
    state.recovery_successes += 1

    state.trajectory.append(
        state.position.copy()
    )

    state.motion_history.append(
        motion
    )

    return True, match_count, inliers


# ============================================================
# NORMAL TRACKING
# ============================================================

def track_frame(
    state,
    frame,
    sift,
    matcher
):

    keypoints, descriptors = detect_features(
        sift,
        frame
    )

    if (
        state.previous_descriptors is None
        or descriptors is None
    ):
        return False, len(keypoints), 0, np.zeros(3)

    matches = match_features(
        matcher,
        state.previous_descriptors,
        descriptors
    )

    match_count = len(matches)

    if match_count < MIN_GOOD_MATCHES:
        return False, len(keypoints), match_count, np.zeros(3)

    inliers, H = calculate_inliers(
        state.previous_keypoints,
        keypoints,
        matches
    )

    if inliers < MIN_TRACKING_INLIERS:
        return False, len(keypoints), match_count, np.zeros(3)

    motion = estimate_motion(
        state.previous_keypoints,
        keypoints,
        matches
    )

    # --------------------------------------------------------
    # Monocular incremental motion
    # Scale remains arbitrary.
    # --------------------------------------------------------

    state.position[0] += motion[0] * 0.005
    state.position[1] += motion[1] * 0.005
    state.position[2] += motion[2] * 0.01

    state.last_position = state.position.copy()

    state.trajectory.append(
        state.position.copy()
    )

    state.motion_history.append(
        motion
    )

    return True, len(keypoints), match_count, motion


# ============================================================
# VISUALIZATION
# ============================================================

def draw_status(
    frame,
    state,
    keypoints_count,
    matches,
    inliers,
    fps
):

    output = frame.copy()

    if state.tracking_lost:

        status = "TRACKING LOST - RECOVERING"

    else:

        status = "TRACKING OK"

    cv2.putText(
        output,
        f"ARIA-S3D | {status}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0) if not state.tracking_lost else (0, 0, 255),
        2
    )

    cv2.putText(
        output,
        f"Frame: {state.frame_index}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1
    )

    cv2.putText(
        output,
        f"Features: {keypoints_count}",
        (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1
    )

    cv2.putText(
        output,
        f"Matches: {matches}",
        (20, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1
    )

    cv2.putText(
        output,
        f"Inliers: {inliers}",
        (20, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1
    )

    cv2.putText(
        output,
        f"Recoveries: {state.total_recoveries}",
        (20, 190),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1
    )

    cv2.putText(
        output,
        f"FPS: {fps:.2f}",
        (20, 220),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1
    )

    return output


# ============================================================
# SAVE TRAJECTORY
# ============================================================

def save_trajectory(state):

    with open(
        TRAJECTORY_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "# frame x y z state\n"
        )

        for i, position in enumerate(
            state.trajectory
        ):

            state_name = "TRACKED"

            f.write(
                f"{i} "
                f"{position[0]:.8f} "
                f"{position[1]:.8f} "
                f"{position[2]:.8f} "
                f"{state_name}\n"
            )


# ============================================================
# SAVE STATISTICS
# ============================================================

def save_statistics(
    state,
    fps_values
):

    total_frames = (
        state.successful_frames
        + state.failed_frames
    )

    if total_frames > 0:

        tracking_rate = (
            state.successful_frames
            / total_frames
            * 100.0
        )

    else:

        tracking_rate = 0.0

    if len(fps_values) > 0:

        average_fps = float(
            np.mean(fps_values)
        )

    else:

        average_fps = 0.0

    if state.total_matches > 0:

        average_matches = (
            state.total_matches
            / max(
                1,
                state.successful_frames
                + state.failed_frames
            )
        )

    else:

        average_matches = 0.0

    if state.total_inliers > 0:

        average_inliers = (
            state.total_inliers
            / max(
                1,
                state.successful_frames
            )
        )

    else:

        average_inliers = 0.0

    with open(
        STATS_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "ARIA-S3D | PHASE 4.6 "
            "TRACKING-LOSS RECOVERY\n"
        )

        f.write(
            "========================================\n"
        )

        f.write(
            f"Frames processed        : {total_frames}\n"
        )

        f.write(
            f"Successful frames       : "
            f"{state.successful_frames}\n"
        )

        f.write(
            f"Failed frames           : "
            f"{state.failed_frames}\n"
        )

        f.write(
            f"Tracking success rate   : "
            f"{tracking_rate:.2f}%\n"
        )

        f.write(
            f"Total recoveries        : "
            f"{state.total_recoveries}\n"
        )

        f.write(
            f"Recovery successes      : "
            f"{state.recovery_successes}\n"
        )

        f.write(
            f"Recovery failures       : "
            f"{state.recovery_failures}\n"
        )

        f.write(
            f"Average matches         : "
            f"{average_matches:.2f}\n"
        )

        f.write(
            f"Average inliers         : "
            f"{average_inliers:.2f}\n"
        )

        f.write(
            f"Average FPS             : "
            f"{average_fps:.2f}\n"
        )

        f.write(
            f"Maximum recovery attempts: "
            f"{state.recovery_attempts}\n"
        )

        f.write(
            "\n"
        )

        f.write(
            "Recovery mechanism:\n"
        )

        f.write(
            "Feature tracking failure is detected "
            "using match and RANSAC-inlier thresholds.\n"
        )

        f.write(
            "A stored keyframe is used to recover "
            "tracking after temporary loss.\n"
        )

        f.write(
            "\n"
        )

        f.write(
            "IMPORTANT:\n"
        )

        f.write(
            "This is a monocular reconstruction.\n"
        )

        f.write(
            "Absolute metric scale remains arbitrary.\n"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print_header(
        "ARIA-S3D | PHASE 4.6\n"
        "TRACKING-LOSS RECOVERY"
    )

    ensure_output_directory()

    # --------------------------------------------------------
    # Video
    # --------------------------------------------------------

    cap = load_video()

    if cap is None:
        return

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    fps_video = cap.get(
        cv2.CAP_PROP_FPS
    )

    if fps_video <= 0:
        fps_video = 30.0

    print()
    print(f"Video resolution : {width} x {height}")
    print(f"Video FPS        : {fps_video:.2f}")
    print(f"SIFT features    : {SIFT_FEATURES}")
    print(f"Minimum matches  : {MIN_GOOD_MATCHES}")
    print(f"Recovery matches : {RECOVERY_MATCHES}")

    print()
    print("Controls:")
    print("Q / ESC : Stop")
    print("R       : Force recovery")
    print("K       : Save current frame as keyframe")

    print()
    print("-" * 70)
    print("TRACKING-LOSS RECOVERY STARTED")
    print("-" * 70)

    # --------------------------------------------------------
    # Initialize
    # --------------------------------------------------------

    sift = create_sift()

    matcher = create_matcher()

    state = TrackingState()

    fps_values = []

    previous_time = time.time()

    # --------------------------------------------------------
    # Read first frame
    # --------------------------------------------------------

    ret, first_frame = cap.read()

    if not ret:

        print("[ERROR] Could not read first frame.")

        cap.release()

        return

    kp, des = detect_features(
        sift,
        first_frame
    )

    state.previous_frame = first_frame

    state.previous_keypoints = kp

    state.previous_descriptors = des

    save_keyframe(
        state,
        first_frame,
        kp,
        des
    )

    state.trajectory.append(
        state.position.copy()
    )

    # --------------------------------------------------------
    # Main loop
    # --------------------------------------------------------

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        state.frame_index += 1

        start_time = time.time()

        # ----------------------------------------------------
        # Normal tracking
        # ----------------------------------------------------

        success, keypoint_count, match_count, motion = track_frame(
            state,
            frame,
            sift,
            matcher
        )

        inliers = 0

        if success:

            state.successful_frames += 1

            state.consecutive_failures = 0

            state.tracking_lost = False

            state.total_matches += match_count

            # Approximate inlier estimate from match count.
            inliers = max(
                MIN_TRACKING_INLIERS,
                int(match_count * 0.6)
            )

            state.total_inliers += inliers

            # ------------------------------------------------
            # Periodically refresh recovery keyframe
            # ------------------------------------------------

            if (
                state.frame_index % 50 == 0
                and keypoint_count > 0
            ):

                kp, des = detect_features(
                    sift,
                    frame
                )

                if des is not None:

                    save_keyframe(
                        state,
                        frame,
                        kp,
                        des
                    )

        else:

            state.failed_frames += 1

            state.consecutive_failures += 1

            state.tracking_lost = True

            # ------------------------------------------------
            # Tracking loss detected
            # ------------------------------------------------

            if (
                state.consecutive_failures
                >= LOSS_THRESHOLD_FRAMES
            ):

                state.recovery_attempts += 1

                recovered, recovery_matches, recovery_inliers = attempt_recovery(
                    state,
                    frame,
                    sift,
                    matcher
                )

                match_count = recovery_matches

                inliers = recovery_inliers

                if recovered:

                    print(
                        f"[RECOVERED] "
                        f"Frame {state.frame_index} | "
                        f"Matches: {recovery_matches} | "
                        f"Inliers: {recovery_inliers}"
                    )

                    # Reset recovery attempt counter
                    state.recovery_attempts = 0

                else:

                    if (
                        state.recovery_attempts
                        >= MAX_RECOVERY_ATTEMPTS
                    ):

                        state.recovery_failures += 1

                        print(
                            f"[WARNING] Recovery failed "
                            f"at frame {state.frame_index}"
                        )

        # ----------------------------------------------------
        # Update previous frame only when useful
        # ----------------------------------------------------

        if success:

            kp, des = detect_features(
                sift,
                frame
            )

            if des is not None:

                state.previous_frame = frame.copy()

                state.previous_keypoints = kp

                state.previous_descriptors = des

        elif not state.tracking_lost:

            kp, des = detect_features(
                sift,
                frame
            )

            if des is not None:

                state.previous_frame = frame.copy()

                state.previous_keypoints = kp

                state.previous_descriptors = des

        # ----------------------------------------------------
        # FPS
        # ----------------------------------------------------

        current_time = time.time()

        elapsed = current_time - previous_time

        if elapsed > 0:

            current_fps = 1.0 / elapsed

        else:

            current_fps = 0.0

        previous_time = current_time

        fps_values.append(
            current_fps
        )

        # ----------------------------------------------------
        # Draw visualization
        # ----------------------------------------------------

        display = draw_status(
            frame,
            state,
            keypoint_count,
            match_count,
            inliers,
            current_fps
        )

        cv2.imshow(
            WINDOW_NAME,
            display
        )

        # ----------------------------------------------------
        # Keyboard controls
        # ----------------------------------------------------

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:

            break

        elif key == ord("r"):

            print(
                "[MANUAL] Recovery requested."
            )

            state.tracking_lost = True

            state.consecutive_failures = (
                LOSS_THRESHOLD_FRAMES
            )

        elif key == ord("k"):

            kp, des = detect_features(
                sift,
                frame
            )

            if des is not None:

                save_keyframe(
                    state,
                    frame,
                    kp,
                    des
                )

    # ========================================================
    # CLEANUP
    # ========================================================

    cap.release()

    cv2.destroyAllWindows()

    # ========================================================
    # SAVE OUTPUTS
    # ========================================================

    save_trajectory(state)

    save_statistics(
        state,
        fps_values
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    total_frames = (
        state.successful_frames
        + state.failed_frames
    )

    if total_frames > 0:

        tracking_rate = (
            state.successful_frames
            / total_frames
            * 100.0
        )

    else:

        tracking_rate = 0.0

    average_fps = (
        float(np.mean(fps_values))
        if fps_values
        else 0.0
    )

    print()
    print("=" * 70)
    print("ARIA-S3D | PHASE 4.6 COMPLETE")
    print("=" * 70)

    print(
        f"Frames processed        : {total_frames}"
    )

    print(
        f"Successful frames       : "
        f"{state.successful_frames}"
    )

    print(
        f"Failed frames           : "
        f"{state.failed_frames}"
    )

    print(
        f"Tracking success rate   : "
        f"{tracking_rate:.2f}%"
    )

    print(
        f"Total recoveries        : "
        f"{state.total_recoveries}"
    )

    print(
        f"Recovery successes      : "
        f"{state.recovery_successes}"
    )

    print(
        f"Recovery failures       : "
        f"{state.recovery_failures}"
    )

    print(
        f"Average FPS             : "
        f"{average_fps:.2f}"
    )

    print()

    print("Generated outputs:")

    print(
        f"  {TRAJECTORY_FILE}"
    )

    print(
        f"  {STATS_FILE}"
    )

    print()
    print("NEXT STEP:")
    print("Phase 4.7 - Keyframe management")

    print()
    print("IMPORTANT:")
    print(
        "This remains a monocular reconstruction."
    )

    print(
        "Absolute metric scale remains arbitrary."
    )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()