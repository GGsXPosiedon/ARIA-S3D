"""
ARIA-S3D
PHASE 4.7 - KEYFRAME MANAGEMENT

Purpose:
    Manage keyframes during real-time monocular reconstruction.

Features:
    - Live video input
    - SIFT feature extraction
    - Keyframe selection based on:
        1. Frame interval
        2. Feature overlap
        3. Scene change
        4. Camera motion proxy
    - Automatic keyframe insertion
    - Keyframe quality scoring
    - Keyframe metadata storage
    - Keyframe image storage
    - Keyframe statistics

Controls:
    Q / ESC : Stop
    R       : Force new keyframe
    K       : Save current frame as keyframe
"""

import os
import cv2
import time
import shutil
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

KEYFRAME_DIR = os.path.join(
    OUTPUT_DIR,
    "keyframes"
)

KEYFRAME_METADATA = os.path.join(
    OUTPUT_DIR,
    "keyframe_metadata.txt"
)

KEYFRAME_STATS = os.path.join(
    OUTPUT_DIR,
    "keyframe_management_stats.txt"
)


# ============================================================
# PARAMETERS
# ============================================================

SIFT_FEATURES = 2500

# Minimum number of good matches before comparing frames
MIN_GOOD_MATCHES = 25

# Minimum number of frames between automatic keyframes
MIN_KEYFRAME_INTERVAL = 20

# Maximum number of frames before forcing a keyframe
MAX_KEYFRAME_INTERVAL = 60

# Feature overlap threshold
# Lower overlap means scene has changed significantly
MIN_FEATURE_OVERLAP = 0.35

# Motion threshold
MOTION_THRESHOLD = 0.08

# Minimum keyframe quality
MIN_KEYFRAME_QUALITY = 20


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def print_header(title):
    print("=" * 70)
    print(title)
    print("=" * 70)


def ensure_directories():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Start fresh for this run
    if os.path.exists(KEYFRAME_DIR):
        shutil.rmtree(KEYFRAME_DIR)

    os.makedirs(KEYFRAME_DIR, exist_ok=True)


def load_video():
    cap = cv2.VideoCapture(VIDEO_FILE)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video:\n{VIDEO_FILE}"
        )

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30.0

    return cap, width, height, fps


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
# FEATURE EXTRACTION
# ============================================================

def detect_features(sift, frame):

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    keypoints, descriptors = sift.detectAndCompute(
        gray,
        None
    )

    return keypoints, descriptors


# ============================================================
# FEATURE MATCHING
# ============================================================

def calculate_feature_overlap(
    matcher,
    descriptors_current,
    descriptors_keyframe
):

    if descriptors_current is None:
        return 0, 0.0

    if descriptors_keyframe is None:
        return 0, 0.0

    if len(descriptors_current) < 2:
        return 0, 0.0

    if len(descriptors_keyframe) < 2:
        return 0, 0.0

    try:

        matches = matcher.knnMatch(
            descriptors_current,
            descriptors_keyframe,
            k=2
        )

    except cv2.error:
        return 0, 0.0

    good_matches = []

    for pair in matches:

        if len(pair) < 2:
            continue

        m, n = pair

        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    good_count = len(good_matches)

    current_count = len(descriptors_current)

    keyframe_count = len(descriptors_keyframe)

    denominator = max(
        1,
        min(
            current_count,
            keyframe_count
        )
    )

    overlap = good_count / denominator

    overlap = min(
        1.0,
        overlap
    )

    return good_count, overlap


# ============================================================
# FRAME DIFFERENCE
# ============================================================

def calculate_scene_change(
    previous_frame,
    current_frame
):

    if previous_frame is None:
        return 1.0

    previous_gray = cv2.cvtColor(
        previous_frame,
        cv2.COLOR_BGR2GRAY
    )

    current_gray = cv2.cvtColor(
        current_frame,
        cv2.COLOR_BGR2GRAY
    )

    previous_gray = cv2.resize(
        previous_gray,
        (320, 180)
    )

    current_gray = cv2.resize(
        current_gray,
        (320, 180)
    )

    difference = cv2.absdiff(
        previous_gray,
        current_gray
    )

    score = np.mean(difference) / 255.0

    return float(score)


# ============================================================
# KEYFRAME QUALITY
# ============================================================

def calculate_keyframe_quality(
    keypoints,
    descriptors
):

    if keypoints is None:
        return 0.0

    feature_count = len(keypoints)

    if descriptors is None:
        return float(feature_count)

    descriptor_quality = min(
        1.0,
        len(descriptors) / float(SIFT_FEATURES)
    )

    quality = (
        feature_count *
        (0.5 + 0.5 * descriptor_quality)
    )

    return float(quality)


# ============================================================
# KEYFRAME MANAGER
# ============================================================

class KeyframeManager:

    def __init__(self):

        self.keyframes = []

        self.total_keyframes = 0

        self.automatic_keyframes = 0

        self.manual_keyframes = 0

        self.forced_keyframes = 0

        self.last_keyframe_frame = -1

        self.last_keyframe_image = None

        self.last_keyframe_keypoints = None

        self.last_keyframe_descriptors = None

        self.keyframe_intervals = []

        self.good_match_history = []

        self.overlap_history = []

        self.quality_history = []

    # --------------------------------------------------------
    # SHOULD CREATE KEYFRAME
    # --------------------------------------------------------

    def should_create_keyframe(
        self,
        frame_id,
        good_matches,
        feature_overlap,
        scene_change,
        quality
    ):

        if self.last_keyframe_frame < 0:
            return True, "initial"

        frame_gap = (
            frame_id -
            self.last_keyframe_frame
        )

        # ----------------------------------------------------
        # FORCE KEYFRAME AFTER LARGE INTERVAL
        # ----------------------------------------------------

        if frame_gap >= MAX_KEYFRAME_INTERVAL:
            return True, "maximum_interval"

        # Do not create too many keyframes
        if frame_gap < MIN_KEYFRAME_INTERVAL:
            return False, "minimum_interval"

        # ----------------------------------------------------
        # LOW FEATURE OVERLAP
        # ----------------------------------------------------

        if (
            good_matches >= MIN_GOOD_MATCHES
            and
            feature_overlap < MIN_FEATURE_OVERLAP
        ):
            return True, "low_feature_overlap"

        # ----------------------------------------------------
        # SCENE CHANGE
        # ----------------------------------------------------

        if scene_change > MOTION_THRESHOLD:
            return True, "scene_change"

        # ----------------------------------------------------
        # LOW KEYFRAME QUALITY
        # ----------------------------------------------------

        if quality < MIN_KEYFRAME_QUALITY:
            return False, "low_quality"

        return False, "stable"

    # --------------------------------------------------------
    # ADD KEYFRAME
    # --------------------------------------------------------

    def add_keyframe(
        self,
        frame_id,
        frame,
        keypoints,
        descriptors,
        quality,
        reason,
        good_matches,
        feature_overlap
    ):

        filename = (
            f"keyframe_{frame_id:06d}.jpg"
        )

        filepath = os.path.join(
            KEYFRAME_DIR,
            filename
        )

        cv2.imwrite(
            filepath,
            frame,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                95
            ]
        )

        if self.last_keyframe_frame >= 0:

            interval = (
                frame_id -
                self.last_keyframe_frame
            )

            self.keyframe_intervals.append(
                interval
            )

        self.last_keyframe_frame = frame_id

        self.last_keyframe_image = frame.copy()

        self.last_keyframe_keypoints = keypoints

        self.last_keyframe_descriptors = descriptors

        record = {
            "frame": frame_id,
            "filename": filename,
            "quality": quality,
            "reason": reason,
            "good_matches": good_matches,
            "feature_overlap": feature_overlap
        }

        self.keyframes.append(record)

        self.total_keyframes += 1

        if reason == "manual":
            self.manual_keyframes += 1

        elif reason == "forced":
            self.forced_keyframes += 1

        else:
            self.automatic_keyframes += 1

        self.good_match_history.append(
            good_matches
        )

        self.overlap_history.append(
            feature_overlap
        )

        self.quality_history.append(
            quality
        )

        print(
            f"[KEYFRAME] "
            f"Frame {frame_id:06d} | "
            f"Reason: {reason} | "
            f"Matches: {good_matches} | "
            f"Overlap: {feature_overlap:.3f} | "
            f"Quality: {quality:.1f}"
        )

    # --------------------------------------------------------
    # SAVE METADATA
    # --------------------------------------------------------

    def save_metadata(self):

        with open(
            KEYFRAME_METADATA,
            "w",
            encoding="utf-8"
        ) as f:

            f.write(
                "ARIA-S3D | KEYFRAME METADATA\n"
            )

            f.write("=" * 70 + "\n\n")

            f.write(
                "Frame | Filename | Quality | "
                "Reason | Matches | Overlap\n"
            )

            f.write("-" * 70 + "\n")

            for kf in self.keyframes:

                f.write(
                    f"{kf['frame']:6d} | "
                    f"{kf['filename']:18s} | "
                    f"{kf['quality']:7.2f} | "
                    f"{kf['reason']:18s} | "
                    f"{kf['good_matches']:7d} | "
                    f"{kf['feature_overlap']:.4f}\n"
                )

    # --------------------------------------------------------
    # SAVE STATISTICS
    # --------------------------------------------------------

    def save_statistics(
        self,
        frames_processed,
        processing_fps
    ):

        if self.keyframe_intervals:

            average_interval = float(
                np.mean(
                    self.keyframe_intervals
                )
            )

            minimum_interval = int(
                np.min(
                    self.keyframe_intervals
                )
            )

            maximum_interval = int(
                np.max(
                    self.keyframe_intervals
                )
            )

        else:

            average_interval = 0.0
            minimum_interval = 0
            maximum_interval = 0

        if self.quality_history:

            average_quality = float(
                np.mean(
                    self.quality_history
                )
            )

        else:

            average_quality = 0.0

        if self.good_match_history:

            average_matches = float(
                np.mean(
                    self.good_match_history
                )
            )

        else:

            average_matches = 0.0

        if self.overlap_history:

            average_overlap = float(
                np.mean(
                    self.overlap_history
                )
            )

        else:

            average_overlap = 0.0

        with open(
            KEYFRAME_STATS,
            "w",
            encoding="utf-8"
        ) as f:

            f.write(
                "ARIA-S3D | PHASE 4.7\n"
            )

            f.write(
                "KEYFRAME MANAGEMENT STATISTICS\n"
            )

            f.write("=" * 70 + "\n\n")

            f.write(
                f"Frames processed        : "
                f"{frames_processed}\n"
            )

            f.write(
                f"Total keyframes         : "
                f"{self.total_keyframes}\n"
            )

            f.write(
                f"Automatic keyframes    : "
                f"{self.automatic_keyframes}\n"
            )

            f.write(
                f"Manual keyframes       : "
                f"{self.manual_keyframes}\n"
            )

            f.write(
                f"Forced keyframes       : "
                f"{self.forced_keyframes}\n"
            )

            f.write(
                f"Average keyframe gap   : "
                f"{average_interval:.2f} frames\n"
            )

            f.write(
                f"Minimum keyframe gap   : "
                f"{minimum_interval} frames\n"
            )

            f.write(
                f"Maximum keyframe gap   : "
                f"{maximum_interval} frames\n"
            )

            f.write(
                f"Average good matches   : "
                f"{average_matches:.2f}\n"
            )

            f.write(
                f"Average feature overlap: "
                f"{average_overlap:.4f}\n"
            )

            f.write(
                f"Average keyframe quality: "
                f"{average_quality:.2f}\n"
            )

            f.write(
                f"Processing FPS         : "
                f"{processing_fps:.2f}\n"
            )


# ============================================================
# MAIN
# ============================================================

def main():

    print_header(
        "ARIA-S3D | PHASE 4.7\n"
        "KEYFRAME MANAGEMENT"
    )

    ensure_directories()

    # --------------------------------------------------------
    # LOAD VIDEO
    # --------------------------------------------------------

    cap, width, height, video_fps = load_video()

    print()
    print(
        f"Video resolution : "
        f"{width} x {height}"
    )

    print(
        f"Video FPS        : "
        f"{video_fps:.2f}"
    )

    print(
        f"SIFT features    : "
        f"{SIFT_FEATURES}"
    )

    print(
        f"Minimum matches  : "
        f"{MIN_GOOD_MATCHES}"
    )

    print(
        f"Keyframe interval: "
        f"{MIN_KEYFRAME_INTERVAL} - "
        f"{MAX_KEYFRAME_INTERVAL} frames"
    )

    print()
    print("Controls:")
    print("Q / ESC : Stop")
    print("R       : Force new keyframe")
    print("K       : Save current frame as keyframe")

    print()
    print("-" * 70)
    print("KEYFRAME MANAGEMENT STARTED")
    print("-" * 70)

    # --------------------------------------------------------
    # INITIALIZE
    # --------------------------------------------------------

    sift = create_sift()

    matcher = create_matcher()

    manager = KeyframeManager()

    frame_id = 0

    frames_processed = 0

    start_time = time.time()

    previous_frame = None

    current_frame = None

    # --------------------------------------------------------
    # VIDEO LOOP
    # --------------------------------------------------------

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        current_frame = frame.copy()

        frame_id += 1

        frames_processed += 1

        # ----------------------------------------------------
        # FEATURE DETECTION
        # ----------------------------------------------------

        keypoints, descriptors = detect_features(
            sift,
            current_frame
        )

        quality = calculate_keyframe_quality(
            keypoints,
            descriptors
        )

        # ----------------------------------------------------
        # FEATURE COMPARISON
        # ----------------------------------------------------

        if (
            manager.last_keyframe_descriptors
            is not None
        ):

            good_matches, feature_overlap = (
                calculate_feature_overlap(
                    matcher,
                    descriptors,
                    manager.last_keyframe_descriptors
                )
            )

        else:

            good_matches = 0
            feature_overlap = 0.0

        # ----------------------------------------------------
        # SCENE CHANGE
        # ----------------------------------------------------

        scene_change = calculate_scene_change(
            previous_frame,
            current_frame
        )

        # ----------------------------------------------------
        # CHECK KEYFRAME CONDITION
        # ----------------------------------------------------

        create_keyframe, reason = (
            manager.should_create_keyframe(
                frame_id,
                good_matches,
                feature_overlap,
                scene_change,
                quality
            )
        )

        # ----------------------------------------------------
        # AUTOMATIC KEYFRAME
        # ----------------------------------------------------

        if create_keyframe:

            manager.add_keyframe(
                frame_id,
                current_frame,
                keypoints,
                descriptors,
                quality,
                reason,
                good_matches,
                feature_overlap
            )

        # ----------------------------------------------------
        # DISPLAY
        # ----------------------------------------------------

        display = current_frame.copy()

        cv2.putText(
            display,
            f"ARIA-S3D | PHASE 4.7",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

        cv2.putText(
            display,
            f"Frame: {frame_id}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            display,
            f"Features: {len(keypoints)}",
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            display,
            f"Matches: {good_matches}",
            (20, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            display,
            f"Keyframes: {manager.total_keyframes}",
            (20, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            display,
            f"Overlap: {feature_overlap:.3f}",
            (20, 190),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        if manager.last_keyframe_frame >= 0:

            gap = (
                frame_id -
                manager.last_keyframe_frame
            )

        else:

            gap = 0

        cv2.putText(
            display,
            f"KF gap: {gap}",
            (20, 220),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "ARIA-S3D | Keyframe Management",
            display
        )

        # ----------------------------------------------------
        # CONTROLS
        # ----------------------------------------------------

        key = cv2.waitKey(1) & 0xFF

        # STOP
        if key == ord("q") or key == 27:
            break

        # FORCE KEYFRAME
        elif key == ord("r"):

            manager.add_keyframe(
                frame_id,
                current_frame,
                keypoints,
                descriptors,
                quality,
                "forced",
                good_matches,
                feature_overlap
            )

        # MANUAL KEYFRAME
        elif key == ord("k"):

            manager.add_keyframe(
                frame_id,
                current_frame,
                keypoints,
                descriptors,
                quality,
                "manual",
                good_matches,
                feature_overlap
            )

        previous_frame = current_frame.copy()

    # --------------------------------------------------------
    # CLEANUP
    # --------------------------------------------------------

    cap.release()

    cv2.destroyAllWindows()

    elapsed = time.time() - start_time

    if elapsed > 0:

        processing_fps = (
            frames_processed /
            elapsed
        )

    else:

        processing_fps = 0.0

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    manager.save_metadata()

    manager.save_statistics(
        frames_processed,
        processing_fps
    )

    # --------------------------------------------------------
    # FINAL OUTPUT
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("ARIA-S3D | PHASE 4.7 COMPLETE")
    print("=" * 70)

    print(
        f"Frames processed        : "
        f"{frames_processed}"
    )

    print(
        f"Total keyframes         : "
        f"{manager.total_keyframes}"
    )

    print(
        f"Automatic keyframes     : "
        f"{manager.automatic_keyframes}"
    )

    print(
        f"Manual keyframes        : "
        f"{manager.manual_keyframes}"
    )

    print(
        f"Forced keyframes        : "
        f"{manager.forced_keyframes}"
    )

    if manager.keyframe_intervals:

        print(
            f"Average keyframe gap    : "
            f"{np.mean(manager.keyframe_intervals):.2f} "
            f"frames"
        )

    print(
        f"Processing FPS          : "
        f"{processing_fps:.2f}"
    )

    print()
    print("Generated outputs:")

    print(
        f"  {KEYFRAME_DIR}"
    )

    print(
        f"  {KEYFRAME_METADATA}"
    )

    print(
        f"  {KEYFRAME_STATS}"
    )

    print()
    print("-" * 70)

    print("NEXT STEP:")
    print("Phase 4.8 - Real-time optimization")

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