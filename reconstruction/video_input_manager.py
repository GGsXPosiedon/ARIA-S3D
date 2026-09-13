"""
ARIA-S3D
PHASE 5.9A
UNIVERSAL DRONE VIDEO INPUT MANAGER

Purpose:
    Register and analyze ANY supported drone video before the
    ARIA-S3D reconstruction pipeline begins.

Usage:
    python reconstruction/video_input_manager.py "path/to/video.mp4"

Outputs:
    data/input/video_metadata.npz
    data/input/video_metadata.txt
    data/input/source_video_path.txt
    data/input/preview_frames/
"""

from pathlib import Path
import argparse
import sys
import time

import cv2
import numpy as np


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_DIR = PROJECT_ROOT / "data" / "input"

PREVIEW_DIR = INPUT_DIR / "preview_frames"

METADATA_NPZ = INPUT_DIR / "video_metadata.npz"

METADATA_TXT = INPUT_DIR / "video_metadata.txt"

SOURCE_PATH_FILE = INPUT_DIR / "source_video_path.txt"


# ============================================================
# CONFIGURATION
# ============================================================

PREVIEW_FRAME_COUNT = 8

MIN_VIDEO_FRAMES = 10

MIN_WIDTH = 320
MIN_HEIGHT = 240

SUPPORTED_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".m4v",
    ".webm"
}


# ============================================================
# DISPLAY
# ============================================================

def header(title):

    print("=" * 70)
    print("ARIA-S3D | UNIVERSAL VIDEO INPUT")
    print(title)
    print("=" * 70)
    print()


def section(number, title):

    print()
    print(f"[{number}] {title}")
    print("-" * 70)


def ok(message):

    print(f"[OK] {message}")


def warning(message):

    print(f"[WARNING] {message}")


def fail(message):

    print(f"[ERROR] {message}")


# ============================================================
# VIDEO QUALITY ANALYSIS
# ============================================================

def analyze_frame_quality(frame):

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    # --------------------------------------------------------
    # Sharpness
    # --------------------------------------------------------

    sharpness = cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()

    # --------------------------------------------------------
    # Brightness
    # --------------------------------------------------------

    brightness = float(
        np.mean(gray)
    )

    # --------------------------------------------------------
    # Contrast
    # --------------------------------------------------------

    contrast = float(
        np.std(gray)
    )

    # --------------------------------------------------------
    # Feature density
    # --------------------------------------------------------

    detector = cv2.SIFT_create(
        nfeatures=1500
    )

    keypoints = detector.detect(
        gray,
        None
    )

    feature_count = len(
        keypoints
    )

    return {
        "sharpness": float(sharpness),
        "brightness": brightness,
        "contrast": contrast,
        "features": int(feature_count)
    }


# ============================================================
# ADAPTIVE FEATURE TARGET
# ============================================================

def recommend_feature_count(
    width,
    height
):

    megapixels = (
        width * height
    ) / 1_000_000.0

    if megapixels < 1.0:

        return 2000

    elif megapixels < 2.5:

        return 3000

    elif megapixels < 5.0:

        return 4000

    else:

        return 5000


# ============================================================
# FRAME SAMPLING RECOMMENDATION
# ============================================================

def recommend_frame_interval(
    fps,
    total_frames
):

    duration = (
        total_frames / fps
        if fps > 0
        else 0.0
    )

    # Short clips
    if duration <= 20:

        return 2

    # Medium clips
    elif duration <= 60:

        return 3

    # Longer videos
    elif duration <= 180:

        return 5

    else:

        return 8


# ============================================================
# VIDEO ANALYSIS
# ============================================================

def analyze_video(video_path):

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        raise RuntimeError(
            "OpenCV could not open the supplied video."
        )

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    fps = float(
        cap.get(
            cv2.CAP_PROP_FPS
        )
    )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    if fps > 0:

        duration = (
            total_frames / fps
        )

    else:

        duration = 0.0

    cap.release()

    return (
        width,
        height,
        fps,
        total_frames,
        duration
    )


# ============================================================
# PREVIEW FRAME EXTRACTION
# ============================================================

def extract_preview_frames(
    video_path,
    total_frames
):

    PREVIEW_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Remove old preview frames.
    for old_file in PREVIEW_DIR.glob(
        "preview_*.jpg"
    ):

        old_file.unlink()

    if total_frames <= 0:

        return [], []

    sample_indices = np.linspace(
        0,
        max(total_frames - 1, 0),
        num=min(
            PREVIEW_FRAME_COUNT,
            total_frames
        ),
        dtype=int
    )

    cap = cv2.VideoCapture(
        str(video_path)
    )

    quality_results = []

    saved_files = []

    for number, frame_index in enumerate(
        sample_indices
    ):

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            int(frame_index)
        )

        success, frame = cap.read()

        if not success or frame is None:

            warning(
                f"Could not read preview frame "
                f"{frame_index}."
            )

            continue

        output_path = (
            PREVIEW_DIR
            /
            f"preview_{number:02d}_frame_{frame_index:06d}.jpg"
        )

        cv2.imwrite(
            str(output_path),
            frame
        )

        stats = analyze_frame_quality(
            frame
        )

        stats["frame"] = int(
            frame_index
        )

        quality_results.append(
            stats
        )

        saved_files.append(
            output_path
        )

    cap.release()

    return (
        saved_files,
        quality_results
    )


# ============================================================
# QUALITY CLASSIFICATION
# ============================================================

def classify_video_quality(
    quality_results
):

    if not quality_results:

        return (
            "UNKNOWN",
            0.0,
            0.0,
            0.0,
            0.0
        )

    sharpness = np.mean([
        item["sharpness"]
        for item in quality_results
    ])

    brightness = np.mean([
        item["brightness"]
        for item in quality_results
    ])

    contrast = np.mean([
        item["contrast"]
        for item in quality_results
    ])

    features = np.mean([
        item["features"]
        for item in quality_results
    ])

    score = 0

    # --------------------------------------------------------
    # Sharpness
    # --------------------------------------------------------

    if sharpness >= 150:

        score += 2

    elif sharpness >= 70:

        score += 1

    # --------------------------------------------------------
    # Brightness
    # --------------------------------------------------------

    if 40 <= brightness <= 215:

        score += 2

    elif 20 <= brightness <= 235:

        score += 1

    # --------------------------------------------------------
    # Contrast
    # --------------------------------------------------------

    if contrast >= 35:

        score += 2

    elif contrast >= 20:

        score += 1

    # --------------------------------------------------------
    # Feature richness
    # --------------------------------------------------------

    if features >= 800:

        score += 2

    elif features >= 300:

        score += 1

    # --------------------------------------------------------
    # Classification
    # --------------------------------------------------------

    if score >= 7:

        classification = "GOOD"

    elif score >= 4:

        classification = "USABLE"

    else:

        classification = "POOR"

    return (
        classification,
        float(sharpness),
        float(brightness),
        float(contrast),
        float(features)
    )


# ============================================================
# ADAPTIVE CAMERA INTRINSIC INITIALIZATION
# ============================================================

def estimate_initial_intrinsics(
    width,
    height
):

    """
    Generic initial camera calibration.

    This is NOT a true calibration.

    It provides a resolution-dependent starting estimate until
    proper metadata/calibration or self-calibration is available.
    """

    focal_guess = float(
        max(
            width,
            height
        )
    )

    fx = focal_guess
    fy = focal_guess

    cx = width / 2.0
    cy = height / 2.0

    K = np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0]
        ],
        dtype=np.float64
    )

    return K


# ============================================================
# SAVE METADATA
# ============================================================

def save_metadata(
    video_path,
    width,
    height,
    fps,
    total_frames,
    duration,
    quality,
    sharpness,
    brightness,
    contrast,
    feature_average,
    sift_features,
    frame_interval,
    K
):

    INPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    SOURCE_PATH_FILE.write_text(
        str(video_path.resolve()),
        encoding="utf-8"
    )

    np.savez(
        METADATA_NPZ,

        video_path=str(
            video_path.resolve()
        ),

        width=width,

        height=height,

        fps=fps,

        total_frames=total_frames,

        duration=duration,

        quality=quality,

        average_sharpness=sharpness,

        average_brightness=brightness,

        average_contrast=contrast,

        average_features=feature_average,

        recommended_sift_features=sift_features,

        recommended_frame_interval=frame_interval,

        K=K
    )

    with open(
        METADATA_TXT,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "ARIA-S3D | UNIVERSAL VIDEO INPUT\n"
        )

        file.write(
            "=" * 70
            +
            "\n\n"
        )

        file.write(
            f"Source video : "
            f"{video_path.resolve()}\n"
        )

        file.write(
            f"Resolution   : "
            f"{width} x {height}\n"
        )

        file.write(
            f"FPS          : "
            f"{fps:.6f}\n"
        )

        file.write(
            f"Frames       : "
            f"{total_frames}\n"
        )

        file.write(
            f"Duration     : "
            f"{duration:.3f} seconds\n\n"
        )

        file.write(
            f"Video quality       : "
            f"{quality}\n"
        )

        file.write(
            f"Average sharpness   : "
            f"{sharpness:.3f}\n"
        )

        file.write(
            f"Average brightness  : "
            f"{brightness:.3f}\n"
        )

        file.write(
            f"Average contrast    : "
            f"{contrast:.3f}\n"
        )

        file.write(
            f"Average SIFT features: "
            f"{feature_average:.2f}\n\n"
        )

        file.write(
            "ADAPTIVE PARAMETERS\n"
        )

        file.write(
            "-" * 70
            +
            "\n"
        )

        file.write(
            f"Recommended SIFT features : "
            f"{sift_features}\n"
        )

        file.write(
            f"Recommended frame interval: "
            f"{frame_interval}\n\n"
        )

        file.write(
            "INITIAL INTRINSIC MATRIX\n"
        )

        file.write(
            "-" * 70
            +
            "\n"
        )

        for row in K:

            file.write(
                " ".join(
                    f"{value:.6f}"
                    for value in row
                )
                +
                "\n"
            )

        file.write(
            "\nIMPORTANT:\n"
        )

        file.write(
            "This intrinsic matrix is an initialization only.\n"
        )

        file.write(
            "Metric calibration should be used when available.\n"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "ARIA-S3D universal drone video input manager"
        )
    )

    parser.add_argument(
        "video",
        type=str,
        help="Path to drone video"
    )

    args = parser.parse_args()

    start_time = time.perf_counter()

    header(
        "DRONE VIDEO ANALYSIS & REGISTRATION"
    )

    video_path = Path(
        args.video
    ).expanduser()

    # ========================================================
    # 1. PATH VALIDATION
    # ========================================================

    section(
        1,
        "Checking video input"
    )

    if not video_path.exists():

        fail(
            f"Video not found:\n"
            f"        {video_path}"
        )

        raise SystemExit(1)

    if not video_path.is_file():

        fail(
            "Input path is not a file."
        )

        raise SystemExit(1)

    extension = (
        video_path.suffix.lower()
    )

    if extension not in SUPPORTED_EXTENSIONS:

        warning(
            f"Extension {extension} is not in "
            "the standard ARIA-S3D extension list."
        )

        warning(
            "OpenCV will still attempt to decode it."
        )

    ok(
        f"Video found:\n"
        f"     {video_path.resolve()}"
    )

    # ========================================================
    # 2. ANALYZE VIDEO
    # ========================================================

    section(
        2,
        "Analyzing video stream"
    )

    try:

        (
            width,
            height,
            fps,
            total_frames,
            duration
        ) = analyze_video(
            video_path
        )

    except Exception as error:

        fail(
            str(error)
        )

        raise SystemExit(1)

    print(
        f"Resolution   : "
        f"{width} x {height}"
    )

    print(
        f"FPS          : "
        f"{fps:.3f}"
    )

    print(
        f"Frames       : "
        f"{total_frames}"
    )

    print(
        f"Duration     : "
        f"{duration:.2f} seconds"
    )

    # ========================================================
    # 3. BASIC SUITABILITY
    # ========================================================

    section(
        3,
        "Checking reconstruction suitability"
    )

    if total_frames < MIN_VIDEO_FRAMES:

        fail(
            "Video contains too few frames "
            "for reconstruction."
        )

        raise SystemExit(1)

    if (
        width < MIN_WIDTH
        or
        height < MIN_HEIGHT
    ):

        warning(
            "Video resolution is very low."
        )

    else:

        ok(
            "Video resolution is suitable."
        )

    if fps <= 0:

        warning(
            "FPS could not be reliably determined."
        )

    else:

        ok(
            "Video timing information is valid."
        )

    # ========================================================
    # 4. PREVIEW EXTRACTION
    # ========================================================

    section(
        4,
        "Extracting diagnostic preview frames"
    )

    (
        preview_files,
        quality_results
    ) = extract_preview_frames(
        video_path,
        total_frames
    )

    print(
        f"Preview frames saved : "
        f"{len(preview_files)}"
    )

    for path in preview_files:

        print(
            f"  {path.name}"
        )

    # ========================================================
    # 5. VIDEO QUALITY
    # ========================================================

    section(
        5,
        "Analyzing visual quality"
    )

    (
        quality,
        sharpness,
        brightness,
        contrast,
        feature_average
    ) = classify_video_quality(
        quality_results
    )

    print(
        f"Quality classification : "
        f"{quality}"
    )

    print(
        f"Average sharpness      : "
        f"{sharpness:.2f}"
    )

    print(
        f"Average brightness     : "
        f"{brightness:.2f}"
    )

    print(
        f"Average contrast       : "
        f"{contrast:.2f}"
    )

    print(
        f"Average SIFT features  : "
        f"{feature_average:.2f}"
    )

    if quality == "GOOD":

        ok(
            "Video appears well suited "
            "for reconstruction."
        )

    elif quality == "USABLE":

        warning(
            "Video is usable but may require "
            "adaptive reconstruction parameters."
        )

    else:

        warning(
            "Video quality may make reliable "
            "3D reconstruction difficult."
        )

    # ========================================================
    # 6. ADAPTIVE SETTINGS
    # ========================================================

    section(
        6,
        "Computing adaptive pipeline parameters"
    )

    sift_features = (
        recommend_feature_count(
            width,
            height
        )
    )

    frame_interval = (
        recommend_frame_interval(
            fps,
            total_frames
        )
    )

    print(
        f"Recommended SIFT features : "
        f"{sift_features}"
    )

    print(
        f"Recommended frame interval: "
        f"{frame_interval}"
    )

    # ========================================================
    # 7. CAMERA INITIALIZATION
    # ========================================================

    section(
        7,
        "Initializing camera model"
    )

    K = estimate_initial_intrinsics(
        width,
        height
    )

    print(
        "Initial intrinsic matrix:"
    )

    print(
        K
    )

    print()

    warning(
        "This is only an initialization."
    )

    warning(
        "Future ARIA-S3D calibration will "
        "refine/replace it."
    )

    # ========================================================
    # 8. SAVE
    # ========================================================

    section(
        8,
        "Registering video with ARIA-S3D"
    )

    save_metadata(
        video_path,
        width,
        height,
        fps,
        total_frames,
        duration,
        quality,
        sharpness,
        brightness,
        contrast,
        feature_average,
        sift_features,
        frame_interval,
        K
    )

    ok(
        f"Metadata saved:\n"
        f"     {METADATA_NPZ}"
    )

    ok(
        f"Human-readable report saved:\n"
        f"     {METADATA_TXT}"
    )

    ok(
        f"Source path registered:\n"
        f"     {SOURCE_PATH_FILE}"
    )

    # ========================================================
    # COMPLETE
    # ========================================================

    elapsed = (
        time.perf_counter()
        -
        start_time
    )

    print()

    print(
        "=" * 70
    )

    print(
        "ARIA-S3D | VIDEO INPUT REGISTERED"
    )

    print(
        "=" * 70
    )

    print()

    print(
        f"Video          : "
        f"{video_path.name}"
    )

    print(
        f"Resolution     : "
        f"{width} x {height}"
    )

    print(
        f"FPS            : "
        f"{fps:.2f}"
    )

    print(
        f"Frames         : "
        f"{total_frames}"
    )

    print(
        f"Quality        : "
        f"{quality}"
    )

    print(
        f"SIFT target    : "
        f"{sift_features}"
    )

    print(
        f"Frame interval : "
        f"{frame_interval}"
    )

    print(
        f"Analysis time  : "
        f"{elapsed:.2f} seconds"
    )

    print()

    print(
        "NEXT:"
    )

    print(
        "The reconstruction pipeline should now read "
        "data/input/video_metadata.npz instead of "
        "hard-coded video/camera parameters."
    )

    print()

    print(
        "ARIA-S3D IS NOW MOVING TOWARD:"
    )

    print(
        "ANY SUPPORTED DRONE VIDEO"
    )

    print(
        "          ↓"
    )

    print(
        "AUTO ANALYSIS"
    )

    print(
        "          ↓"
    )

    print(
        "ADAPTIVE RECONSTRUCTION"
    )

    print(
        "          ↓"
    )

    print(
        "3D MODEL + TEXTURE"
    )

    print(
        "=" * 70
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()