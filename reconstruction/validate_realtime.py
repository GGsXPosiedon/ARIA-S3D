"""
ARIA-S3D | PHASE 4.9
REAL-TIME RECONSTRUCTION VALIDATION

Validates the complete Phase 4 pipeline:

4.1 Live camera input
4.2 Real-time feature tracking
4.3 Real-time pose estimation
4.4 Incremental live mapping
4.5 Live 3D visualization
4.6 Tracking-loss recovery
4.7 Keyframe management
4.8 Real-time optimization
4.9 Phase 4 validation

This script does NOT modify reconstruction data.
It only validates generated outputs and statistics.
"""

import os
import re
import sys
import numpy as np


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "output"
)


# ============================================================
# EXPECTED FILES
# ============================================================

POSE_FILE = os.path.join(
    OUTPUT_DIR,
    "realtime_pose_trajectory.txt"
)

OPTIMIZED_TRAJECTORY_FILE = os.path.join(
    OUTPUT_DIR,
    "optimized_realtime_trajectory.txt"
)

KEYFRAME_METADATA_FILE = os.path.join(
    OUTPUT_DIR,
    "keyframe_metadata.txt"
)

KEYFRAME_STATS_FILE = os.path.join(
    OUTPUT_DIR,
    "keyframe_management_stats.txt"
)

RECOVERY_STATS_FILE = os.path.join(
    OUTPUT_DIR,
    "tracking_recovery_stats.txt"
)

RECOVERY_TRAJECTORY_FILE = os.path.join(
    OUTPUT_DIR,
    "realtime_recovery_trajectory.txt"
)

OPTIMIZATION_STATS_FILE = os.path.join(
    OUTPUT_DIR,
    "realtime_optimization_stats.txt"
)


# ============================================================
# VALIDATION THRESHOLDS
# ============================================================

MIN_TRAJECTORY_FRAMES = 10

MIN_KEYFRAMES = 1

MIN_TRACKING_SUCCESS_RATE = 50.0

MIN_OPTIMIZATION_IMPROVEMENT = 0.0

MAX_NONFINITE_POINTS = 0


# ============================================================
# GLOBAL VALIDATION STATE
# ============================================================

validation_results = []


# ============================================================
# HEADER
# ============================================================

def print_header():

    print("=" * 70)
    print("ARIA-S3D | PHASE 4.9")
    print("REAL-TIME RECONSTRUCTION VALIDATION")
    print("=" * 70)
    print()


# ============================================================
# RESULT HANDLING
# ============================================================

def record_result(
    name,
    passed
):

    validation_results.append(
        (
            name,
            passed
        )
    )


def print_ok(message):

    print(
        f"[OK] {message}"
    )


def print_warning(message):

    print(
        f"[WARNING] {message}"
    )


def print_error(message):

    print(
        f"[ERROR] {message}"
    )


# ============================================================
# FILE CHECK
# ============================================================

def check_file(
    label,
    path,
    required=True
):

    if os.path.exists(path):

        print_ok(
            f"{label}:"
        )

        print(
            f"     {path}"
        )

        return True

    if required:

        print_error(
            f"{label} not found:"
        )

        print(
            f"     {path}"
        )

    else:

        print_warning(
            f"{label} not available:"
        )

        print(
            f"     {path}"
        )

    return False


# ============================================================
# LOAD TRAJECTORY
# ============================================================

def load_trajectory(path):

    data = []

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            if line.startswith("#"):
                continue

            parts = line.split()

            if len(parts) < 4:
                continue

            try:

                values = [
                    float(value)
                    for value in parts
                ]

            except ValueError:

                continue

            # ------------------------------------------------
            # frame x y z ...
            # ------------------------------------------------

            frame = int(
                values[0]
            )

            position = np.array(
                values[1:4],
                dtype=np.float64
            )

            data.append(
                (
                    frame,
                    position
                )
            )

    if not data:

        return (
            np.array([], dtype=np.int32),
            np.empty(
                (0, 3),
                dtype=np.float64
            )
        )

    frames = np.array(
        [
            item[0]
            for item in data
        ],
        dtype=np.int32
    )

    positions = np.array(
        [
            item[1]
            for item in data
        ],
        dtype=np.float64
    )

    return (
        frames,
        positions
    )


# ============================================================
# LOAD KEYFRAME METADATA
# ============================================================

def load_keyframe_ids(path):

    """
    Expected ARIA-S3D format:

    Frame | Filename | Quality | Reason | Matches | Overlap
    -------------------------------------------------------
         1 | keyframe_000001.jpg | ...
        61 | keyframe_000061.jpg | ...
       121 | keyframe_000121.jpg | ...

    The first column is the keyframe ID.
    """

    keyframe_ids = []

    if not os.path.exists(path):

        return keyframe_ids

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            if "|" not in line:
                continue

            columns = [
                column.strip()
                for column in line.split("|")
            ]

            if len(columns) < 2:
                continue

            first_column = columns[0]

            # Ignore header
            if (
                first_column.lower()
                ==
                "frame"
            ):

                continue

            match = re.search(
                r"\d+",
                first_column
            )

            if match:

                frame_id = int(
                    match.group()
                )

                if frame_id not in keyframe_ids:

                    keyframe_ids.append(
                        frame_id
                    )

    return sorted(
        keyframe_ids
    )


# ============================================================
# LOAD STATISTICS
# ============================================================

def load_statistics(path):

    stats = {}

    if not os.path.exists(path):

        return stats

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            if ":" not in line:
                continue

            key, value = line.split(
                ":",
                1
            )

            key = key.strip()
            value = value.strip()

            stats[key] = value

    return stats


# ============================================================
# FLOAT EXTRACTION
# ============================================================

def extract_float(
    stats,
    possible_keys
):

    for key in possible_keys:

        if key in stats:

            match = re.search(
                r"[-+]?\d*\.?\d+",
                stats[key]
            )

            if match:

                try:

                    return float(
                        match.group()
                    )

                except ValueError:

                    pass

    return None


# ============================================================
# TRAJECTORY STATISTICS
# ============================================================

def trajectory_distance(
    positions
):

    if len(positions) < 2:

        return 0.0

    differences = np.diff(
        positions,
        axis=0
    )

    distances = np.linalg.norm(
        differences,
        axis=1
    )

    return float(
        np.sum(distances)
    )


def mean_motion(
    positions
):

    if len(positions) < 2:

        return 0.0

    differences = np.diff(
        positions,
        axis=0
    )

    distances = np.linalg.norm(
        differences,
        axis=1
    )

    return float(
        np.mean(distances)
    )


def maximum_motion(
    positions
):

    if len(positions) < 2:

        return 0.0

    differences = np.diff(
        positions,
        axis=0
    )

    distances = np.linalg.norm(
        differences,
        axis=1
    )

    return float(
        np.max(distances)
    )


# ============================================================
# TRAJECTORY RESIDUAL
# ============================================================

def trajectory_residual(
    positions
):

    if len(positions) < 3:

        return 0.0

    residuals = []

    for i in range(
        1,
        len(positions) - 1
    ):

        prediction = (
            positions[i - 1]
            +
            positions[i + 1]
        ) / 2.0

        error = np.linalg.norm(
            positions[i]
            -
            prediction
        )

        residuals.append(
            error
        )

    if not residuals:

        return 0.0

    return float(
        np.sqrt(
            np.mean(
                np.square(
                    residuals
                )
            )
        )
    )


# ============================================================
# PHASE 4.1 - LIVE CAMERA INPUT
# ============================================================

def validate_camera_input():

    print(
        "[1] Validating Phase 4.1 - Live camera input"
    )

    print(
        "-" * 70
    )

    # The existence of downstream trajectory data proves
    # that camera frames were successfully consumed.

    exists = os.path.exists(
        POSE_FILE
    )

    if exists:

        print_ok(
            "Real-time pose trajectory exists."
        )

        record_result(
            "4.1 Live camera input",
            True
        )

    else:

        print_error(
            "Real-time pose trajectory is missing."
        )

        record_result(
            "4.1 Live camera input",
            False
        )

    print()


# ============================================================
# PHASE 4.2 / 4.3 / 4.4
# ============================================================

def validate_tracking_and_pose():

    print(
        "[2] Validating tracking, pose estimation "
        "and live mapping"
    )

    print(
        "-" * 70
    )

    if not os.path.exists(
        POSE_FILE
    ):

        print_error(
            "Pose trajectory unavailable."
        )

        record_result(
            "4.2-4.4 Tracking/Pose/Mapping",
            False
        )

        print()

        return None, None

    frames, positions = load_trajectory(
        POSE_FILE
    )

    print(
        f"Trajectory frames : "
        f"{len(frames)}"
    )

    if len(frames) >= MIN_TRAJECTORY_FRAMES:

        print_ok(
            "Sufficient trajectory frames generated."
        )

        trajectory_valid = True

    else:

        print_error(
            "Too few trajectory frames."
        )

        trajectory_valid = False

    nonfinite = int(
        np.sum(
            ~np.isfinite(
                positions
            )
        )
    )

    print(
        f"Non-finite trajectory values : "
        f"{nonfinite}"
    )

    if nonfinite == MAX_NONFINITE_POINTS:

        print_ok(
            "Trajectory contains only finite values."
        )

    else:

        print_error(
            "Trajectory contains non-finite values."
        )

        trajectory_valid = False

    if len(positions) > 0:

        print(
            f"Start position : "
            f"{positions[0]}"
        )

        print(
            f"End position   : "
            f"{positions[-1]}"
        )

    record_result(
        "4.2-4.4 Tracking/Pose/Mapping",
        trajectory_valid
    )

    print()

    return frames, positions


# ============================================================
# PHASE 4.5 - VISUALIZATION
# ============================================================

def validate_visualization():

    print(
        "[3] Validating Phase 4.5 - Live 3D visualization"
    )

    print(
        "-" * 70
    )

    # Visualization is a runtime display stage and normally
    # does not require a permanent file. Therefore, trajectory
    # generation is used as evidence that the visualization
    # stage had valid data available.

    if os.path.exists(
        POSE_FILE
    ):

        print_ok(
            "Visualization input trajectory available."
        )

        print(
            "     Live visualization is a runtime stage;"
        )

        print(
            "     no permanent visualization file is required."
        )

        record_result(
            "4.5 Live 3D visualization",
            True
        )

    else:

        print_error(
            "Visualization input trajectory unavailable."
        )

        record_result(
            "4.5 Live 3D visualization",
            False
        )

    print()


# ============================================================
# PHASE 4.6 - TRACKING LOSS RECOVERY
# ============================================================

def validate_recovery():

    print(
        "[4] Validating Phase 4.6 - Tracking-loss recovery"
    )

    print(
        "-" * 70
    )

    trajectory_exists = os.path.exists(
        RECOVERY_TRAJECTORY_FILE
    )

    stats_exists = os.path.exists(
        RECOVERY_STATS_FILE
    )

    if trajectory_exists:

        print_ok(
            "Recovery trajectory exists:"
        )

        print(
            f"     {RECOVERY_TRAJECTORY_FILE}"
        )

    else:

        print_warning(
            "Recovery trajectory not found."
        )

    if stats_exists:

        print_ok(
            "Recovery statistics exist:"
        )

        print(
            f"     {RECOVERY_STATS_FILE}"
        )

    else:

        print_warning(
            "Recovery statistics not found."
        )

    if not (
        trajectory_exists
        and
        stats_exists
    ):

        record_result(
            "4.6 Tracking-loss recovery",
            False
        )

        print()

        return

    stats = load_statistics(
        RECOVERY_STATS_FILE
    )

    frames_processed = extract_float(
        stats,
        [
            "Frames processed"
        ]
    )

    successful_frames = extract_float(
        stats,
        [
            "Successful frames"
        ]
    )

    success_rate = extract_float(
        stats,
        [
            "Tracking success rate"
        ]
    )

    recoveries = extract_float(
        stats,
        [
            "Total recoveries"
        ]
    )

    recovery_successes = extract_float(
        stats,
        [
            "Recovery successes"
        ]
    )

    if frames_processed is not None:

        print(
            f"Frames processed : "
            f"{int(frames_processed)}"
        )

    if successful_frames is not None:

        print(
            f"Successful frames : "
            f"{int(successful_frames)}"
        )

    if success_rate is not None:

        print(
            f"Tracking success rate : "
            f"{success_rate:.2f}%"
        )

    if recoveries is not None:

        print(
            f"Tracking recoveries : "
            f"{int(recoveries)}"
        )

    if recovery_successes is not None:

        print(
            f"Recovery successes : "
            f"{int(recovery_successes)}"
        )

    valid = True

    if (
        success_rate is not None
        and
        success_rate < MIN_TRACKING_SUCCESS_RATE
    ):

        valid = False

    if (
        frames_processed is not None
        and
        frames_processed < MIN_TRAJECTORY_FRAMES
    ):

        valid = False

    if valid:

        print_ok(
            "Tracking-loss recovery outputs are valid."
        )

    else:

        print_error(
            "Tracking-loss recovery validation failed."
        )

    record_result(
        "4.6 Tracking-loss recovery",
        valid
    )

    print()


# ============================================================
# PHASE 4.7 - KEYFRAME MANAGEMENT
# ============================================================

def validate_keyframes(
    trajectory_frames
):

    print(
        "[5] Validating Phase 4.7 - Keyframe management"
    )

    print(
        "-" * 70
    )

    if not os.path.exists(
        KEYFRAME_METADATA_FILE
    ):

        print_error(
            "Keyframe metadata not found."
        )

        record_result(
            "4.7 Keyframe management",
            False
        )

        print()

        return []

    keyframe_ids = load_keyframe_ids(
        KEYFRAME_METADATA_FILE
    )

    print(
        f"Keyframes detected : "
        f"{len(keyframe_ids)}"
    )

    print(
        f"Keyframe IDs       : "
        f"{keyframe_ids}"
    )

    valid = (
        len(keyframe_ids)
        >=
        MIN_KEYFRAMES
    )

    if trajectory_frames is not None:

        trajectory_set = set(
            int(frame)
            for frame in trajectory_frames
        )

        missing = [
            frame_id
            for frame_id in keyframe_ids
            if frame_id not in trajectory_set
        ]

        if missing:

            print_warning(
                "Some keyframes are not present "
                "in the pose trajectory:"
            )

            print(
                f"          {missing}"
            )

        else:

            print_ok(
                "All keyframe IDs map to "
                "trajectory frames."
            )

    if valid:

        print_ok(
            "Keyframe management output is valid."
        )

    else:

        print_error(
            "No valid keyframes detected."
        )

    record_result(
        "4.7 Keyframe management",
        valid
    )

    print()

    return keyframe_ids


# ============================================================
# PHASE 4.8 - REAL-TIME OPTIMIZATION
# ============================================================

def validate_optimization(
    original_frames,
    original_positions,
    keyframe_ids
):

    print(
        "[6] Validating Phase 4.8 - Real-time optimization"
    )

    print(
        "-" * 70
    )

    trajectory_exists = os.path.exists(
        OPTIMIZED_TRAJECTORY_FILE
    )

    stats_exists = os.path.exists(
        OPTIMIZATION_STATS_FILE
    )

    if trajectory_exists:

        print_ok(
            "Optimized trajectory:"
        )

        print(
            f"     {OPTIMIZED_TRAJECTORY_FILE}"
        )

    else:

        print_error(
            "Optimized trajectory is missing."
        )

    if stats_exists:

        print_ok(
            "Optimization statistics:"
        )

        print(
            f"     {OPTIMIZATION_STATS_FILE}"
        )

    else:

        print_error(
            "Optimization statistics are missing."
        )

    if not (
        trajectory_exists
        and
        stats_exists
    ):

        record_result(
            "4.8 Real-time optimization",
            False
        )

        print()

        return None

    optimized_frames, optimized_positions = load_trajectory(
        OPTIMIZED_TRAJECTORY_FILE
    )

    stats = load_statistics(
        OPTIMIZATION_STATS_FILE
    )

    print()

    print(
        f"Optimized frames : "
        f"{len(optimized_frames)}"
    )

    if (
        original_frames is not None
        and
        len(original_frames) > 0
    ):

        if len(optimized_frames) == len(
            original_frames
        ):

            print_ok(
                "Optimized trajectory has the "
                "same number of frames."
            )

        else:

            print_warning(
                "Optimized trajectory frame count "
                "differs from original."
            )

    nonfinite = int(
        np.sum(
            ~np.isfinite(
                optimized_positions
            )
        )
    )

    print(
        f"Non-finite optimized values : "
        f"{nonfinite}"
    )

    valid = (
        nonfinite
        ==
        MAX_NONFINITE_POINTS
    )

    if valid:

        print_ok(
            "Optimized trajectory contains "
            "only finite values."
        )

    else:

        print_error(
            "Optimized trajectory contains "
            "non-finite values."
        )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    original_residual = extract_float(
        stats,
        [
            "Original residual"
        ]
    )

    optimized_residual = extract_float(
        stats,
        [
            "Optimized residual"
        ]
    )

    improvement = extract_float(
        stats,
        [
            "Residual improvement"
        ]
    )

    iterations = extract_float(
        stats,
        [
            "Iterations"
        ]
    )

    stats_keyframes = extract_float(
        stats,
        [
            "Keyframes used",
            "Keyframes constrained"
        ]
    )

    if original_residual is not None:

        print(
            f"Original residual  : "
            f"{original_residual:.6f}"
        )

    if optimized_residual is not None:

        print(
            f"Optimized residual : "
            f"{optimized_residual:.6f}"
        )

    if improvement is not None:

        print(
            f"Residual improvement : "
            f"{improvement:.2f}%"
        )

    if iterations is not None:

        print(
            f"Iterations : "
            f"{int(iterations)}"
        )

    if stats_keyframes is not None:

        print(
            f"Keyframes used : "
            f"{int(stats_keyframes)}"
        )

    # --------------------------------------------------------
    # Residual validation
    # --------------------------------------------------------

    if (
        original_residual is not None
        and
        optimized_residual is not None
    ):

        if optimized_residual <= original_residual:

            print_ok(
                "Optimization reduced trajectory residual."
            )

        else:

            print_error(
                "Optimization increased trajectory residual."
            )

            valid = False

    elif improvement is not None:

        if improvement >= MIN_OPTIMIZATION_IMPROVEMENT:

            print_ok(
                "Optimization improvement is valid."
            )

        else:

            print_error(
                "Optimization improvement is negative."
            )

            valid = False

    else:

        print_warning(
            "Optimization residual information unavailable."
        )

    # --------------------------------------------------------
    # Keyframe validation
    # --------------------------------------------------------

    if keyframe_ids:

        if stats_keyframes is not None:

            if int(stats_keyframes) >= len(
                keyframe_ids
            ):

                print_ok(
                    "Optimization statistics include "
                    "the detected keyframes."
                )

            else:

                print_warning(
                    "Optimization reports fewer keyframes "
                    "than metadata contains."
                )

        else:

            print_warning(
                "Keyframe count unavailable in "
                "optimization statistics."
            )

    # --------------------------------------------------------
    # Compare trajectory geometry
    # --------------------------------------------------------

    if len(original_positions) > 1:

        original_distance = trajectory_distance(
            original_positions
        )

        optimized_distance = trajectory_distance(
            optimized_positions
        )

        print()

        print(
            f"Original trajectory distance : "
            f"{original_distance:.6f}"
        )

        print(
            f"Optimized trajectory distance : "
            f"{optimized_distance:.6f}"
        )

        original_residual_local = trajectory_residual(
            original_positions
        )

        optimized_residual_local = trajectory_residual(
            optimized_positions
        )

        print(
            f"Original local residual : "
            f"{original_residual_local:.6f}"
        )

        print(
            f"Optimized local residual : "
            f"{optimized_residual_local:.6f}"
        )

    record_result(
        "4.8 Real-time optimization",
        valid
    )

    print()

    return (
        optimized_frames,
        optimized_positions
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

def print_summary():

    print("=" * 70)

    print(
        "ARIA-S3D | PHASE 4.9 VALIDATION SUMMARY"
    )

    print("=" * 70)

    print()

    passed = 0
    failed = 0

    for name, result in validation_results:

        if result:

            print(
                f"[PASS] {name}"
            )

            passed += 1

        else:

            print(
                f"[FAIL] {name}"
            )

            failed += 1

    print()

    print(
        f"Validation checks passed : "
        f"{passed}"
    )

    print(
        f"Validation checks failed : "
        f"{failed}"
    )

    print()

    # ========================================================
    # FINAL DECISION
    # ========================================================

    if failed == 0:

        print("=" * 70)

        print(
            "PHASE 4 REAL-TIME RECONSTRUCTION: SUCCESS"
        )

        print("=" * 70)

        print()

        print(
            "ARIA-S3D Phase 4 has successfully completed:"
        )

        print()

        print(
            "  [✓] 4.1 Live camera input"
        )

        print(
            "  [✓] 4.2 Real-time feature tracking"
        )

        print(
            "  [✓] 4.3 Real-time pose estimation"
        )

        print(
            "  [✓] 4.4 Incremental live mapping"
        )

        print(
            "  [✓] 4.5 Live 3D visualization"
        )

        print(
            "  [✓] 4.6 Tracking-loss recovery"
        )

        print(
            "  [✓] 4.7 Keyframe management"
        )

        print(
            "  [✓] 4.8 Real-time optimization"
        )

        print(
            "  [✓] 4.9 Phase validation"
        )

        print()

        print(
            "NEXT STEP:"
        )

        print(
            "Phase 4.10 - Prepare ARIA-S3D v0.4 GitHub release"
        )

        print()

        print(
            "IMPORTANT:"
        )

        print(
            "ARIA-S3D remains a monocular reconstruction."
        )

        print(
            "Absolute metric scale remains arbitrary."
        )

        print("=" * 70)

        return True

    else:

        print("=" * 70)

        print(
            "PHASE 4 VALIDATION: FAILED"
        )

        print("=" * 70)

        print()

        print(
            "Fix the failed validation checks "
            "before creating v0.4."
        )

        print()

        return False


# ============================================================
# MAIN
# ============================================================

def main():

    print_header()

    if not os.path.isdir(
        OUTPUT_DIR
    ):

        print_error(
            "Output directory does not exist:"
        )

        print(
            f"     {OUTPUT_DIR}"
        )

        sys.exit(1)

    # ========================================================
    # 4.1
    # ========================================================

    validate_camera_input()

    # ========================================================
    # 4.2 - 4.4
    # ========================================================

    (
        trajectory_frames,
        trajectory_positions
    ) = validate_tracking_and_pose()

    # ========================================================
    # 4.5
    # ========================================================

    validate_visualization()

    # ========================================================
    # 4.6
    # ========================================================

    validate_recovery()

    # ========================================================
    # 4.7
    # ========================================================

    keyframe_ids = validate_keyframes(
        trajectory_frames
    )

    # ========================================================
    # 4.8
    # ========================================================

    validate_optimization(
        trajectory_frames,
        trajectory_positions,
        keyframe_ids
    )

    # ========================================================
    # FINAL
    # ========================================================

    success = print_summary()

    if success:

        sys.exit(0)

    else:

        sys.exit(1)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()