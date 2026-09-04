"""
ARIA-S3D | PHASE 4.8
REAL-TIME OPTIMIZATION

Uses:
    - Real-time camera trajectory
    - Keyframe constraints
    - Trajectory smoothing
    - Velocity consistency
    - Motion outlier handling

Inputs:
    data/output/realtime_pose_trajectory.txt
    data/output/keyframe_metadata.txt

Outputs:
    data/output/optimized_realtime_trajectory.txt
    data/output/realtime_optimization_stats.txt
"""

import os
import time
import re
import numpy as np


# ============================================================
# PATHS
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

POSE_FILE = os.path.join(
    OUTPUT_DIR,
    "realtime_pose_trajectory.txt"
)

KEYFRAME_FILE = os.path.join(
    OUTPUT_DIR,
    "keyframe_metadata.txt"
)

OPTIMIZED_TRAJECTORY_FILE = os.path.join(
    OUTPUT_DIR,
    "optimized_realtime_trajectory.txt"
)

STATS_FILE = os.path.join(
    OUTPUT_DIR,
    "realtime_optimization_stats.txt"
)


# ============================================================
# OPTIMIZATION PARAMETERS
# ============================================================

SMOOTHING_WEIGHT = 0.20
VELOCITY_WEIGHT = 0.35
KEYFRAME_WEIGHT = 0.75

OUTLIER_THRESHOLD = 3.0

MAX_ITERATIONS = 10


# ============================================================
# HEADER
# ============================================================

def print_header():

    print("=" * 70)
    print("ARIA-S3D | PHASE 4.8")
    print("REAL-TIME OPTIMIZATION")
    print("=" * 70)
    print()


# ============================================================
# LOAD TRAJECTORY
# ============================================================

def load_trajectory(path):

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Trajectory file not found:\n{path}"
        )

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

            if len(parts) < 3:
                continue

            try:

                values = [
                    float(value)
                    for value in parts
                ]

            except ValueError:

                continue

            # ------------------------------------------------
            # frame x y z rx ry rz
            # ------------------------------------------------

            if len(values) >= 7:

                frame = int(values[0])

                position = np.array(
                    values[1:4],
                    dtype=np.float64
                )

                rotation = np.array(
                    values[4:7],
                    dtype=np.float64
                )

                data.append(
                    (
                        frame,
                        position,
                        rotation
                    )
                )

            # ------------------------------------------------
            # frame x y z
            # ------------------------------------------------

            elif len(values) >= 4:

                frame = int(values[0])

                position = np.array(
                    values[1:4],
                    dtype=np.float64
                )

                rotation = np.zeros(
                    3,
                    dtype=np.float64
                )

                data.append(
                    (
                        frame,
                        position,
                        rotation
                    )
                )

            # ------------------------------------------------
            # x y z
            # ------------------------------------------------

            else:

                frame = len(data)

                position = np.array(
                    values[:3],
                    dtype=np.float64
                )

                rotation = np.zeros(
                    3,
                    dtype=np.float64
                )

                data.append(
                    (
                        frame,
                        position,
                        rotation
                    )
                )

    if len(data) < 3:

        raise RuntimeError(
            "Not enough trajectory points "
            "for optimization."
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

    rotations = np.array(
        [
            item[2]
            for item in data
        ],
        dtype=np.float64
    )

    return (
        frames,
        positions,
        rotations
    )


# ============================================================
# LOAD KEYFRAME METADATA
# ============================================================

def load_keyframes(path):

    """
    Parses the actual ARIA-S3D keyframe metadata format:

        Frame | Filename | Quality | Reason | Matches | Overlap
        -------------------------------------------------------
             1 | keyframe_000001.jpg | ...
            61 | keyframe_000061.jpg | ...
           121 | keyframe_000121.jpg | ...

    The first column contains the frame ID.
    """

    keyframes = []

    if not os.path.exists(path):

        print(
            "[WARNING] Keyframe metadata not found."
        )

        return keyframes

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        for line_number, line in enumerate(
            f,
            start=1
        ):

            stripped = line.strip()

            # Ignore empty lines
            if not stripped:
                continue

            # Ignore separator
            if set(stripped) <= set("-= "):

                continue

            # Ignore header
            if stripped.lower().startswith(
                "frame"
            ):

                continue

            # ------------------------------------------------
            # Split table row
            # ------------------------------------------------

            if "|" in stripped:

                columns = [
                    column.strip()
                    for column in stripped.split("|")
                ]

                if len(columns) < 2:
                    continue

                first_column = columns[0]

                # ------------------------------------------------
                # First column = frame ID
                # ------------------------------------------------

                match = re.search(
                    r"\d+",
                    first_column
                )

                if match:

                    frame_id = int(
                        match.group()
                    )

                    if frame_id not in keyframes:

                        keyframes.append(
                            frame_id
                        )

    keyframes.sort()

    # ========================================================
    # RESULT
    # ========================================================

    if len(keyframes) == 0:

        print(
            "[WARNING] No keyframe IDs could be parsed."
        )

    else:

        print(
            f"[OK] Parsed {len(keyframes)} "
            f"keyframe IDs from metadata."
        )

    return keyframes


# ============================================================
# MAP KEYFRAME IDS TO TRAJECTORY INDICES
# ============================================================

def get_keyframe_indices(
    frames,
    keyframe_ids
):

    if not keyframe_ids:

        return []

    frame_to_index = {
        int(frame): index
        for index, frame in enumerate(frames)
    }

    indices = []

    unmatched = []

    for frame_id in keyframe_ids:

        frame_id = int(frame_id)

        if frame_id in frame_to_index:

            indices.append(
                frame_to_index[frame_id]
            )

        else:

            unmatched.append(
                frame_id
            )

    indices = sorted(
        list(
            set(indices)
        )
    )

    if unmatched:

        print(
            "[WARNING] Keyframes not found "
            "in trajectory:"
        )

        print(
            f"          {unmatched}"
        )

    if indices:

        print(
            f"[OK] Mapped {len(indices)} keyframes "
            "to trajectory positions."
        )

    return indices


# ============================================================
# TRAJECTORY DISTANCE
# ============================================================

def trajectory_distance(points):

    if len(points) < 2:

        return 0.0

    differences = np.diff(
        points,
        axis=0
    )

    distances = np.linalg.norm(
        differences,
        axis=1
    )

    return float(
        np.sum(distances)
    )


# ============================================================
# MEAN MOTION
# ============================================================

def mean_motion(points):

    if len(points) < 2:

        return 0.0

    differences = np.diff(
        points,
        axis=0
    )

    distances = np.linalg.norm(
        differences,
        axis=1
    )

    return float(
        np.mean(distances)
    )


# ============================================================
# MAXIMUM MOTION
# ============================================================

def maximum_motion(points):

    if len(points) < 2:

        return 0.0

    differences = np.diff(
        points,
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

def trajectory_residual(points):

    if len(points) < 3:

        return 0.0

    residuals = []

    for i in range(
        1,
        len(points) - 1
    ):

        prediction = (
            points[i - 1]
            +
            points[i + 1]
        ) / 2.0

        error = np.linalg.norm(
            points[i]
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
# MOTION OUTLIERS
# ============================================================

def detect_motion_outliers(points):

    if len(points) < 4:

        return np.zeros(
            len(points),
            dtype=bool
        )

    differences = np.diff(
        points,
        axis=0
    )

    motion = np.linalg.norm(
        differences,
        axis=1
    )

    median_motion = np.median(
        motion
    )

    mad = np.median(
        np.abs(
            motion
            -
            median_motion
        )
    )

    if mad < 1e-9:

        return np.zeros(
            len(points),
            dtype=bool
        )

    robust_score = (
        np.abs(
            motion
            -
            median_motion
        )
        /
        (
            1.4826 * mad
        )
    )

    outliers = (
        robust_score
        >
        OUTLIER_THRESHOLD
    )

    result = np.zeros(
        len(points),
        dtype=bool
    )

    result[1:] = outliers

    return result


# ============================================================
# OPTIMIZATION PASS
# ============================================================

def optimization_pass(
    original,
    current,
    keyframe_indices,
    outlier_mask
):

    optimized = current.copy()

    n = len(current)

    keyframe_set = set(
        keyframe_indices
    )

    # --------------------------------------------------------
    # Optimize normal trajectory points
    # --------------------------------------------------------

    for i in range(
        1,
        n - 1
    ):

        # Keyframes handled separately
        if i in keyframe_set:

            continue

        previous = current[i - 1]

        current_point = current[i]

        next_point = current[i + 1]

        # ----------------------------------------------------
        # Smoothing prediction
        # ----------------------------------------------------

        smooth_prediction = (
            previous
            +
            next_point
        ) / 2.0

        # ----------------------------------------------------
        # Velocity prediction
        # ----------------------------------------------------

        velocity_prediction = (
            2.0
            *
            current_point
            -
            previous
        )

        # ----------------------------------------------------
        # Combined prediction
        # ----------------------------------------------------

        predicted = (
            (
                1.0
                -
                VELOCITY_WEIGHT
            )
            *
            smooth_prediction
            +
            VELOCITY_WEIGHT
            *
            velocity_prediction
        )

        # ----------------------------------------------------
        # Outlier handling
        # ----------------------------------------------------

        if outlier_mask[i]:

            optimized[i] = (
                0.15
                *
                original[i]
                +
                0.85
                *
                predicted
            )

        else:

            optimized[i] = (
                (
                    1.0
                    -
                    SMOOTHING_WEIGHT
                )
                *
                current_point
                +
                SMOOTHING_WEIGHT
                *
                predicted
            )

    # ========================================================
    # KEYFRAME CONSTRAINTS
    # ========================================================

    for index in keyframe_indices:

        if (
            0 <= index < n
        ):

            optimized[index] = (
                KEYFRAME_WEIGHT
                *
                original[index]
                +
                (
                    1.0
                    -
                    KEYFRAME_WEIGHT
                )
                *
                current[index]
            )

    # ========================================================
    # PRESERVE START AND END
    # ========================================================

    optimized[0] = original[0]

    optimized[-1] = original[-1]

    return optimized


# ============================================================
# SAVE TRAJECTORY
# ============================================================

def save_trajectory(
    path,
    frames,
    positions,
    rotations
):

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "# ARIA-S3D optimized real-time trajectory\n"
        )

        f.write(
            "# frame x y z rx ry rz\n"
        )

        for i in range(
            len(frames)
        ):

            f.write(
                f"{int(frames[i]):06d} "
                f"{positions[i, 0]:.9f} "
                f"{positions[i, 1]:.9f} "
                f"{positions[i, 2]:.9f} "
                f"{rotations[i, 0]:.9f} "
                f"{rotations[i, 1]:.9f} "
                f"{rotations[i, 2]:.9f}\n"
            )


# ============================================================
# SAVE STATISTICS
# ============================================================

def save_statistics(
    path,
    frames_processed,
    keyframes,
    outliers,
    original_distance,
    optimized_distance,
    original_motion,
    optimized_motion,
    original_max_motion,
    optimized_max_motion,
    original_residual,
    optimized_residual,
    iterations,
    processing_time
):

    if original_residual > 1e-9:

        improvement = (
            (
                original_residual
                -
                optimized_residual
            )
            /
            original_residual
        ) * 100.0

    else:

        improvement = 0.0

    if processing_time > 1e-9:

        fps = (
            frames_processed
            /
            processing_time
        )

    else:

        fps = 0.0

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "ARIA-S3D | PHASE 4.8 "
            "REAL-TIME OPTIMIZATION\n"
        )

        f.write(
            "=" * 60
            +
            "\n\n"
        )

        f.write(
            f"Frames processed : "
            f"{frames_processed}\n"
        )

        f.write(
            f"Keyframes used : "
            f"{len(keyframes)}\n"
        )

        f.write(
            f"Keyframe IDs : "
            f"{keyframes}\n"
        )

        f.write(
            f"Motion outliers : "
            f"{outliers}\n"
        )

        f.write(
            f"Iterations : "
            f"{iterations}\n"
        )

        f.write("\n")

        f.write(
            f"Original trajectory distance : "
            f"{original_distance:.6f}\n"
        )

        f.write(
            f"Optimized trajectory distance : "
            f"{optimized_distance:.6f}\n"
        )

        f.write(
            f"Original mean motion : "
            f"{original_motion:.6f}\n"
        )

        f.write(
            f"Optimized mean motion : "
            f"{optimized_motion:.6f}\n"
        )

        f.write(
            f"Original maximum motion : "
            f"{original_max_motion:.6f}\n"
        )

        f.write(
            f"Optimized maximum motion : "
            f"{optimized_max_motion:.6f}\n"
        )

        f.write("\n")

        f.write(
            f"Original residual : "
            f"{original_residual:.6f}\n"
        )

        f.write(
            f"Optimized residual : "
            f"{optimized_residual:.6f}\n"
        )

        f.write(
            f"Residual improvement : "
            f"{improvement:.2f}%\n"
        )

        f.write("\n")

        f.write(
            f"Processing time : "
            f"{processing_time:.6f} sec\n"
        )

        f.write(
            f"Processing FPS : "
            f"{fps:.2f}\n"
        )

        f.write("\n")

        f.write(
            "Optimization parameters:\n"
        )

        f.write(
            f"Smoothing weight : "
            f"{SMOOTHING_WEIGHT}\n"
        )

        f.write(
            f"Velocity weight : "
            f"{VELOCITY_WEIGHT}\n"
        )

        f.write(
            f"Keyframe weight : "
            f"{KEYFRAME_WEIGHT}\n"
        )

        f.write(
            f"Outlier threshold : "
            f"{OUTLIER_THRESHOLD}\n"
        )

        f.write(
            f"Maximum iterations : "
            f"{MAX_ITERATIONS}\n"
        )

        f.write("\n")

        f.write(
            "NOTE:\n"
        )

        f.write(
            "This stage performs real-time "
            "trajectory optimization.\n"
        )

        f.write(
            "It does not perform full Bundle Adjustment.\n"
        )

        f.write(
            "Monocular scale remains arbitrary.\n"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print_header()

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # ========================================================
    # 1. CHECK INPUTS
    # ========================================================

    print(
        "[1] Checking Phase 4 outputs"
    )

    print(
        "-" * 70
    )

    if not os.path.exists(
        POSE_FILE
    ):

        print(
            "[ERROR] Real-time pose trajectory "
            "not found:"
        )

        print(
            f"        {POSE_FILE}"
        )

        return

    print(
        "[OK] Pose trajectory:"
    )

    print(
        f"     {POSE_FILE}"
    )

    if os.path.exists(
        KEYFRAME_FILE
    ):

        print(
            "[OK] Keyframe metadata:"
        )

        print(
            f"     {KEYFRAME_FILE}"
        )

    else:

        print(
            "[WARNING] Keyframe metadata unavailable."
        )

    print()

    # ========================================================
    # 2. LOAD TRAJECTORY
    # ========================================================

    print(
        "[2] Loading real-time trajectory"
    )

    print(
        "-" * 70
    )

    (
        frames,
        positions,
        rotations
    ) = load_trajectory(
        POSE_FILE
    )

    print(
        f"Trajectory frames : "
        f"{len(frames)}"
    )

    print(
        f"Start position    : "
        f"{positions[0]}"
    )

    print(
        f"End position      : "
        f"{positions[-1]}"
    )

    print()

    # ========================================================
    # 3. KEYFRAME CONSTRAINTS
    # ========================================================

    print(
        "[3] Loading keyframe constraints"
    )

    print(
        "-" * 70
    )

    keyframe_ids = load_keyframes(
        KEYFRAME_FILE
    )

    keyframe_indices = get_keyframe_indices(
        frames,
        keyframe_ids
    )

    print()

    print(
        f"Keyframe IDs found : "
        f"{len(keyframe_ids)}"
    )

    print(
        f"Keyframe positions : "
        f"{len(keyframe_indices)}"
    )

    if keyframe_ids:

        print(
            f"Keyframe IDs       : "
            f"{keyframe_ids}"
        )

    print()

    # ========================================================
    # 4. INITIAL STATISTICS
    # ========================================================

    print(
        "[4] Computing initial trajectory statistics"
    )

    print(
        "-" * 70
    )

    original_distance = trajectory_distance(
        positions
    )

    original_motion = mean_motion(
        positions
    )

    original_max_motion = maximum_motion(
        positions
    )

    original_residual = trajectory_residual(
        positions
    )

    print(
        f"Trajectory distance : "
        f"{original_distance:.6f}"
    )

    print(
        f"Mean motion         : "
        f"{original_motion:.6f}"
    )

    print(
        f"Maximum motion      : "
        f"{original_max_motion:.6f}"
    )

    print(
        f"Trajectory residual : "
        f"{original_residual:.6f}"
    )

    print()

    # ========================================================
    # 5. OUTLIERS
    # ========================================================

    print(
        "[5] Detecting trajectory outliers"
    )

    print(
        "-" * 70
    )

    outlier_mask = detect_motion_outliers(
        positions
    )

    outlier_count = int(
        np.sum(
            outlier_mask
        )
    )

    print(
        f"Motion outliers detected : "
        f"{outlier_count}"
    )

    if outlier_count > 0:

        print(
            "[OK] Robust optimization will "
            "handle motion outliers."
        )

    else:

        print(
            "[OK] No significant trajectory "
            "outliers detected."
        )

    print()

    # ========================================================
    # 6. OPTIMIZATION
    # ========================================================

    print(
        "[6] Running real-time optimization"
    )

    print(
        "-" * 70
    )

    print(
        f"Maximum iterations : "
        f"{MAX_ITERATIONS}"
    )

    print(
        f"Smoothing weight   : "
        f"{SMOOTHING_WEIGHT}"
    )

    print(
        f"Velocity weight    : "
        f"{VELOCITY_WEIGHT}"
    )

    print(
        f"Keyframe weight    : "
        f"{KEYFRAME_WEIGHT}"
    )

    print()

    start_time = time.perf_counter()

    optimized_positions = positions.copy()

    for iteration in range(
        MAX_ITERATIONS
    ):

        optimized_positions = optimization_pass(
            positions,
            optimized_positions,
            keyframe_indices,
            outlier_mask
        )

        residual = trajectory_residual(
            optimized_positions
        )

        print(
            f"Iteration {iteration + 1:02d} "
            f"| trajectory residual = "
            f"{residual:.6f}"
        )

    processing_time = (
        time.perf_counter()
        -
        start_time
    )

    print()

    # ========================================================
    # 7. RESULTS
    # ========================================================

    optimized_distance = trajectory_distance(
        optimized_positions
    )

    optimized_motion = mean_motion(
        optimized_positions
    )

    optimized_max_motion = maximum_motion(
        optimized_positions
    )

    optimized_residual = trajectory_residual(
        optimized_positions
    )

    if original_residual > 1e-9:

        improvement = (
            (
                original_residual
                -
                optimized_residual
            )
            /
            original_residual
        ) * 100.0

    else:

        improvement = 0.0

    print(
        "[7] Optimization results"
    )

    print(
        "-" * 70
    )

    print(
        f"Original residual  : "
        f"{original_residual:.6f}"
    )

    print(
        f"Optimized residual : "
        f"{optimized_residual:.6f}"
    )

    print(
        f"Residual improvement : "
        f"{improvement:.2f}%"
    )

    print()

    print(
        f"Optimized trajectory distance : "
        f"{optimized_distance:.6f}"
    )

    print(
        f"Optimized mean motion         : "
        f"{optimized_motion:.6f}"
    )

    print(
        f"Optimized maximum motion      : "
        f"{optimized_max_motion:.6f}"
    )

    print()

    # ========================================================
    # 8. SAVE TRAJECTORY
    # ========================================================

    print(
        "[8] Saving optimized trajectory"
    )

    print(
        "-" * 70
    )

    save_trajectory(
        OPTIMIZED_TRAJECTORY_FILE,
        frames,
        optimized_positions,
        rotations
    )

    print(
        "[OK] Optimized trajectory saved:"
    )

    print(
        f"     {OPTIMIZED_TRAJECTORY_FILE}"
    )

    # ========================================================
    # SAVE STATS
    # ========================================================

    save_statistics(
        STATS_FILE,
        len(frames),
        keyframe_ids,
        outlier_count,
        original_distance,
        optimized_distance,
        original_motion,
        optimized_motion,
        original_max_motion,
        optimized_max_motion,
        original_residual,
        optimized_residual,
        MAX_ITERATIONS,
        processing_time
    )

    print(
        "[OK] Optimization statistics saved:"
    )

    print(
        f"     {STATS_FILE}"
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()

    print("=" * 70)

    print(
        "ARIA-S3D | PHASE 4.8 COMPLETE"
    )

    print("=" * 70)

    print()

    print(
        f"Frames optimized     : "
        f"{len(frames)}"
    )

    print(
        f"Keyframes constrained: "
        f"{len(keyframe_indices)}"
    )

    print(
        f"Outliers handled     : "
        f"{outlier_count}"
    )

    print(
        f"Iterations           : "
        f"{MAX_ITERATIONS}"
    )

    print(
        f"Residual improvement : "
        f"{improvement:.2f}%"
    )

    if processing_time > 1e-9:

        processing_fps = (
            len(frames)
            /
            processing_time
        )

    else:

        processing_fps = 0.0

    print(
        f"Processing FPS       : "
        f"{processing_fps:.2f}"
    )

    print()

    print(
        "Generated outputs:"
    )

    print(
        f"  {OPTIMIZED_TRAJECTORY_FILE}"
    )

    print(
        f"  {STATS_FILE}"
    )

    print()

    print(
        "-" * 70
    )

    print(
        "NEXT STEP:"
    )

    print(
        "Phase 4.9 - Phase 4 validation"
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "This remains a monocular reconstruction."
    )

    print(
        "Absolute metric scale remains arbitrary."
    )

    print(
        "Keyframe constraints are now explicitly "
        "included in trajectory optimization."
    )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()