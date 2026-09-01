"""
ARIA-S3D
Global Camera Trajectory Optimization

Phase 3:
- Loads the globally reconstructed camera trajectory
- Removes obviously unstable trajectory jumps
- Applies robust smoothing
- Produces an optimized global camera trajectory

NOTE:
This is a monocular reconstruction, so absolute scale remains unknown.
"""

import os
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_TRAJECTORY = "data/output/global_camera_trajectory.txt"
OUTPUT_TRAJECTORY = "data/output/optimized_camera_trajectory.txt"

# Maximum allowed camera movement between consecutive frames.
# This is in the arbitrary reconstruction scale.
MAX_STEP = 5.0

# Smoothing strength
SMOOTHING_WINDOW = 5


# ============================================================
# LOAD TRAJECTORY
# ============================================================

def load_trajectory(path):
    """
    Load camera trajectory.

    Expected format:
        frame_id x y z

    If the trajectory contains additional values,
    the first four numeric values are used.
    """

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Trajectory file not found: {path}"
        )

    data = []

    with open(path, "r") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            # Skip comments
            if line.startswith("#"):
                continue

            parts = line.split()

            try:
                values = [float(x) for x in parts]
            except ValueError:
                continue

            if len(values) >= 4:
                frame_id = int(values[0])
                x, y, z = values[1:4]

                data.append(
                    [frame_id, x, y, z]
                )

    if len(data) < 3:
        raise ValueError(
            "Not enough camera poses found."
        )

    return np.array(data, dtype=float)


# ============================================================
# REMOVE LARGE TRAJECTORY JUMPS
# ============================================================

def remove_trajectory_jumps(trajectory):
    """
    Detect unrealistic camera jumps between consecutive poses.
    """

    optimized = [trajectory[0]]

    removed = 0

    for i in range(1, len(trajectory)):

        previous = optimized[-1]
        current = trajectory[i]

        previous_position = previous[1:4]
        current_position = current[1:4]

        distance = np.linalg.norm(
            current_position - previous_position
        )

        if distance <= MAX_STEP:

            optimized.append(current)

        else:

            removed += 1

    optimized = np.array(optimized)

    return optimized, removed


# ============================================================
# MOVING AVERAGE SMOOTHING
# ============================================================

def smooth_trajectory(trajectory, window=5):
    """
    Smooth camera positions using a moving average.

    Frame IDs are kept unchanged.
    """

    result = trajectory.copy()

    positions = trajectory[:, 1:4]

    half_window = window // 2

    for i in range(len(positions)):

        start = max(0, i - half_window)
        end = min(
            len(positions),
            i + half_window + 1
        )

        result[i, 1:4] = np.mean(
            positions[start:end],
            axis=0
        )

    return result


# ============================================================
# COMPUTE TRAJECTORY STATISTICS
# ============================================================

def trajectory_statistics(trajectory):

    positions = trajectory[:, 1:4]

    distances = []

    for i in range(1, len(positions)):

        distance = np.linalg.norm(
            positions[i] - positions[i - 1]
        )

        distances.append(distance)

    distances = np.array(distances)

    total_distance = np.sum(distances)

    mean_step = np.mean(distances)

    max_step = np.max(distances)

    return (
        total_distance,
        mean_step,
        max_step
    )


# ============================================================
# SAVE TRAJECTORY
# ============================================================

def save_trajectory(path, trajectory):

    os.makedirs(
        os.path.dirname(path),
        exist_ok=True
    )

    with open(path, "w") as f:

        f.write(
            "# ARIA-S3D Optimized Global Camera Trajectory\n"
        )

        f.write(
            "# frame_id x y z\n"
        )

        for row in trajectory:

            frame_id = int(row[0])

            x = row[1]
            y = row[2]
            z = row[3]

            f.write(
                f"{frame_id} "
                f"{x:.8f} "
                f"{y:.8f} "
                f"{z:.8f}\n"
            )


# ============================================================
# MAIN OPTIMIZATION PIPELINE
# ============================================================

def main():

    print("=" * 60)
    print("ARIA-S3D | GLOBAL CAMERA TRAJECTORY OPTIMIZATION")
    print("=" * 60)

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    print("\nLoading camera trajectory:")
    print(INPUT_TRAJECTORY)

    trajectory = load_trajectory(
        INPUT_TRAJECTORY
    )

    print(
        f"Camera poses loaded: {len(trajectory)}"
    )

    # --------------------------------------------------------
    # ORIGINAL STATISTICS
    # --------------------------------------------------------

    (
        original_total,
        original_mean,
        original_max
    ) = trajectory_statistics(
        trajectory
    )

    print("\nOriginal trajectory statistics:")

    print(
        f"Total camera movement: "
        f"{original_total:.4f}"
    )

    print(
        f"Mean frame movement: "
        f"{original_mean:.4f}"
    )

    print(
        f"Maximum frame movement: "
        f"{original_max:.4f}"
    )

    # --------------------------------------------------------
    # REMOVE JUMPS
    # --------------------------------------------------------

    print("\nDetecting unstable camera jumps...")

    filtered, removed = remove_trajectory_jumps(
        trajectory
    )

    print(
        f"Unstable poses removed: {removed}"
    )

    print(
        f"Remaining poses: {len(filtered)}"
    )

    # --------------------------------------------------------
    # SMOOTH TRAJECTORY
    # --------------------------------------------------------

    print("\nApplying global trajectory smoothing...")

    optimized = smooth_trajectory(
        filtered,
        SMOOTHING_WINDOW
    )

    # --------------------------------------------------------
    # OPTIMIZED STATISTICS
    # --------------------------------------------------------

    (
        optimized_total,
        optimized_mean,
        optimized_max
    ) = trajectory_statistics(
        optimized
    )

    print("\nOptimized trajectory statistics:")

    print(
        f"Total camera movement: "
        f"{optimized_total:.4f}"
    )

    print(
        f"Mean frame movement: "
        f"{optimized_mean:.4f}"
    )

    print(
        f"Maximum frame movement: "
        f"{optimized_max:.4f}"
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    save_trajectory(
        OUTPUT_TRAJECTORY,
        optimized
    )

    print("\nGLOBAL OPTIMIZATION COMPLETE")
    print("-" * 60)

    print(
        f"Optimized camera trajectory saved to:"
    )

    print(
        f"  {OUTPUT_TRAJECTORY}"
    )

    print("\nIMPORTANT:")
    print(
        "This optimization improves trajectory consistency "
        "but does not recover absolute metric scale."
    )

    print(
        "A future Bundle Adjustment stage can jointly optimize "
        "camera poses and 3D points."
    )

    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()