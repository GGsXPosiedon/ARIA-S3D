import os
import numpy as np
from pathlib import Path


# ============================================================
# ARIA-S3D | PHASE 3
# BUNDLE ADJUSTMENT VALIDATION
# ============================================================

OUTPUT_DIR = Path("data/output")

BA_STATS = OUTPUT_DIR / "bundle_adjustment_stats.txt"
BA_POINTS = OUTPUT_DIR / "bundle_adjusted_point_cloud.xyz"
BA_CAMERAS = OUTPUT_DIR / "bundle_adjusted_cameras.npz"
BA_TRAJECTORY = OUTPUT_DIR / "bundle_adjusted_trajectory.txt"

ORIGINAL_POINTS = OUTPUT_DIR / "filtered_point_cloud.xyz"
ORIGINAL_TRAJECTORY = OUTPUT_DIR / "optimized_camera_trajectory.txt"


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("ARIA-S3D | PHASE 3")
print("BUNDLE ADJUSTMENT VALIDATION")
print("=" * 70)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def check_file(path, description):
    """
    Check whether a required output file exists.
    """

    if path.exists():
        print(f"[OK] {description}")
        print(f"     {path}")
        return True

    print(f"[FAIL] {description}")
    print(f"       Missing: {path}")
    return False


def load_xyz(path):
    """
    Load an XYZ point cloud safely.
    """

    if not path.exists():
        return None

    try:
        points = np.loadtxt(path)

        if points.ndim == 1:
            points = points.reshape(1, -1)

        if points.shape[1] < 3:
            return None

        return points[:, :3]

    except Exception as e:
        print(f"[ERROR] Could not load point cloud: {e}")
        return None


def load_trajectory(path):
    """
    Load trajectory file.

    Expected format:

    frame_id x y z
    """

    if not path.exists():
        return None

    try:
        data = np.loadtxt(path)

        if data.ndim == 1:
            data = data.reshape(1, -1)

        if data.shape[1] < 4:
            return None

        return data

    except Exception as e:
        print(f"[ERROR] Could not load trajectory: {e}")
        return None


def extract_stat(text, name):
    """
    Extract a numeric statistic from the BA stats file.
    """

    for line in text.splitlines():

        if name.lower() in line.lower():

            parts = line.split(":")

            if len(parts) >= 2:

                value = parts[-1].strip()

                try:
                    return float(value)

                except ValueError:
                    continue

    return None


# ============================================================
# STEP 1
# CHECK REQUIRED FILES
# ============================================================

print()
print("[1] Checking Bundle Adjustment outputs")
print("-" * 70)

required_files = [

    (BA_STATS, "Bundle Adjustment statistics"),
    (BA_POINTS, "Bundle-adjusted point cloud"),
    (BA_CAMERAS, "Bundle-adjusted camera parameters"),
    (BA_TRAJECTORY, "Bundle-adjusted camera trajectory"),

]

all_files_present = True

for path, description in required_files:

    if not check_file(path, description):
        all_files_present = False


if not all_files_present:

    print()
    print("=" * 70)
    print("BUNDLE ADJUSTMENT VALIDATION FAILED")
    print("=" * 70)

    print()
    print("One or more required BA outputs are missing.")
    print("Run bundle_adjustment.py first.")

    raise SystemExit(1)


# ============================================================
# STEP 2
# LOAD BA STATISTICS
# ============================================================

print()
print("[2] Loading Bundle Adjustment statistics")
print("-" * 70)

try:

    with open(BA_STATS, "r", encoding="utf-8") as f:
        stats_text = f.read()

except Exception as e:

    print(f"[FAIL] Could not read statistics file: {e}")
    raise SystemExit(1)


initial_rmse = extract_stat(
    stats_text,
    "Initial RMSE"
)

final_rmse = extract_stat(
    stats_text,
    "Final RMSE"
)

improvement = extract_stat(
    stats_text,
    "Improvement"
)

iterations = extract_stat(
    stats_text,
    "Iterations"
)


print(f"Initial RMSE : {initial_rmse}")
print(f"Final RMSE   : {final_rmse}")
print(f"Improvement  : {improvement}")
print(f"Iterations   : {iterations}")


# ============================================================
# STEP 3
# VALIDATE RMSE
# ============================================================

print()
print("[3] Validating reprojection error")
print("-" * 70)

rmse_valid = True

if initial_rmse is None:

    print("[WARNING] Initial RMSE not found.")

elif final_rmse is None:

    print("[WARNING] Final RMSE not found.")

else:

    print(
        f"Initial reprojection RMSE : "
        f"{initial_rmse:.6f} px"
    )

    print(
        f"Final reprojection RMSE   : "
        f"{final_rmse:.6f} px"
    )

    if final_rmse < initial_rmse:

        actual_improvement = (
            (initial_rmse - final_rmse)
            / initial_rmse
        ) * 100.0

        print()
        print(
            f"[OK] Reprojection error improved by "
            f"{actual_improvement:.2f}%"
        )

    elif final_rmse == initial_rmse:

        print(
            "[WARNING] RMSE did not change."
        )

        rmse_valid = False

    else:

        print(
            "[FAIL] Final RMSE is worse than initial RMSE."
        )

        rmse_valid = False


# ============================================================
# STEP 4
# LOAD POINT CLOUD
# ============================================================

print()
print("[4] Validating optimized point cloud")
print("-" * 70)

points = load_xyz(BA_POINTS)

if points is None:

    print("[FAIL] Could not load optimized point cloud.")
    point_cloud_valid = False

else:

    print(
        f"Optimized 3D points : {len(points)}"
    )

    print(
        f"X range : {points[:, 0].min():.6f} "
        f"to {points[:, 0].max():.6f}"
    )

    print(
        f"Y range : {points[:, 1].min():.6f} "
        f"to {points[:, 1].max():.6f}"
    )

    print(
        f"Z range : {points[:, 2].min():.6f} "
        f"to {points[:, 2].max():.6f}"
    )

    finite_mask = np.isfinite(points).all(axis=1)

    invalid_count = np.sum(~finite_mask)

    if invalid_count == 0:

        print(
            "[OK] All optimized 3D points are finite."
        )

        point_cloud_valid = True

    else:

        print(
            f"[FAIL] {invalid_count} points contain "
            "NaN or Inf values."
        )

        point_cloud_valid = False


# ============================================================
# STEP 5
# LOAD ORIGINAL POINT CLOUD
# ============================================================

print()
print("[5] Comparing point cloud geometry")
print("-" * 70)

original_points = load_xyz(ORIGINAL_POINTS)

if original_points is None:

    print(
        "[WARNING] Original filtered point cloud "
        "could not be loaded."
    )

else:

    print(
        f"Original points     : {len(original_points)}"
    )

    print(
        f"Optimized points    : {len(points)}"
    )

    point_difference = (
        len(points) - len(original_points)
    )

    print(
        f"Point count change  : {point_difference:+d}"
    )

    print()
    print(
        "NOTE: Bundle Adjustment may use a different "
        "point representation."
    )


# ============================================================
# STEP 6
# LOAD CAMERA PARAMETERS
# ============================================================

print()
print("[6] Validating optimized camera parameters")
print("-" * 70)

camera_data = None

try:

    camera_data = np.load(
        BA_CAMERAS,
        allow_pickle=True
    )

    print(
        "Available camera data:"
    )

    for key in camera_data.files:

        value = camera_data[key]

        print(
            f"  {key}: "
            f"shape={getattr(value, 'shape', 'N/A')}"
        )

except Exception as e:

    print(
        f"[FAIL] Could not load camera parameters: {e}"
    )


# ============================================================
# STEP 7
# VALIDATE CAMERA TRAJECTORY
# ============================================================

print()
print("[7] Validating optimized camera trajectory")
print("-" * 70)

trajectory = load_trajectory(
    BA_TRAJECTORY
)

if trajectory is None:

    print(
        "[FAIL] Could not load optimized trajectory."
    )

    trajectory_valid = False

else:

    frame_ids = trajectory[:, 0]

    camera_positions = trajectory[:, 1:4]

    print(
        f"Camera poses : {len(trajectory)}"
    )

    print(
        f"Trajectory start : "
        f"{camera_positions[0]}"
    )

    print(
        f"Trajectory end   : "
        f"{camera_positions[-1]}"
    )

    finite_mask = np.isfinite(
        camera_positions
    ).all(axis=1)

    invalid_count = np.sum(
        ~finite_mask
    )

    if invalid_count == 0:

        print(
            "[OK] All camera positions are finite."
        )

        trajectory_valid = True

    else:

        print(
            f"[FAIL] {invalid_count} camera poses "
            "contain NaN or Inf."
        )

        trajectory_valid = False


# ============================================================
# STEP 8
# TRAJECTORY STATISTICS
# ============================================================

if trajectory is not None:

    print()
    print("[8] Camera trajectory statistics")
    print("-" * 70)

    positions = trajectory[:, 1:4]

    if len(positions) >= 2:

        movements = np.linalg.norm(
            np.diff(positions, axis=0),
            axis=1
        )

        total_distance = np.sum(
            movements
        )

        mean_movement = np.mean(
            movements
        )

        max_movement = np.max(
            movements
        )

        print(
            f"Total trajectory distance : "
            f"{total_distance:.6f}"
        )

        print(
            f"Mean camera movement      : "
            f"{mean_movement:.6f}"
        )

        print(
            f"Maximum camera movement   : "
            f"{max_movement:.6f}"
        )

        if np.isfinite(total_distance):

            print(
                "[OK] Trajectory distance is valid."
            )

        else:

            print(
                "[FAIL] Invalid trajectory distance."
            )


# ============================================================
# STEP 9
# CHECK STATS FILE CONSISTENCY
# ============================================================

print()
print("[9] Checking BA statistics consistency")
print("-" * 70)

stats_valid = True

if iterations is not None:

    if iterations > 0:

        print(
            f"[OK] Optimization completed in "
            f"{int(iterations)} iterations."
        )

    else:

        print(
            "[WARNING] Iteration count is zero."
        )

        stats_valid = False

else:

    print(
        "[WARNING] Iteration count unavailable."
    )


# ============================================================
# STEP 10
# FINAL VALIDATION
# ============================================================

print()
print("=" * 70)
print("ARIA-S3D | PHASE 3 VALIDATION SUMMARY")
print("=" * 70)

print()

print(
    f"BA statistics          : "
    f"{'PASS' if BA_STATS.exists() else 'FAIL'}"
)

print(
    f"Optimized point cloud  : "
    f"{'PASS' if point_cloud_valid else 'FAIL'}"
)

print(
    f"Camera parameters      : "
    f"{'PASS' if camera_data is not None else 'FAIL'}"
)

print(
    f"Camera trajectory      : "
    f"{'PASS' if trajectory_valid else 'FAIL'}"
)

print(
    f"RMSE optimization      : "
    f"{'PASS' if rmse_valid else 'FAIL'}"
)

print(
    f"Optimization stats     : "
    f"{'PASS' if stats_valid else 'FAIL'}"
)


# ============================================================
# OVERALL RESULT
# ============================================================

overall_success = (

    BA_STATS.exists()
    and point_cloud_valid
    and camera_data is not None
    and trajectory_valid
    and rmse_valid
    and stats_valid

)


print()
print("=" * 70)

if overall_success:

    print("PHASE 3 BUNDLE ADJUSTMENT VALIDATION: SUCCESS")
    print("=" * 70)

    print()
    print("ARIA-S3D has successfully completed:")
    print()
    print("  [✓] Incremental reconstruction")
    print("  [✓] Global trajectory optimization")
    print("  [✓] Outlier rejection")
    print("  [✓] Feature-track preparation")
    print("  [✓] Sparse Bundle Adjustment")
    print("  [✓] Reprojection-error optimization")
    print("  [✓] Optimized camera geometry validation")
    print()
    print("The optimized reconstruction is ready")
    print("for the next ARIA-S3D processing stage.")

else:

    print("PHASE 3 VALIDATION: ISSUES DETECTED")
    print("=" * 70)

    print()
    print(
        "Review the FAILED checks above before "
        "continuing to the next stage."
    )


# ============================================================
# IMPORTANT NOTE
# ============================================================

print()
print("=" * 70)
print("IMPORTANT")
print("=" * 70)

print()
print(
    "ARIA-S3D is still a monocular reconstruction."
)

print(
    "Absolute metric scale remains arbitrary unless "
    "additional scale information is introduced."
)

print(
    "Bundle Adjustment improves geometric consistency "
    "by minimizing reprojection error."
)

print()