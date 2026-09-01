"""
ARIA-S3D | GLOBAL POINT CLOUD MERGING

Phase 2:
- Load incremental point cloud
- Merge nearby/duplicate points using voxel downsampling
- Remove extreme outliers
- Save global point cloud
- Preserve camera trajectory

Note:
Monocular reconstruction still has arbitrary translation scale.
"""

import os
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_CLOUD = "data/output/incremental_point_cloud.xyz"
INPUT_TRAJECTORY = "data/output/camera_trajectory.txt"

OUTPUT_CLOUD = "data/output/global_point_cloud.xyz"
OUTPUT_TRAJECTORY = "data/output/global_camera_trajectory.txt"

# Points inside the same voxel are merged
VOXEL_SIZE = 0.05

# Remove points that are extremely far from the cloud center
OUTLIER_STD = 3.0


# ============================================================
# LOAD POINT CLOUD
# ============================================================

def load_point_cloud(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Point cloud not found: {path}")

    points = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            values = line.split()

            if len(values) < 3:
                continue

            try:
                x, y, z = map(float, values[:3])
                points.append([x, y, z])
            except ValueError:
                continue

    if not points:
        raise ValueError("No valid 3D points found.")

    return np.asarray(points, dtype=np.float64)


# ============================================================
# VOXEL-BASED POINT MERGING
# ============================================================

def voxel_merge(points, voxel_size):
    """
    Merge points falling inside the same voxel.

    The representative point is the mean position
    of all points inside that voxel.
    """

    voxel_indices = np.floor(points / voxel_size).astype(np.int64)

    voxel_dict = {}

    for point, voxel in zip(points, voxel_indices):
        key = tuple(voxel)

        if key not in voxel_dict:
            voxel_dict[key] = []

        voxel_dict[key].append(point)

    merged_points = []

    for voxel_points in voxel_dict.values():
        voxel_points = np.asarray(voxel_points)
        merged_points.append(np.mean(voxel_points, axis=0))

    return np.asarray(merged_points)


# ============================================================
# OUTLIER REMOVAL
# ============================================================

def remove_outliers(points, std_multiplier=3.0):
    """
    Remove extreme points using distance from cloud centroid.
    """

    center = np.mean(points, axis=0)

    distances = np.linalg.norm(points - center, axis=1)

    mean_distance = np.mean(distances)
    std_distance = np.std(distances)

    threshold = mean_distance + std_multiplier * std_distance

    mask = distances <= threshold

    return points[mask]


# ============================================================
# SAVE POINT CLOUD
# ============================================================

def save_point_cloud(path, points):

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w") as f:

        for point in points:
            f.write(
                f"{point[0]:.8f} "
                f"{point[1]:.8f} "
                f"{point[2]:.8f}\n"
            )


# ============================================================
# COPY CAMERA TRAJECTORY
# ============================================================

def copy_trajectory(input_path, output_path):

    if not os.path.exists(input_path):
        print("WARNING: Camera trajectory not found.")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(input_path, "r") as src:
        data = src.read()

    with open(output_path, "w") as dst:
        dst.write(data)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)
    print("ARIA-S3D | GLOBAL POINT CLOUD MERGING")
    print("=" * 65)

    # --------------------------------------------------------
    # Load cloud
    # --------------------------------------------------------

    print(f"\nLoading point cloud:")
    print(f"  {INPUT_CLOUD}")

    points = load_point_cloud(INPUT_CLOUD)

    print(f"Original points: {len(points)}")

    # --------------------------------------------------------
    # Original statistics
    # --------------------------------------------------------

    print("\nOriginal cloud statistics:")

    print(f"X range: {points[:, 0].min():.4f} "
          f"to {points[:, 0].max():.4f}")

    print(f"Y range: {points[:, 1].min():.4f} "
          f"to {points[:, 1].max():.4f}")

    print(f"Z range: {points[:, 2].min():.4f} "
          f"to {points[:, 2].max():.4f}")

    # --------------------------------------------------------
    # Remove extreme outliers
    # --------------------------------------------------------

    print("\nRemoving extreme outliers...")

    filtered_points = remove_outliers(
        points,
        OUTLIER_STD
    )

    print(f"Points after outlier removal: "
          f"{len(filtered_points)}")

    # --------------------------------------------------------
    # Global voxel merge
    # --------------------------------------------------------

    print("\nPerforming global voxel merge...")
    print(f"Voxel size: {VOXEL_SIZE}")

    global_points = voxel_merge(
        filtered_points,
        VOXEL_SIZE
    )

    print(f"Points after global merging: "
          f"{len(global_points)}")

    # --------------------------------------------------------
    # Save global cloud
    # --------------------------------------------------------

    save_point_cloud(
        OUTPUT_CLOUD,
        global_points
    )

    print("\nGlobal point cloud saved to:")
    print(f"  {OUTPUT_CLOUD}")

    # --------------------------------------------------------
    # Preserve trajectory
    # --------------------------------------------------------

    copy_trajectory(
        INPUT_TRAJECTORY,
        OUTPUT_TRAJECTORY
    )

    print("\nCamera trajectory saved to:")
    print(f"  {OUTPUT_TRAJECTORY}")

    # --------------------------------------------------------
    # Final statistics
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("GLOBAL MERGE COMPLETE")
    print("=" * 65)

    print(f"Input points:       {len(points)}")
    print(f"After filtering:    {len(filtered_points)}")
    print(f"Final global cloud: {len(global_points)}")

    reduction = (
        1 - len(global_points) / len(points)
    ) * 100

    print(f"Point reduction:    {reduction:.2f}%")

    print("\nIMPORTANT:")
    print("This is still a monocular reconstruction.")
    print("Absolute translation scale remains unknown.")
    print("The next advanced step will improve global consistency.")


if __name__ == "__main__":
    main()