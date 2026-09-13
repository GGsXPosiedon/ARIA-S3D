import os
import time
import numpy as np
import open3d as o3d


# ============================================================
# ARIA-S3D | PHASE 5.1
# POINT-CLOUD PREPROCESSING
# ============================================================

INPUT_PATH = "data/output/bundle_adjusted_point_cloud.xyz"
OUTPUT_DIR = "data/output"

CLEANED_POINT_CLOUD = os.path.join(
    OUTPUT_DIR,
    "preprocessed_point_cloud.ply"
)

CLEANED_XYZ = os.path.join(
    OUTPUT_DIR,
    "preprocessed_point_cloud.xyz"
)

STATS_PATH = os.path.join(
    OUTPUT_DIR,
    "point_cloud_preprocessing_stats.txt"
)


# ============================================================
# CONFIGURATION
# ============================================================

# Statistical outlier removal
NB_NEIGHBORS = 30
STD_RATIO = 2.0

# Remove exact/near duplicate points
DUPLICATE_PRECISION = 6


# ============================================================
# SETUP
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

start_time = time.perf_counter()


print("=" * 70)
print("ARIA-S3D | PHASE 5.1")
print("POINT-CLOUD PREPROCESSING")
print("=" * 70)


# ============================================================
# CHECK INPUT
# ============================================================

print()
print("[1] Checking input point cloud")
print("-" * 70)

if not os.path.exists(INPUT_PATH):
    print("[ERROR] Input point cloud not found:")
    print(f"        {INPUT_PATH}")
    raise SystemExit(1)

print("[OK] Input point cloud found:")
print(f"     {INPUT_PATH}")


# ============================================================
# LOAD RAW POINT CLOUD
# ============================================================

print()
print("[2] Loading optimized point cloud")
print("-" * 70)

try:
    raw_points = np.loadtxt(INPUT_PATH)
except Exception as exc:
    print("[ERROR] Failed to load point cloud.")
    print(f"        {exc}")
    raise SystemExit(1)


if raw_points.ndim == 1:
    raw_points = raw_points.reshape(1, -1)


if raw_points.shape[1] < 3:
    print("[ERROR] Point cloud must contain at least X Y Z.")
    raise SystemExit(1)


raw_points = raw_points[:, :3]

print(f"Raw points loaded : {len(raw_points)}")


# ============================================================
# FINITE POINT VALIDATION
# ============================================================

print()
print("[3] Removing invalid / non-finite points")
print("-" * 70)

finite_mask = np.isfinite(raw_points).all(axis=1)

finite_points = raw_points[finite_mask]

invalid_count = len(raw_points) - len(finite_points)

print(f"Invalid points removed : {invalid_count}")
print(f"Finite points remaining : {len(finite_points)}")


if len(finite_points) < NB_NEIGHBORS:
    print(
        "[ERROR] Not enough valid points for "
        "statistical outlier removal."
    )
    raise SystemExit(1)


# ============================================================
# DUPLICATE REMOVAL
# ============================================================

print()
print("[4] Removing duplicate points")
print("-" * 70)

rounded_points = np.round(
    finite_points,
    decimals=DUPLICATE_PRECISION
)

_, unique_indices = np.unique(
    rounded_points,
    axis=0,
    return_index=True
)

unique_indices = np.sort(unique_indices)

unique_points = finite_points[unique_indices]

duplicate_count = (
    len(finite_points) -
    len(unique_points)
)

print(f"Duplicate points removed : {duplicate_count}")
print(f"Unique points remaining  : {len(unique_points)}")


if len(unique_points) < NB_NEIGHBORS:
    print(
        "[ERROR] Not enough unique points for "
        "statistical filtering."
    )
    raise SystemExit(1)


# ============================================================
# OPEN3D POINT CLOUD
# ============================================================

print()
print("[5] Creating Open3D point cloud")
print("-" * 70)

pcd = o3d.geometry.PointCloud()

pcd.points = o3d.utility.Vector3dVector(
    unique_points
)

print("[OK] Open3D point cloud created.")
print(f"     Points: {len(pcd.points)}")


# ============================================================
# STATISTICAL OUTLIER REMOVAL
# ============================================================

print()
print("[6] Statistical outlier removal")
print("-" * 70)

print(f"Neighbors : {NB_NEIGHBORS}")
print(f"Std ratio : {STD_RATIO}")

filtered_pcd, inlier_indices = (
    pcd.remove_statistical_outlier(
        nb_neighbors=NB_NEIGHBORS,
        std_ratio=STD_RATIO
    )
)

filtered_points = np.asarray(
    filtered_pcd.points
)

outlier_count = (
    len(unique_points) -
    len(filtered_points)
)

print()
print(f"Outliers removed : {outlier_count}")
print(f"Clean points     : {len(filtered_points)}")


if len(filtered_points) == 0:
    print("[ERROR] Filtering removed all points.")
    raise SystemExit(1)


# ============================================================
# COMPUTE BOUNDING BOX
# ============================================================

print()
print("[7] Computing cleaned cloud geometry")
print("-" * 70)

x_min = filtered_points[:, 0].min()
x_max = filtered_points[:, 0].max()

y_min = filtered_points[:, 1].min()
y_max = filtered_points[:, 1].max()

z_min = filtered_points[:, 2].min()
z_max = filtered_points[:, 2].max()

print(f"X range : {x_min:.6f} to {x_max:.6f}")
print(f"Y range : {y_min:.6f} to {y_max:.6f}")
print(f"Z range : {z_min:.6f} to {z_max:.6f}")


# ============================================================
# SAVE PLY
# ============================================================

print()
print("[8] Saving preprocessed point cloud")
print("-" * 70)

if not o3d.io.write_point_cloud(
    CLEANED_POINT_CLOUD,
    filtered_pcd
):
    print("[ERROR] Failed to save PLY point cloud.")
    raise SystemExit(1)

print("[OK] PLY point cloud saved:")
print(f"     {CLEANED_POINT_CLOUD}")


# ============================================================
# SAVE XYZ
# ============================================================

np.savetxt(
    CLEANED_XYZ,
    filtered_points,
    fmt="%.6f"
)

print("[OK] XYZ point cloud saved:")
print(f"     {CLEANED_XYZ}")


# ============================================================
# STATISTICS
# ============================================================

processing_time = (
    time.perf_counter() -
    start_time
)

removal_total = (
    len(raw_points) -
    len(filtered_points)
)

retention_percentage = (
    len(filtered_points) /
    len(raw_points) *
    100.0
)


# ============================================================
# SAVE STATISTICS
# ============================================================

with open(STATS_PATH, "w") as f:

    f.write("ARIA-S3D | PHASE 5.1\n")
    f.write("POINT-CLOUD PREPROCESSING\n")
    f.write("=" * 70 + "\n\n")

    f.write(
        f"Input point cloud : {INPUT_PATH}\n"
    )

    f.write(
        f"Raw points        : {len(raw_points)}\n"
    )

    f.write(
        f"Invalid removed   : {invalid_count}\n"
    )

    f.write(
        f"Duplicates removed: {duplicate_count}\n"
    )

    f.write(
        f"Outliers removed  : {outlier_count}\n"
    )

    f.write(
        f"Final points      : {len(filtered_points)}\n"
    )

    f.write(
        f"Total removed     : {removal_total}\n"
    )

    f.write(
        f"Retention         : {retention_percentage:.4f}%\n"
    )

    f.write("\n")

    f.write(
        f"X range : {x_min:.6f} to {x_max:.6f}\n"
    )

    f.write(
        f"Y range : {y_min:.6f} to {y_max:.6f}\n"
    )

    f.write(
        f"Z range : {z_min:.6f} to {z_max:.6f}\n"
    )

    f.write("\n")

    f.write(
        f"Statistical neighbors : {NB_NEIGHBORS}\n"
    )

    f.write(
        f"Statistical std ratio : {STD_RATIO}\n"
    )

    f.write(
        f"Processing time       : {processing_time:.4f} seconds\n"
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("ARIA-S3D | PHASE 5.1 COMPLETE")
print("=" * 70)

print()
print(f"Raw points            : {len(raw_points)}")
print(f"Invalid points removed: {invalid_count}")
print(f"Duplicates removed    : {duplicate_count}")
print(f"Statistical outliers  : {outlier_count}")
print(f"Final clean points    : {len(filtered_points)}")
print(f"Retention              : {retention_percentage:.2f}%")

print()
print("Generated outputs:")

print(f"  {CLEANED_POINT_CLOUD}")
print(f"  {CLEANED_XYZ}")
print(f"  {STATS_PATH}")

print()
print("NEXT STEP:")
print("Phase 5.2 - Point-cloud downsampling")

print()
print("IMPORTANT:")
print("The reconstruction remains monocular.")
print("Absolute metric scale remains arbitrary.")
print("=" * 70)