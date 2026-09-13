import os
import time
import numpy as np
import open3d as o3d


# ============================================================
# ARIA-S3D | PHASE 5.2
# POINT-CLOUD DOWNSAMPLING
# ============================================================

INPUT_PATH = "data/output/preprocessed_point_cloud.ply"
OUTPUT_DIR = "data/output"

OUTPUT_PLY = os.path.join(
    OUTPUT_DIR,
    "downsampled_point_cloud.ply"
)

OUTPUT_XYZ = os.path.join(
    OUTPUT_DIR,
    "downsampled_point_cloud.xyz"
)

STATS_PATH = os.path.join(
    OUTPUT_DIR,
    "point_cloud_downsampling_stats.txt"
)


# ============================================================
# CONFIGURATION
# ============================================================

# Monocular reconstruction has arbitrary scale.
# Therefore this is a tunable geometric voxel size.
VOXEL_SIZE = 0.10


# ============================================================
# SETUP
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

start_time = time.perf_counter()


print("=" * 70)
print("ARIA-S3D | PHASE 5.2")
print("POINT-CLOUD DOWNSAMPLING")
print("=" * 70)


# ============================================================
# CHECK INPUT
# ============================================================

print()
print("[1] Checking preprocessed point cloud")
print("-" * 70)

if not os.path.exists(INPUT_PATH):
    print("[ERROR] Preprocessed point cloud not found:")
    print(f"        {INPUT_PATH}")
    raise SystemExit(1)

print("[OK] Input point cloud found:")
print(f"     {INPUT_PATH}")


# ============================================================
# LOAD POINT CLOUD
# ============================================================

print()
print("[2] Loading preprocessed point cloud")
print("-" * 70)

pcd = o3d.io.read_point_cloud(INPUT_PATH)

if pcd.is_empty():
    print("[ERROR] Loaded point cloud is empty.")
    raise SystemExit(1)

points_before = np.asarray(pcd.points)

print(f"Input points : {len(points_before)}")


# ============================================================
# VALIDATE POINTS
# ============================================================

print()
print("[3] Validating point cloud")
print("-" * 70)

if not np.isfinite(points_before).all():
    print("[ERROR] Input contains non-finite points.")
    raise SystemExit(1)

print("[OK] All input points are finite.")


# ============================================================
# COMPUTE INPUT GEOMETRY
# ============================================================

input_min = points_before.min(axis=0)
input_max = points_before.max(axis=0)

print()
print("Input geometry:")
print(
    f"X range : {input_min[0]:.6f} "
    f"to {input_max[0]:.6f}"
)

print(
    f"Y range : {input_min[1]:.6f} "
    f"to {input_max[1]:.6f}"
)

print(
    f"Z range : {input_min[2]:.6f} "
    f"to {input_max[2]:.6f}"
)


# ============================================================
# VOXEL DOWNSAMPLING
# ============================================================

print()
print("[4] Voxel downsampling")
print("-" * 70)

print(f"Voxel size : {VOXEL_SIZE}")

downsampled_pcd = pcd.voxel_down_sample(
    voxel_size=VOXEL_SIZE
)

points_after = np.asarray(
    downsampled_pcd.points
)

print()
print(f"Points before : {len(points_before)}")
print(f"Points after  : {len(points_after)}")


# ============================================================
# VALIDATE OUTPUT
# ============================================================

if len(points_after) == 0:
    print("[ERROR] Downsampling produced an empty point cloud.")
    raise SystemExit(1)

if not np.isfinite(points_after).all():
    print("[ERROR] Downsampled cloud contains non-finite points.")
    raise SystemExit(1)

print("[OK] Downsampled point cloud is valid.")


# ============================================================
# COMPUTE OUTPUT GEOMETRY
# ============================================================

output_min = points_after.min(axis=0)
output_max = points_after.max(axis=0)

print()
print("[5] Computing downsampled geometry")
print("-" * 70)

print(
    f"X range : {output_min[0]:.6f} "
    f"to {output_max[0]:.6f}"
)

print(
    f"Y range : {output_min[1]:.6f} "
    f"to {output_max[1]:.6f}"
)

print(
    f"Z range : {output_min[2]:.6f} "
    f"to {output_max[2]:.6f}"
)


# ============================================================
# DOWNSAMPLING STATISTICS
# ============================================================

removed_points = (
    len(points_before) -
    len(points_after)
)

retention_percentage = (
    len(points_after) /
    len(points_before)
) * 100.0

reduction_percentage = (
    removed_points /
    len(points_before)
) * 100.0


print()
print("[6] Downsampling statistics")
print("-" * 70)

print(f"Points removed      : {removed_points}")
print(f"Retention           : {retention_percentage:.2f}%")
print(f"Reduction           : {reduction_percentage:.2f}%")


# ============================================================
# SAVE PLY
# ============================================================

print()
print("[7] Saving downsampled point cloud")
print("-" * 70)

success = o3d.io.write_point_cloud(
    OUTPUT_PLY,
    downsampled_pcd
)

if not success:
    print("[ERROR] Failed to save PLY.")
    raise SystemExit(1)

print("[OK] PLY saved:")
print(f"     {OUTPUT_PLY}")


# ============================================================
# SAVE XYZ
# ============================================================

np.savetxt(
    OUTPUT_XYZ,
    points_after,
    fmt="%.6f"
)

print("[OK] XYZ saved:")
print(f"     {OUTPUT_XYZ}")


# ============================================================
# PROCESSING TIME
# ============================================================

processing_time = (
    time.perf_counter() -
    start_time
)


# ============================================================
# SAVE STATISTICS
# ============================================================

with open(STATS_PATH, "w") as f:

    f.write(
        "ARIA-S3D | PHASE 5.2\n"
    )

    f.write(
        "POINT-CLOUD DOWNSAMPLING\n"
    )

    f.write(
        "=" * 70 + "\n\n"
    )

    f.write(
        f"Input point cloud : {INPUT_PATH}\n"
    )

    f.write(
        f"Output point cloud: {OUTPUT_PLY}\n\n"
    )

    f.write(
        f"Input points      : {len(points_before)}\n"
    )

    f.write(
        f"Output points     : {len(points_after)}\n"
    )

    f.write(
        f"Points removed    : {removed_points}\n"
    )

    f.write(
        f"Retention         : "
        f"{retention_percentage:.4f}%\n"
    )

    f.write(
        f"Reduction         : "
        f"{reduction_percentage:.4f}%\n"
    )

    f.write(
        f"Voxel size        : {VOXEL_SIZE:.6f}\n\n"
    )

    f.write(
        f"Input X range : "
        f"{input_min[0]:.6f} "
        f"to {input_max[0]:.6f}\n"
    )

    f.write(
        f"Input Y range : "
        f"{input_min[1]:.6f} "
        f"to {input_max[1]:.6f}\n"
    )

    f.write(
        f"Input Z range : "
        f"{input_min[2]:.6f} "
        f"to {input_max[2]:.6f}\n\n"
    )

    f.write(
        f"Output X range : "
        f"{output_min[0]:.6f} "
        f"to {output_max[0]:.6f}\n"
    )

    f.write(
        f"Output Y range : "
        f"{output_min[1]:.6f} "
        f"to {output_max[1]:.6f}\n"
    )

    f.write(
        f"Output Z range : "
        f"{output_min[2]:.6f} "
        f"to {output_max[2]:.6f}\n\n"
    )

    f.write(
        f"Processing time : "
        f"{processing_time:.6f} seconds\n"
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("ARIA-S3D | PHASE 5.2 COMPLETE")
print("=" * 70)

print()
print(f"Input points       : {len(points_before)}")
print(f"Output points      : {len(points_after)}")
print(f"Points removed     : {removed_points}")
print(f"Retention          : {retention_percentage:.2f}%")
print(f"Reduction          : {reduction_percentage:.2f}%")
print(f"Voxel size         : {VOXEL_SIZE}")

print()
print("Generated outputs:")

print(f"  {OUTPUT_PLY}")
print(f"  {OUTPUT_XYZ}")
print(f"  {STATS_PATH}")

print()
print("NEXT STEP:")
print("Phase 5.3 - Surface normal estimation")

print()
print("IMPORTANT:")
print("Voxel size is expressed in the reconstruction's arbitrary")
print("monocular coordinate scale.")
print("Absolute metric scale remains arbitrary.")

print("=" * 70)