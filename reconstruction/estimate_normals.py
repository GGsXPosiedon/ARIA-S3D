import os
import time
import numpy as np
import open3d as o3d


# ============================================================
# ARIA-S3D | PHASE 5.3
# SURFACE NORMAL ESTIMATION
# ============================================================

INPUT_PATH = "data/output/downsampled_point_cloud.ply"
OUTPUT_DIR = "data/output"

OUTPUT_PLY = os.path.join(
    OUTPUT_DIR,
    "normal_estimated_point_cloud.ply"
)

OUTPUT_XYZ = os.path.join(
    OUTPUT_DIR,
    "normal_estimated_point_cloud.xyz"
)

STATS_PATH = os.path.join(
    OUTPUT_DIR,
    "surface_normal_estimation_stats.txt"
)


# ============================================================
# CONFIGURATION
# ============================================================

# Number of neighboring points used to estimate normals.
NORMAL_RADIUS = 0.5
NORMAL_MAX_NN = 50

# Orientation parameters.
ORIENTATION_K = 30


# ============================================================
# SETUP
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

start_time = time.perf_counter()


print("=" * 70)
print("ARIA-S3D | PHASE 5.3")
print("SURFACE NORMAL ESTIMATION")
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
# LOAD POINT CLOUD
# ============================================================

print()
print("[2] Loading downsampled point cloud")
print("-" * 70)

pcd = o3d.io.read_point_cloud(INPUT_PATH)

if pcd.is_empty():
    print("[ERROR] Point cloud is empty.")
    raise SystemExit(1)

points = np.asarray(pcd.points)

print(f"Points loaded : {len(points)}")


# ============================================================
# VALIDATE INPUT
# ============================================================

print()
print("[3] Validating input geometry")
print("-" * 70)

if not np.isfinite(points).all():
    print("[ERROR] Input contains non-finite coordinates.")
    raise SystemExit(1)

print("[OK] All point coordinates are finite.")


if len(points) < ORIENTATION_K:
    print(
        "[ERROR] Not enough points for normal orientation."
    )
    raise SystemExit(1)


# ============================================================
# ESTIMATE NORMALS
# ============================================================

print()
print("[4] Estimating surface normals")
print("-" * 70)

print(f"Search radius : {NORMAL_RADIUS}")
print(f"Maximum NN    : {NORMAL_MAX_NN}")

search_param = o3d.geometry.KDTreeSearchParamHybrid(
    radius=NORMAL_RADIUS,
    max_nn=NORMAL_MAX_NN
)

pcd.estimate_normals(
    search_param=search_param
)

normals = np.asarray(pcd.normals)

if len(normals) != len(points):
    print("[ERROR] Number of normals does not match points.")
    raise SystemExit(1)

print(f"[OK] Normals estimated : {len(normals)}")


# ============================================================
# VALIDATE ESTIMATED NORMALS
# ============================================================

print()
print("[5] Validating estimated normals")
print("-" * 70)

finite_normals = np.isfinite(normals).all(axis=1)

invalid_normals = np.count_nonzero(
    ~finite_normals
)

print(f"Non-finite normals : {invalid_normals}")

if invalid_normals > 0:
    print(
        "[ERROR] Invalid normals detected."
    )
    raise SystemExit(1)

print("[OK] All normals are finite.")


# ============================================================
# NORMAL MAGNITUDE CHECK
# ============================================================

normal_lengths = np.linalg.norm(
    normals,
    axis=1
)

invalid_lengths = (
    ~np.isfinite(normal_lengths)
    |
    (normal_lengths < 1e-8)
)

invalid_length_count = np.count_nonzero(
    invalid_lengths
)

print()
print("[6] Checking normal magnitudes")
print("-" * 70)

print(
    f"Minimum normal length : "
    f"{normal_lengths.min():.6f}"
)

print(
    f"Maximum normal length : "
    f"{normal_lengths.max():.6f}"
)

print(
    f"Invalid normal lengths: "
    f"{invalid_length_count}"
)

if invalid_length_count > 0:
    print(
        "[ERROR] Zero-length or invalid normals detected."
    )
    raise SystemExit(1)

print("[OK] Normal magnitudes are valid.")


# ============================================================
# ORIENT NORMALS
# ============================================================

print()
print("[7] Orienting normals consistently")
print("-" * 70)

print(
    f"Orientation neighborhood : "
    f"{ORIENTATION_K}"
)

try:

    pcd.orient_normals_consistent_tangent_plane(
        ORIENTATION_K
    )

except Exception as exc:

    print("[WARNING] Consistent orientation failed:")
    print(f"          {exc}")
    print()
    print(
        "[WARNING] Keeping estimated normals."
    )


# ============================================================
# RECHECK NORMALS
# ============================================================

normals = np.asarray(
    pcd.normals
)

finite_after_orientation = (
    np.isfinite(normals).all(axis=1)
)

invalid_after_orientation = np.count_nonzero(
    ~finite_after_orientation
)

normal_lengths_after = np.linalg.norm(
    normals,
    axis=1
)

invalid_length_after = np.count_nonzero(
    ~np.isfinite(normal_lengths_after)
    |
    (normal_lengths_after < 1e-8)
)


if invalid_after_orientation > 0:
    print(
        "[ERROR] Invalid normals after orientation."
    )
    raise SystemExit(1)

if invalid_length_after > 0:
    print(
        "[ERROR] Invalid normal magnitudes after orientation."
    )
    raise SystemExit(1)

print("[OK] Normal orientation validated.")


# ============================================================
# NORMAL STATISTICS
# ============================================================

normal_mean = normals.mean(axis=0)
normal_min = normals.min(axis=0)
normal_max = normals.max(axis=0)

print()
print("[8] Normal statistics")
print("-" * 70)

print(
    "Mean normal : "
    f"[{normal_mean[0]:.6f}, "
    f"{normal_mean[1]:.6f}, "
    f"{normal_mean[2]:.6f}]"
)

print(
    "Normal X range : "
    f"{normal_min[0]:.6f} "
    f"to {normal_max[0]:.6f}"
)

print(
    "Normal Y range : "
    f"{normal_min[1]:.6f} "
    f"to {normal_max[1]:.6f}"
)

print(
    "Normal Z range : "
    f"{normal_min[2]:.6f} "
    f"to {normal_max[2]:.6f}"
)


# ============================================================
# SAVE PLY
# ============================================================

print()
print("[9] Saving normal-enhanced point cloud")
print("-" * 70)

success = o3d.io.write_point_cloud(
    OUTPUT_PLY,
    pcd
)

if not success:
    print("[ERROR] Failed to save PLY.")
    raise SystemExit(1)

print("[OK] PLY saved:")
print(f"     {OUTPUT_PLY}")


# ============================================================
# SAVE XYZ + NORMALS
# ============================================================

print()
print("[10] Saving XYZ + normal data")
print("-" * 70)

output_data = np.hstack(
    (
        points,
        normals
    )
)

np.savetxt(
    OUTPUT_XYZ,
    output_data,
    fmt="%.6f",
    header="X Y Z NX NY NZ",
    comments=""
)

print("[OK] XYZ + normal data saved:")
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
        "ARIA-S3D | PHASE 5.3\n"
    )

    f.write(
        "SURFACE NORMAL ESTIMATION\n"
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
        f"Point count : {len(points)}\n"
    )

    f.write(
        f"Invalid normals : {invalid_normals}\n"
    )

    f.write(
        f"Invalid normal lengths : "
        f"{invalid_length_count}\n"
    )

    f.write(
        f"Normal radius : "
        f"{NORMAL_RADIUS:.6f}\n"
    )

    f.write(
        f"Maximum neighbors : "
        f"{NORMAL_MAX_NN}\n"
    )

    f.write(
        f"Orientation K : "
        f"{ORIENTATION_K}\n\n"
    )

    f.write(
        f"Mean normal X : "
        f"{normal_mean[0]:.6f}\n"
    )

    f.write(
        f"Mean normal Y : "
        f"{normal_mean[1]:.6f}\n"
    )

    f.write(
        f"Mean normal Z : "
        f"{normal_mean[2]:.6f}\n\n"
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
print("ARIA-S3D | PHASE 5.3 COMPLETE")
print("=" * 70)

print()
print(f"Points processed     : {len(points)}")
print(f"Normals generated    : {len(normals)}")
print(f"Invalid normals      : {invalid_normals}")
print(f"Normal radius        : {NORMAL_RADIUS}")
print(f"Maximum neighbors    : {NORMAL_MAX_NN}")
print(f"Orientation K        : {ORIENTATION_K}")

print()
print("Generated outputs:")

print(f"  {OUTPUT_PLY}")
print(f"  {OUTPUT_XYZ}")
print(f"  {STATS_PATH}")

print()
print("NEXT STEP:")
print("Phase 5.4 - Surface reconstruction")

print()
print("IMPORTANT:")
print("The reconstruction remains monocular.")
print("Absolute metric scale remains arbitrary.")
print("Surface reconstruction will use the estimated normals.")

print("=" * 70)