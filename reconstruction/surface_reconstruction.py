import os
import time
import numpy as np
import open3d as o3d


# ============================================================
# ARIA-S3D | PHASE 5.4
# SURFACE RECONSTRUCTION
# ============================================================

INPUT_PATH = "data/output/normal_estimated_point_cloud.ply"
OUTPUT_DIR = "data/output"

RAW_MESH_PATH = os.path.join(
    OUTPUT_DIR,
    "poisson_mesh_raw.ply"
)

CLEAN_MESH_PATH = os.path.join(
    OUTPUT_DIR,
    "reconstructed_mesh.ply"
)

CLEAN_MESH_OBJ_PATH = os.path.join(
    OUTPUT_DIR,
    "reconstructed_mesh.obj"
)

STATS_PATH = os.path.join(
    OUTPUT_DIR,
    "surface_reconstruction_stats.txt"
)


# ============================================================
# PARAMETERS
# ============================================================

POISSON_DEPTH = 8
POISSON_SCALE = 1.1
LINEAR_FIT = False

DENSITY_PERCENTILE = 2.0

MIN_TRIANGLE_AREA = 1e-10


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def print_header(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def safe_float(value):
    return float(value)


def validate_points(points):
    if len(points) == 0:
        return False

    return np.isfinite(points).all()


def validate_triangles(triangles):
    if len(triangles) == 0:
        return False

    return np.isfinite(triangles).all()


# ============================================================
# START
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

print_header(
    "ARIA-S3D | PHASE 5.4\n"
    "SURFACE RECONSTRUCTION"
)


# ============================================================
# 1. CHECK INPUT
# ============================================================

print()
print("[1] Checking normal-estimated point cloud")
print("-" * 70)

if not os.path.exists(INPUT_PATH):
    print("[ERROR] Input point cloud not found:")
    print(f"        {INPUT_PATH}")
    raise SystemExit(1)

print("[OK] Input point cloud found:")
print(f"     {INPUT_PATH}")


# ============================================================
# 2. LOAD POINT CLOUD
# ============================================================

print()
print("[2] Loading oriented point cloud")
print("-" * 70)

pcd = o3d.io.read_point_cloud(INPUT_PATH)

if pcd.is_empty():
    print("[ERROR] Point cloud is empty.")
    raise SystemExit(1)

points = np.asarray(pcd.points)

print(f"Points loaded : {len(points)}")

if pcd.has_normals():
    normals = np.asarray(pcd.normals)
    print(f"Normals loaded: {len(normals)}")
else:
    print("[ERROR] Input point cloud does not contain normals.")
    raise SystemExit(1)


# ============================================================
# 3. VALIDATE INPUT
# ============================================================

print()
print("[3] Validating point cloud")
print("-" * 70)

if not validate_points(points):
    print("[ERROR] Point cloud contains non-finite coordinates.")
    raise SystemExit(1)

print("[OK] All point coordinates are finite.")

if len(normals) != len(points):
    print("[ERROR] Point/normal count mismatch.")
    raise SystemExit(1)

if not np.isfinite(normals).all():
    print("[ERROR] Normals contain non-finite values.")
    raise SystemExit(1)

normal_lengths = np.linalg.norm(normals, axis=1)

valid_normals = (
    np.isfinite(normal_lengths) &
    (normal_lengths > 0.0)
)

if not np.all(valid_normals):
    print("[ERROR] Invalid normal vectors detected.")
    raise SystemExit(1)

print("[OK] Point cloud and normals are valid.")


# ============================================================
# 4. INPUT GEOMETRY
# ============================================================

print()
print("[4] Input geometry")
print("-" * 70)

mins = points.min(axis=0)
maxs = points.max(axis=0)

print(
    f"X range : {mins[0]:.6f} to {maxs[0]:.6f}"
)

print(
    f"Y range : {mins[1]:.6f} to {maxs[1]:.6f}"
)

print(
    f"Z range : {mins[2]:.6f} to {maxs[2]:.6f}"
)


# ============================================================
# 5. POISSON RECONSTRUCTION
# ============================================================

print()
print("[5] Running Poisson surface reconstruction")
print("-" * 70)

print(f"Poisson depth : {POISSON_DEPTH}")
print(f"Poisson scale : {POISSON_SCALE}")
print(f"Linear fit    : {LINEAR_FIT}")

print()
print(
    "This may take some time depending on CPU and available memory..."
)

start_time = time.perf_counter()

try:

    mesh, densities = (
        o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd,
            depth=POISSON_DEPTH,
            scale=POISSON_SCALE,
            linear_fit=LINEAR_FIT
        )
    )

except Exception as e:

    print()
    print("[ERROR] Poisson reconstruction failed.")
    print(f"        {e}")
    raise SystemExit(1)

processing_time = time.perf_counter() - start_time

print()
print("[OK] Poisson reconstruction completed.")
print(f"Processing time : {processing_time:.3f} seconds")


# ============================================================
# 6. VALIDATE RAW MESH
# ============================================================

print()
print("[6] Validating raw mesh")
print("-" * 70)

raw_vertices = np.asarray(mesh.vertices)
raw_triangles = np.asarray(mesh.triangles)
density_values = np.asarray(densities)

print(f"Raw mesh vertices  : {len(raw_vertices)}")
print(f"Raw mesh triangles : {len(raw_triangles)}")
print(f"Density values     : {len(density_values)}")

if len(raw_vertices) == 0:
    print("[ERROR] Poisson mesh contains no vertices.")
    raise SystemExit(1)

if len(raw_triangles) == 0:
    print("[ERROR] Poisson mesh contains no triangles.")
    raise SystemExit(1)

if not np.isfinite(raw_vertices).all():
    print("[ERROR] Raw mesh contains non-finite vertices.")
    raise SystemExit(1)

if not np.isfinite(raw_triangles).all():
    print("[ERROR] Raw mesh contains invalid triangle indices.")
    raise SystemExit(1)

if not np.isfinite(density_values).all():
    print("[ERROR] Density values contain non-finite values.")
    raise SystemExit(1)

print("[OK] Raw mesh geometry is finite.")


# ============================================================
# 7. COMPUTE RAW MESH NORMALS
# ============================================================

print()
print("[7] Computing raw mesh normals")
print("-" * 70)

mesh.compute_vertex_normals()

print("[OK] Raw mesh vertex normals computed.")


# ============================================================
# 8. SAVE RAW POISSON MESH
# ============================================================

print()
print("[8] Saving raw Poisson mesh")
print("-" * 70)

# IMPORTANT:
# Open3D 0.19.0 does NOT support:
# write_triangle_normals=True
#
# Only write_vertex_normals is supported.

raw_success = o3d.io.write_triangle_mesh(
    RAW_MESH_PATH,
    mesh,
    write_ascii=False,
    compressed=False,
    write_vertex_normals=True,
    write_vertex_colors=True,
    write_triangle_uvs=True,
    print_progress=False
)

if not raw_success:
    print("[ERROR] Failed to save raw Poisson mesh.")
    raise SystemExit(1)

print("[OK] Raw mesh saved:")
print(f"     {RAW_MESH_PATH}")


# ============================================================
# 9. DENSITY STATISTICS
# ============================================================

print()
print("[9] Computing vertex-density statistics")
print("-" * 70)

density_min = float(np.min(density_values))
density_max = float(np.max(density_values))
density_mean = float(np.mean(density_values))
density_median = float(np.median(density_values))

print(f"Minimum density : {density_min:.6f}")
print(f"Maximum density : {density_max:.6f}")
print(f"Mean density    : {density_mean:.6f}")
print(f"Median density  : {density_median:.6f}")


# ============================================================
# 10. DENSITY-BASED CLEANING
# ============================================================

print()
print("[10] Removing low-density mesh regions")
print("-" * 70)

density_threshold = float(
    np.percentile(
        density_values,
        DENSITY_PERCENTILE
    )
)

low_density_mask = (
    density_values < density_threshold
)

low_density_count = int(
    np.sum(low_density_mask)
)

print(
    f"Density percentile : "
    f"{DENSITY_PERCENTILE:.2f}%"
)

print(
    f"Density threshold  : "
    f"{density_threshold:.6f}"
)

print(
    f"Vertices considered low-density : "
    f"{low_density_count}"
)


# ============================================================
# 11. CREATE CLEAN MESH
# ============================================================

print()
print("[11] Cleaning reconstructed mesh")
print("-" * 70)

clean_mesh = o3d.geometry.TriangleMesh(mesh)

# Open3D removes triangles connected to low-density vertices.
clean_mesh.remove_vertices_by_mask(
    low_density_mask
)

# IMPORTANT:
# Open3D legacy geometry methods generally modify the object
# in-place and return None.
#
# Therefore DO NOT do:
#
# clean_mesh = clean_mesh.remove_unreferenced_vertices()
#
# Correct:
clean_mesh.remove_unreferenced_vertices()

clean_mesh.remove_degenerate_triangles()
clean_mesh.remove_duplicated_triangles()
clean_mesh.remove_duplicated_vertices()

print("[OK] Low-density regions removed.")
print("[OK] Unreferenced vertices removed.")
print("[OK] Degenerate triangles removed.")
print("[OK] Duplicate triangles removed.")
print("[OK] Duplicate vertices removed.")


# ============================================================
# 12. RECOMPUTE NORMALS
# ============================================================

print()
print("[12] Recomputing cleaned mesh normals")
print("-" * 70)

if len(clean_mesh.vertices) == 0:
    print("[ERROR] Cleaning removed all mesh vertices.")
    raise SystemExit(1)

if len(clean_mesh.triangles) == 0:
    print("[ERROR] Cleaning removed all mesh triangles.")
    raise SystemExit(1)

clean_mesh.compute_vertex_normals()

print("[OK] Cleaned mesh normals computed.")


# ============================================================
# 13. VALIDATE CLEAN MESH
# ============================================================

print()
print("[13] Validating cleaned mesh")
print("-" * 70)

clean_vertices = np.asarray(
    clean_mesh.vertices
)

clean_triangles = np.asarray(
    clean_mesh.triangles
)

print(
    f"Clean mesh vertices  : "
    f"{len(clean_vertices)}"
)

print(
    f"Clean mesh triangles : "
    f"{len(clean_triangles)}"
)

if not np.isfinite(clean_vertices).all():
    print("[ERROR] Clean mesh contains non-finite vertices.")
    raise SystemExit(1)

if not validate_triangles(clean_triangles):
    print("[ERROR] Clean mesh contains invalid triangles.")
    raise SystemExit(1)

if np.max(clean_triangles) >= len(clean_vertices):
    print("[ERROR] Triangle index exceeds vertex count.")
    raise SystemExit(1)

print("[OK] Clean mesh geometry is finite.")
print("[OK] Triangle indices are valid.")


# ============================================================
# 14. MESH GEOMETRY
# ============================================================

print()
print("[14] Clean mesh geometry")
print("-" * 70)

clean_mins = clean_vertices.min(axis=0)
clean_maxs = clean_vertices.max(axis=0)

print(
    f"X range : "
    f"{clean_mins[0]:.6f} to {clean_maxs[0]:.6f}"
)

print(
    f"Y range : "
    f"{clean_mins[1]:.6f} to {clean_maxs[1]:.6f}"
)

print(
    f"Z range : "
    f"{clean_mins[2]:.6f} to {clean_maxs[2]:.6f}"
)


# ============================================================
# 15. MESH STATISTICS
# ============================================================

print()
print("[15] Mesh statistics")
print("-" * 70)

raw_vertex_count = len(raw_vertices)
raw_triangle_count = len(raw_triangles)

clean_vertex_count = len(clean_vertices)
clean_triangle_count = len(clean_triangles)

vertices_removed = (
    raw_vertex_count -
    clean_vertex_count
)

triangles_removed = (
    raw_triangle_count -
    clean_triangle_count
)

vertex_retention = (
    clean_vertex_count /
    raw_vertex_count *
    100.0
)

triangle_retention = (
    clean_triangle_count /
    raw_triangle_count *
    100.0
)

print(
    f"Raw vertices        : "
    f"{raw_vertex_count}"
)

print(
    f"Clean vertices      : "
    f"{clean_vertex_count}"
)

print(
    f"Vertices removed    : "
    f"{vertices_removed}"
)

print(
    f"Vertex retention    : "
    f"{vertex_retention:.2f}%"
)

print()

print(
    f"Raw triangles       : "
    f"{raw_triangle_count}"
)

print(
    f"Clean triangles     : "
    f"{clean_triangle_count}"
)

print(
    f"Triangles removed   : "
    f"{triangles_removed}"
)

print(
    f"Triangle retention  : "
    f"{triangle_retention:.2f}%"
)


# ============================================================
# 16. SAVE CLEAN PLY
# ============================================================

print()
print("[16] Saving reconstructed mesh")
print("-" * 70)

mesh_success = o3d.io.write_triangle_mesh(
    CLEAN_MESH_PATH,
    clean_mesh,
    write_ascii=False,
    compressed=False,
    write_vertex_normals=True,
    write_vertex_colors=True,
    write_triangle_uvs=True,
    print_progress=False
)

if not mesh_success:
    print("[ERROR] Failed to save reconstructed mesh.")
    raise SystemExit(1)

print("[OK] Reconstructed mesh saved:")
print(f"     {CLEAN_MESH_PATH}")


# ============================================================
# 17. SAVE OBJ
# ============================================================

print()
print("[17] Saving OBJ mesh")
print("-" * 70)

obj_success = o3d.io.write_triangle_mesh(
    CLEAN_MESH_OBJ_PATH,
    clean_mesh,
    write_ascii=False,
    compressed=False,
    write_vertex_normals=True,
    write_vertex_colors=True,
    write_triangle_uvs=True,
    print_progress=False
)

if obj_success:
    print("[OK] OBJ mesh saved:")
    print(f"     {CLEAN_MESH_OBJ_PATH}")
else:
    print("[WARNING] OBJ export failed.")
    print("          PLY output remains valid.")


# ============================================================
# 18. SAVE STATISTICS
# ============================================================

print()
print("[18] Saving reconstruction statistics")
print("-" * 70)

with open(STATS_PATH, "w", encoding="utf-8") as f:

    f.write("ARIA-S3D | SURFACE RECONSTRUCTION\n")
    f.write("=" * 70 + "\n\n")

    f.write("INPUT\n")
    f.write("-" * 70 + "\n")
    f.write(f"Input point cloud: {INPUT_PATH}\n")
    f.write(f"Input points: {len(points)}\n\n")

    f.write("POISSON PARAMETERS\n")
    f.write("-" * 70 + "\n")
    f.write(f"Depth: {POISSON_DEPTH}\n")
    f.write(f"Scale: {POISSON_SCALE}\n")
    f.write(f"Linear fit: {LINEAR_FIT}\n")
    f.write(
        f"Processing time: "
        f"{processing_time:.6f} seconds\n\n"
    )

    f.write("DENSITY\n")
    f.write("-" * 70 + "\n")
    f.write(
        f"Minimum density: "
        f"{density_min:.6f}\n"
    )

    f.write(
        f"Maximum density: "
        f"{density_max:.6f}\n"
    )

    f.write(
        f"Mean density: "
        f"{density_mean:.6f}\n"
    )

    f.write(
        f"Median density: "
        f"{density_median:.6f}\n"
    )

    f.write(
        f"Density percentile: "
        f"{DENSITY_PERCENTILE:.2f}\n"
    )

    f.write(
        f"Density threshold: "
        f"{density_threshold:.6f}\n"
    )

    f.write(
        f"Low-density vertices: "
        f"{low_density_count}\n\n"
    )

    f.write("RAW MESH\n")
    f.write("-" * 70 + "\n")
    f.write(
        f"Vertices: "
        f"{raw_vertex_count}\n"
    )

    f.write(
        f"Triangles: "
        f"{raw_triangle_count}\n\n"
    )

    f.write("CLEAN MESH\n")
    f.write("-" * 70 + "\n")
    f.write(
        f"Vertices: "
        f"{clean_vertex_count}\n"
    )

    f.write(
        f"Triangles: "
        f"{clean_triangle_count}\n"
    )

    f.write(
        f"Vertices removed: "
        f"{vertices_removed}\n"
    )

    f.write(
        f"Triangles removed: "
        f"{triangles_removed}\n"
    )

    f.write(
        f"Vertex retention: "
        f"{vertex_retention:.2f}%\n"
    )

    f.write(
        f"Triangle retention: "
        f"{triangle_retention:.2f}%\n\n"
    )

    f.write("OUTPUTS\n")
    f.write("-" * 70 + "\n")
    f.write(
        f"Raw mesh: "
        f"{RAW_MESH_PATH}\n"
    )

    f.write(
        f"Clean mesh: "
        f"{CLEAN_MESH_PATH}\n"
    )

    f.write(
        f"OBJ mesh: "
        f"{CLEAN_MESH_OBJ_PATH}\n"
    )


print("[OK] Statistics saved:")
print(f"     {STATS_PATH}")


# ============================================================
# COMPLETE
# ============================================================

print_header(
    "ARIA-S3D | PHASE 5.4 COMPLETE"
)

print()
print(
    f"Input points          : {len(points)}"
)

print(
    f"Raw mesh vertices     : {raw_vertex_count}"
)

print(
    f"Raw mesh triangles    : {raw_triangle_count}"
)

print(
    f"Clean mesh vertices   : {clean_vertex_count}"
)

print(
    f"Clean mesh triangles  : {clean_triangle_count}"
)

print(
    f"Vertex retention      : "
    f"{vertex_retention:.2f}%"
)

print(
    f"Triangle retention    : "
    f"{triangle_retention:.2f}%"
)

print()
print("Generated outputs:")

print(
    f"  {RAW_MESH_PATH}"
)

print(
    f"  {CLEAN_MESH_PATH}"
)

print(
    f"  {CLEAN_MESH_OBJ_PATH}"
)

print(
    f"  {STATS_PATH}"
)

print()
print("NEXT STEP:")
print("Phase 5.5 - Mesh cleanup and validation")

print()
print("IMPORTANT:")
print("The reconstruction remains monocular.")
print("Absolute metric scale remains arbitrary.")
print("Poisson reconstruction converts the oriented point cloud")
print("into a continuous surface mesh.")
print()