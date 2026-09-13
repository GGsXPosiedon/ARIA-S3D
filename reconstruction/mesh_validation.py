import os
import sys
import time
import numpy as np
import open3d as o3d


# ============================================================
# ARIA-S3D | PHASE 5.6
# ROBUST MESH VALIDATION v2
#
# Purpose:
#   Validate reconstructed meshes without performing an
#   unbounded full-mesh self-intersection test.
#
# Designed for:
#   - Drone video reconstruction
#   - Large Poisson meshes
#   - Open / non-watertight scene geometry
#
# Open3D 0.19 compatible
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_MESH = os.path.join(
    "data",
    "output",
    "cleaned_mesh.ply"
)

OUTPUT_DIR = os.path.join(
    "data",
    "output"
)

OUTPUT_MESH = os.path.join(
    OUTPUT_DIR,
    "validated_mesh.ply"
)

STATS_FILE = os.path.join(
    OUTPUT_DIR,
    "mesh_validation_stats.txt"
)

# Near-zero triangle area threshold.
# This is deliberately relative to the mesh scale.
NEAR_ZERO_AREA_RATIO = 1e-10

# Maximum number of triangles used for optional sampled
# self-intersection screening.
MAX_INTERSECTION_SAMPLE = 5000

# Enable sampled self-intersection screening.
ENABLE_SELF_INTERSECTION_SCREENING = True


# ============================================================
# HELPERS
# ============================================================

def separator():
    print("-" * 70)


def fail(message):
    print()
    print("[FAIL]", message)
    print()
    sys.exit(1)


def safe_float(value):
    if np.isfinite(value):
        return float(value)
    return 0.0


def triangle_area_statistics(vertices, triangles):
    """
    Calculate triangle areas using vector cross products.
    """

    if len(triangles) == 0:
        return (
            np.array([], dtype=np.float64),
            0.0,
            0.0,
            0.0
        )

    tri_vertices = vertices[triangles]

    v1 = tri_vertices[:, 1] - tri_vertices[:, 0]
    v2 = tri_vertices[:, 2] - tri_vertices[:, 0]

    cross = np.cross(v1, v2)

    areas = 0.5 * np.linalg.norm(
        cross,
        axis=1
    )

    return (
        areas,
        float(np.min(areas)),
        float(np.max(areas)),
        float(np.mean(areas))
    )


def count_duplicate_triangles(triangles):
    """
    Counts duplicate triangles independent of vertex ordering.

    Example:
        [1, 2, 3]
        [3, 1, 2]

    are considered the same triangle.
    """

    if len(triangles) == 0:
        return 0

    sorted_triangles = np.sort(
        triangles,
        axis=1
    )

    unique_triangles = np.unique(
        sorted_triangles,
        axis=0
    )

    return (
        len(sorted_triangles)
        - len(unique_triangles)
    )


def get_boundary_edge_count(mesh):
    """
    Count edges that belong to exactly one triangle.
    """

    triangles = np.asarray(
        mesh.triangles,
        dtype=np.int64
    )

    if len(triangles) == 0:
        return 0

    edges = np.vstack([
        triangles[:, [0, 1]],
        triangles[:, [1, 2]],
        triangles[:, [2, 0]]
    ])

    edges = np.sort(
        edges,
        axis=1
    )

    unique_edges, counts = np.unique(
        edges,
        axis=0,
        return_counts=True
    )

    return int(
        np.sum(counts == 1)
    )


def get_nonmanifold_edge_count(mesh):
    """
    Count edges shared by more than two triangles.
    """

    triangles = np.asarray(
        mesh.triangles,
        dtype=np.int64
    )

    if len(triangles) == 0:
        return 0

    edges = np.vstack([
        triangles[:, [0, 1]],
        triangles[:, [1, 2]],
        triangles[:, [2, 0]]
    ])

    edges = np.sort(
        edges,
        axis=1
    )

    _, counts = np.unique(
        edges,
        axis=0,
        return_counts=True
    )

    return int(
        np.sum(counts > 2)
    )


def sample_self_intersection_check(mesh):
    """
    Safe lightweight screening.

    IMPORTANT:
    This intentionally does NOT call:

        mesh.is_self_intersecting()

    on the complete mesh.

    Large reconstructed meshes can make that operation
    computationally expensive.

    Instead, a representative triangle subset is placed
    into a smaller mesh and Open3D's self-intersection
    test is performed on that subset.
    """

    triangles = np.asarray(
        mesh.triangles,
        dtype=np.int64
    )

    if len(triangles) == 0:
        return False, 0

    sample_count = min(
        len(triangles),
        MAX_INTERSECTION_SAMPLE
    )

    if len(triangles) <= sample_count:

        sampled_indices = np.arange(
            len(triangles)
        )

    else:

        rng = np.random.default_rng(
            seed=42
        )

        sampled_indices = rng.choice(
            len(triangles),
            size=sample_count,
            replace=False
        )

    sampled_triangles = triangles[
        sampled_indices
    ]

    # Find vertices used by sampled triangles.
    used_vertices = np.unique(
        sampled_triangles
    )

    vertex_map = -np.ones(
        len(mesh.vertices),
        dtype=np.int64
    )

    vertex_map[
        used_vertices
    ] = np.arange(
        len(used_vertices)
    )

    remapped_triangles = vertex_map[
        sampled_triangles
    ]

    sampled_mesh = o3d.geometry.TriangleMesh()

    sampled_mesh.vertices = (
        o3d.utility.Vector3dVector(
            np.asarray(
                mesh.vertices
            )[used_vertices]
        )
    )

    sampled_mesh.triangles = (
        o3d.utility.Vector3iVector(
            remapped_triangles
        )
    )

    # Remove simple invalid topology before screening.
    sampled_mesh.remove_duplicated_triangles()
    sampled_mesh.remove_degenerate_triangles()
    sampled_mesh.remove_unreferenced_vertices()

    if sampled_mesh.is_empty():
        return False, sample_count

    try:
        result = sampled_mesh.is_self_intersecting()
    except Exception as exc:

        print(
            "[WARNING] Sampled self-intersection test "
            f"could not complete: {exc}"
        )

        return False, sample_count

    return bool(result), sample_count


# ============================================================
# START
# ============================================================

start_time = time.time()

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

print("=" * 70)
print("ARIA-S3D | PHASE 5.6")
print("ROBUST MESH VALIDATION v2")
print("=" * 70)


# ============================================================
# [1] CHECK INPUT
# ============================================================

print()
print("[1] Checking cleaned mesh")
separator()

if not os.path.isfile(INPUT_MESH):
    fail(
        "Input mesh not found:\n"
        f"    {INPUT_MESH}"
    )

print(
    "[OK] Cleaned mesh found:\n"
    f"     {INPUT_MESH}"
)


# ============================================================
# [2] LOAD MESH
# ============================================================

print()
print("[2] Loading cleaned mesh")
separator()

mesh = o3d.io.read_triangle_mesh(
    INPUT_MESH
)

if mesh.is_empty():
    fail("Loaded mesh is empty.")

vertices = np.asarray(
    mesh.vertices
)

triangles = np.asarray(
    mesh.triangles,
    dtype=np.int64
)

print(
    f"Vertices  : {len(vertices)}"
)

print(
    f"Triangles : {len(triangles)}"
)


if len(vertices) == 0:
    fail("Mesh contains no vertices.")

if len(triangles) == 0:
    fail("Mesh contains no triangles.")


# ============================================================
# [3] GEOMETRY VALIDATION
# ============================================================

print()
print("[3] Validating mesh geometry")
separator()

finite_vertices = np.isfinite(
    vertices
).all(axis=1)

invalid_vertices = int(
    np.sum(~finite_vertices)
)

if invalid_vertices == 0:

    print(
        "[OK] All vertex coordinates are finite."
    )

else:

    print(
        f"[WARNING] Invalid vertices detected: "
        f"{invalid_vertices}"
    )


# Triangle index validation

triangle_indices_valid = (
    triangles.min() >= 0
    and
    triangles.max() < len(vertices)
)

if triangle_indices_valid:

    print(
        "[OK] All triangle indices are valid."
    )

else:

    fail(
        "Invalid triangle indices detected."
    )


# ============================================================
# [4] TRIANGLE INTEGRITY
# ============================================================

print()
print("[4] Checking triangle integrity")
separator()

repeated_vertex_mask = (
    (triangles[:, 0] == triangles[:, 1])
    |
    (triangles[:, 1] == triangles[:, 2])
    |
    (triangles[:, 0] == triangles[:, 2])
)

repeated_vertex_triangles = int(
    np.sum(repeated_vertex_mask)
)

print(
    "Repeated-vertex triangles : "
    f"{repeated_vertex_triangles}"
)

if repeated_vertex_triangles == 0:

    print(
        "[OK] No repeated-vertex triangles."
    )

else:

    print(
        "[WARNING] Repeated-vertex triangles detected."
    )


# ============================================================
# [5] TRIANGLE AREA ANALYSIS
# ============================================================

print()
print("[5] Checking triangle areas")
separator()

areas, min_area, max_area, mean_area = (
    triangle_area_statistics(
        vertices,
        triangles
    )
)

mesh_scale = (
    np.linalg.norm(
        vertices.max(axis=0)
        -
        vertices.min(axis=0)
    )
)

if mesh_scale <= 0:
    mesh_scale = 1.0

near_zero_threshold = (
    mesh_scale ** 2
    *
    NEAR_ZERO_AREA_RATIO
)

zero_area_triangles = int(
    np.sum(areas <= 0.0)
)

near_zero_triangles = int(
    np.sum(
        areas <= near_zero_threshold
    )
)

print(
    f"Mesh scale             : {mesh_scale:.6f}"
)

print(
    f"Near-zero threshold    : "
    f"{near_zero_threshold:.12e}"
)

print(
    f"Minimum triangle area  : "
    f"{min_area:.12f}"
)

print(
    f"Maximum triangle area  : "
    f"{max_area:.12f}"
)

print(
    f"Mean triangle area     : "
    f"{mean_area:.12f}"
)

print(
    f"Zero-area triangles    : "
    f"{zero_area_triangles}"
)

print(
    f"Near-zero triangles    : "
    f"{near_zero_triangles}"
)

if zero_area_triangles == 0:

    print(
        "[OK] No zero-area triangles."
    )

else:

    print(
        "[WARNING] Zero-area triangles detected."
    )


# ============================================================
# [6] DUPLICATE TRIANGLE ANALYSIS
# ============================================================

print()
print("[6] Checking duplicate triangles")
separator()

duplicate_triangles = count_duplicate_triangles(
    triangles
)

print(
    f"Duplicate triangles : "
    f"{duplicate_triangles}"
)

if duplicate_triangles == 0:

    print(
        "[OK] No duplicate triangles."
    )

else:

    print(
        "[WARNING] Duplicate triangles detected."
    )


# ============================================================
# [7] TOPOLOGY
# ============================================================

print()
print("[7] Checking mesh topology")
separator()

try:

    edge_manifold = mesh.is_edge_manifold(
        allow_boundary_edges=True
    )

    closed_edge_manifold = mesh.is_edge_manifold(
        allow_boundary_edges=False
    )

    vertex_manifold = mesh.is_vertex_manifold()

    orientable = mesh.is_orientable()

except Exception as exc:

    print(
        "[WARNING] Open3D topology check failed:"
    )

    print(
        f"          {exc}"
    )

    edge_manifold = False
    closed_edge_manifold = False
    vertex_manifold = False
    orientable = False


non_manifold_edges = (
    get_nonmanifold_edge_count(mesh)
)

non_manifold_vertices = len(
    mesh.get_non_manifold_vertices()
)

boundary_edges = (
    get_boundary_edge_count(mesh)
)

print(
    f"Edge manifold          : "
    f"{edge_manifold}"
)

print(
    f"Closed edge manifold   : "
    f"{closed_edge_manifold}"
)

print(
    f"Vertex manifold        : "
    f"{vertex_manifold}"
)

print(
    f"Orientable             : "
    f"{orientable}"
)

print(
    f"Non-manifold edges     : "
    f"{non_manifold_edges}"
)

print(
    f"Non-manifold vertices  : "
    f"{non_manifold_vertices}"
)

print(
    f"Boundary edges         : "
    f"{boundary_edges}"
)


# ============================================================
# [8] BOUNDARY ANALYSIS
# ============================================================

print()
print("[8] Checking boundary topology")
separator()

if boundary_edges == 0:

    print(
        "[OK] Mesh has no boundary edges."
    )

else:

    print(
        "[INFO] Mesh contains open boundary edges."
    )

    print(
        "       This is acceptable for many drone-scene "
        "reconstructions."
    )


# ============================================================
# [9] SELF-INTERSECTION SCREENING
# ============================================================

print()
print("[9] Self-intersection screening")
separator()

sampled_intersection = False
sample_size = 0

if ENABLE_SELF_INTERSECTION_SCREENING:

    print(
        "Full-mesh self-intersection test: DISABLED"
    )

    print(
        "Reason:"
    )

    print(
        "  Large reconstructed meshes can make a "
        "full test computationally expensive."
    )

    print(
        f"Running sampled screening on up to "
        f"{MAX_INTERSECTION_SAMPLE} triangles..."
    )

    screening_start = time.time()

    (
        sampled_intersection,
        sample_size
    ) = sample_self_intersection_check(
        mesh
    )

    screening_time = (
        time.time()
        -
        screening_start
    )

    print(
        f"Sample triangles tested : "
        f"{sample_size}"
    )

    print(
        f"Screening time          : "
        f"{screening_time:.3f} seconds"
    )

    if sampled_intersection:

        print(
            "[WARNING] Possible self-intersection "
            "detected in sampled geometry."
        )

    else:

        print(
            "[OK] No self-intersection detected "
            "in sampled geometry."
        )

else:

    print(
        "[INFO] Self-intersection screening disabled."
    )


# ============================================================
# [10] WATERTIGHT STATUS
# ============================================================

print()
print("[10] Watertightness assessment")
separator()

watertight = False

try:

    watertight = mesh.is_watertight()

except Exception:

    watertight = (
        edge_manifold
        and
        vertex_manifold
        and
        boundary_edges == 0
        and
        not sampled_intersection
    )

print(
    f"Watertight : {watertight}"
)

if watertight:

    print(
        "[OK] Mesh is watertight."
    )

else:

    print(
        "[INFO] Mesh is not watertight."
    )

    print(
        "       This does not automatically invalidate "
        "a drone reconstruction."
    )


# ============================================================
# [11] GEOMETRIC RANGE
# ============================================================

print()
print("[11] Mesh geometry")
separator()

minimum_xyz = vertices.min(
    axis=0
)

maximum_xyz = vertices.max(
    axis=0
)

print(
    f"X range : "
    f"{minimum_xyz[0]:.6f} to "
    f"{maximum_xyz[0]:.6f}"
)

print(
    f"Y range : "
    f"{minimum_xyz[1]:.6f} to "
    f"{maximum_xyz[1]:.6f}"
)

print(
    f"Z range : "
    f"{minimum_xyz[2]:.6f} to "
    f"{maximum_xyz[2]:.6f}"
)


# ============================================================
# [12] FINAL VALIDATION DECISION
# ============================================================

print()
print("[12] Final validation decision")
separator()

critical_failures = []

if invalid_vertices > 0:
    critical_failures.append(
        "non-finite vertices"
    )

if not triangle_indices_valid:
    critical_failures.append(
        "invalid triangle indices"
    )

if len(triangles) == 0:
    critical_failures.append(
        "no triangles"
    )

if repeated_vertex_triangles > 0:
    critical_failures.append(
        "repeated-vertex triangles"
    )

if zero_area_triangles > 0:
    critical_failures.append(
        "zero-area triangles"
    )

if non_manifold_edges > 0:
    critical_failures.append(
        "non-manifold edges"
    )


if len(critical_failures) == 0:

    validation_status = "PASS"

    print(
        "[PASS] Mesh passes critical geometry "
        "and topology validation."
    )

else:

    validation_status = "FAIL"

    print(
        "[FAIL] Critical mesh problems detected:"
    )

    for item in critical_failures:

        print(
            f"       - {item}"
        )


# ============================================================
# [13] SAVE VALIDATED MESH
# ============================================================

print()
print("[13] Saving validated mesh")
separator()

mesh.compute_triangle_normals()
mesh.compute_vertex_normals()

success = o3d.io.write_triangle_mesh(
    OUTPUT_MESH,
    mesh,
    write_ascii=False,
    compressed=False,
    write_vertex_normals=True,
    write_vertex_colors=True,
    write_triangle_uvs=False
)

if success:

    print(
        "[OK] Validated mesh saved:"
    )

    print(
        f"     {OUTPUT_MESH}"
    )

else:

    print(
        "[WARNING] Validated mesh could not be saved."
    )


# ============================================================
# [14] SAVE STATISTICS
# ============================================================

print()
print("[14] Saving validation statistics")
separator()

processing_time = (
    time.time()
    -
    start_time
)

with open(
    STATS_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "ARIA-S3D | PHASE 5.6\n"
    )

    f.write(
        "ROBUST MESH VALIDATION v2\n"
    )

    f.write(
        "=" * 70
        +
        "\n\n"
    )

    f.write(
        f"Validation status : "
        f"{validation_status}\n"
    )

    f.write(
        f"Vertices : "
        f"{len(vertices)}\n"
    )

    f.write(
        f"Triangles : "
        f"{len(triangles)}\n"
    )

    f.write(
        f"Invalid vertices : "
        f"{invalid_vertices}\n"
    )

    f.write(
        f"Repeated-vertex triangles : "
        f"{repeated_vertex_triangles}\n"
    )

    f.write(
        f"Minimum triangle area : "
        f"{min_area:.12f}\n"
    )

    f.write(
        f"Maximum triangle area : "
        f"{max_area:.12f}\n"
    )

    f.write(
        f"Mean triangle area : "
        f"{mean_area:.12f}\n"
    )

    f.write(
        f"Zero-area triangles : "
        f"{zero_area_triangles}\n"
    )

    f.write(
        f"Near-zero triangles : "
        f"{near_zero_triangles}\n"
    )

    f.write(
        f"Duplicate triangles : "
        f"{duplicate_triangles}\n"
    )

    f.write(
        f"Edge manifold : "
        f"{edge_manifold}\n"
    )

    f.write(
        f"Closed edge manifold : "
        f"{closed_edge_manifold}\n"
    )

    f.write(
        f"Vertex manifold : "
        f"{vertex_manifold}\n"
    )

    f.write(
        f"Orientable : "
        f"{orientable}\n"
    )

    f.write(
        f"Non-manifold edges : "
        f"{non_manifold_edges}\n"
    )

    f.write(
        f"Non-manifold vertices : "
        f"{non_manifold_vertices}\n"
    )

    f.write(
        f"Boundary edges : "
        f"{boundary_edges}\n"
    )

    f.write(
        f"Watertight : "
        f"{watertight}\n"
    )

    f.write(
        f"Sample intersection test : "
        f"{sampled_intersection}\n"
    )

    f.write(
        f"Intersection sample size : "
        f"{sample_size}\n"
    )

    f.write(
        f"Mesh scale : "
        f"{mesh_scale:.6f}\n"
    )

    f.write(
        f"Processing time : "
        f"{processing_time:.3f} seconds\n"
    )


print(
    "[OK] Validation statistics saved:"
)

print(
    f"     {STATS_FILE}"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("ARIA-S3D | PHASE 5.6 VALIDATION SUMMARY")
print("=" * 70)

print()
print(
    f"Validation status       : {validation_status}"
)

print(
    f"Vertices                : {len(vertices)}"
)

print(
    f"Triangles               : {len(triangles)}"
)

print(
    f"Duplicate triangles     : {duplicate_triangles}"
)

print(
    f"Near-zero triangles     : {near_zero_triangles}"
)

print(
    f"Non-manifold edges      : {non_manifold_edges}"
)

print(
    f"Non-manifold vertices   : {non_manifold_vertices}"
)

print(
    f"Boundary edges          : {boundary_edges}"
)

print(
    f"Watertight              : {watertight}"
)

print(
    f"Sampled self-intersect  : "
    f"{sampled_intersection}"
)

print(
    f"Processing time         : "
    f"{processing_time:.3f} seconds"
)

print()

if validation_status == "PASS":

    print(
        "[PASS] PHASE 5.6 MESH VALIDATION SUCCESS"
    )

    print()
    print(
        "The reconstructed mesh passed all critical"
    )

    print(
        "geometry and topology checks."
    )

    print()
    print(
        "NEXT STEP:"
    )

    print(
        "Phase 5.7 - Adaptive mesh optimization"
    )

else:

    print(
        "[FAIL] PHASE 5.6 VALIDATION FAILED"
    )

    print()
    print(
        "Critical mesh problems must be resolved"
    )

    print(
        "before continuing to Phase 5.7."
    )


print()
print("IMPORTANT:")
print(
    "ARIA-S3D remains a monocular reconstruction."
)

print(
    "Absolute metric scale remains arbitrary."
)

print(
    "Open boundaries and non-watertight geometry are "
    "not automatically failures for outdoor drone scenes."
)

print(
    "Full-mesh self-intersection testing is intentionally "
    "avoided for large meshes."
)

print("=" * 70)


# ============================================================
# EXIT STATUS
# ============================================================

if validation_status == "FAIL":
    sys.exit(1)

sys.exit(0)

