import os
import time
import numpy as np
import open3d as o3d


# ============================================================
# ARIA-S3D | PHASE 5.5
# MESH CLEANUP AND VALIDATION
# ============================================================

INPUT_MESH = os.path.join(
    "data",
    "output",
    "reconstructed_mesh.ply"
)

OUTPUT_DIR = os.path.join(
    "data",
    "output"
)

CLEANED_PLY = os.path.join(
    OUTPUT_DIR,
    "cleaned_mesh.ply"
)

CLEANED_OBJ = os.path.join(
    OUTPUT_DIR,
    "cleaned_mesh.obj"
)

STATS_FILE = os.path.join(
    OUTPUT_DIR,
    "mesh_cleanup_stats.txt"
)


# ============================================================
# CONFIGURATION
# ============================================================

REMOVE_NON_MANIFOLD = True
RECOMPUTE_NORMALS = True

# A triangle is considered extremely small if its area is
# below this threshold.
MIN_TRIANGLE_AREA = 1e-12


# ============================================================
# HELPERS
# ============================================================

def print_section(number, title):
    print()
    print(f"[{number}] {title}")
    print("-" * 70)


def finite_array(array):
    array = np.asarray(array)

    if array.size == 0:
        return True

    return bool(np.isfinite(array).all())


def mesh_geometry_valid(mesh):
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    if len(vertices) == 0:
        return False

    if len(triangles) == 0:
        return False

    if not finite_array(vertices):
        return False

    if not finite_array(triangles):
        return False

    if triangles.min() < 0:
        return False

    if triangles.max() >= len(vertices):
        return False

    return True


def geometry_range(mesh):
    vertices = np.asarray(mesh.vertices)

    if len(vertices) == 0:
        return None

    return (
        vertices.min(axis=0),
        vertices.max(axis=0)
    )


def triangle_areas(mesh):
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    if len(triangles) == 0:
        return np.empty(0, dtype=np.float64)

    a = vertices[triangles[:, 0]]
    b = vertices[triangles[:, 1]]
    c = vertices[triangles[:, 2]]

    cross = np.cross(
        b - a,
        c - a
    )

    areas = 0.5 * np.linalg.norm(
        cross,
        axis=1
    )

    return areas


def count_repeated_vertex_triangles(mesh):
    triangles = np.asarray(mesh.triangles)

    if len(triangles) == 0:
        return 0

    invalid = (
        (triangles[:, 0] == triangles[:, 1]) |
        (triangles[:, 1] == triangles[:, 2]) |
        (triangles[:, 0] == triangles[:, 2])
    )

    return int(np.count_nonzero(invalid))


def mesh_statistics(mesh):
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    stats = {}

    stats["vertices"] = len(vertices)
    stats["triangles"] = len(triangles)

    if len(vertices) > 0:
        stats["finite_vertices"] = int(
            np.isfinite(vertices).all(axis=1).sum()
        )

        stats["x_min"] = float(vertices[:, 0].min())
        stats["x_max"] = float(vertices[:, 0].max())

        stats["y_min"] = float(vertices[:, 1].min())
        stats["y_max"] = float(vertices[:, 1].max())

        stats["z_min"] = float(vertices[:, 2].min())
        stats["z_max"] = float(vertices[:, 2].max())

    else:
        stats["finite_vertices"] = 0

    areas = triangle_areas(mesh)

    if len(areas) > 0:

        stats["min_triangle_area"] = float(
            areas.min()
        )

        stats["max_triangle_area"] = float(
            areas.max()
        )

        stats["mean_triangle_area"] = float(
            areas.mean()
        )

        stats["zero_area_triangles"] = int(
            np.count_nonzero(areas <= MIN_TRIANGLE_AREA)
        )

        stats["nonfinite_triangle_areas"] = int(
            np.count_nonzero(~np.isfinite(areas))
        )

    else:

        stats["min_triangle_area"] = 0.0
        stats["max_triangle_area"] = 0.0
        stats["mean_triangle_area"] = 0.0
        stats["zero_area_triangles"] = 0
        stats["nonfinite_triangle_areas"] = 0

    stats["repeated_vertex_triangles"] = (
        count_repeated_vertex_triangles(mesh)
    )

    try:
        stats["non_manifold_edges"] = len(
            mesh.get_non_manifold_edges(
                allow_boundary_edges=True
            )
        )
    except Exception:
        stats["non_manifold_edges"] = -1

    try:
        stats["non_manifold_vertices"] = len(
            mesh.get_non_manifold_vertices()
        )
    except Exception:
        stats["non_manifold_vertices"] = -1

    try:
        stats["boundary_edges"] = len(
            mesh.get_non_manifold_edges(
                allow_boundary_edges=False
            )
        )
    except Exception:
        stats["boundary_edges"] = -1

    try:
        stats["watertight"] = bool(
            mesh.is_watertight()
        )
    except Exception:
        stats["watertight"] = False

    return stats


def print_mesh_statistics(stats, prefix=""):

    print(
        f"{prefix}Vertices : "
        f"{stats['vertices']}"
    )

    print(
        f"{prefix}Triangles: "
        f"{stats['triangles']}"
    )

    print(
        f"{prefix}Finite vertices: "
        f"{stats['finite_vertices']}"
    )

    print(
        f"{prefix}Minimum triangle area : "
        f"{stats['min_triangle_area']:.12f}"
    )

    print(
        f"{prefix}Maximum triangle area : "
        f"{stats['max_triangle_area']:.12f}"
    )

    print(
        f"{prefix}Mean triangle area : "
        f"{stats['mean_triangle_area']:.12f}"
    )

    print(
        f"{prefix}Zero-area triangles : "
        f"{stats['zero_area_triangles']}"
    )

    print(
        f"{prefix}Repeated-vertex triangles : "
        f"{stats['repeated_vertex_triangles']}"
    )

    print(
        f"{prefix}Non-manifold edges : "
        f"{stats['non_manifold_edges']}"
    )

    print(
        f"{prefix}Non-manifold vertices : "
        f"{stats['non_manifold_vertices']}"
    )

    print(
        f"{prefix}Boundary/non-manifold edges : "
        f"{stats['boundary_edges']}"
    )

    print(
        f"{prefix}Watertight : "
        f"{stats['watertight']}"
    )


# ============================================================
# START
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

print("=" * 70)
print("ARIA-S3D | PHASE 5.5")
print("MESH CLEANUP AND VALIDATION")
print("=" * 70)


# ============================================================
# 1. CHECK INPUT
# ============================================================

print_section(
    1,
    "Checking reconstructed mesh"
)

if not os.path.exists(INPUT_MESH):

    print(
        "[ERROR] Reconstructed mesh not found:"
    )

    print(
        f"        {INPUT_MESH}"
    )

    raise SystemExit(1)

print(
    "[OK] Input mesh found:"
)

print(
    f"     {INPUT_MESH}"
)


# ============================================================
# 2. LOAD MESH
# ============================================================

print_section(
    2,
    "Loading reconstructed mesh"
)

mesh = o3d.io.read_triangle_mesh(
    INPUT_MESH,
    enable_post_processing=True
)

if mesh.is_empty():

    print(
        "[ERROR] Open3D loaded an empty mesh."
    )

    raise SystemExit(1)

print(
    f"Vertices  : {len(mesh.vertices)}"
)

print(
    f"Triangles : {len(mesh.triangles)}"
)


# ============================================================
# 3. INITIAL VALIDATION
# ============================================================

print_section(
    3,
    "Validating input mesh"
)

vertices = np.asarray(
    mesh.vertices
)

triangles = np.asarray(
    mesh.triangles
)

if not finite_array(vertices):

    print(
        "[ERROR] Mesh contains non-finite vertices."
    )

    raise SystemExit(1)

print(
    "[OK] All vertex coordinates are finite."
)


if len(triangles) == 0:

    print(
        "[ERROR] Mesh contains no triangles."
    )

    raise SystemExit(1)


if triangles.min() < 0:

    print(
        "[ERROR] Negative triangle index detected."
    )

    raise SystemExit(1)


if triangles.max() >= len(vertices):

    print(
        "[ERROR] Triangle index exceeds vertex count."
    )

    raise SystemExit(1)

print(
    "[OK] Triangle indices are valid."
)


# ============================================================
# 4. ORIGINAL STATISTICS
# ============================================================

print_section(
    4,
    "Computing original mesh statistics"
)

original_stats = mesh_statistics(
    mesh
)

print_mesh_statistics(
    original_stats
)

original_vertices = original_stats["vertices"]
original_triangles = original_stats["triangles"]


# ============================================================
# 5. REMOVE DUPLICATED VERTICES
# ============================================================

print_section(
    5,
    "Removing duplicated vertices"
)

before = len(mesh.vertices)

mesh.remove_duplicated_vertices()

after = len(mesh.vertices)

removed = before - after

print(
    f"Vertices before : {before}"
)

print(
    f"Vertices after  : {after}"
)

print(
    f"Duplicated vertices removed : {removed}"
)

print(
    "[OK] Duplicated vertex cleanup completed."
)


# ============================================================
# 6. REMOVE DUPLICATED TRIANGLES
# ============================================================

print_section(
    6,
    "Removing duplicated triangles"
)

before = len(mesh.triangles)

mesh.remove_duplicated_triangles()

after = len(mesh.triangles)

removed = before - after

print(
    f"Triangles before : {before}"
)

print(
    f"Triangles after  : {after}"
)

print(
    f"Duplicated triangles removed : {removed}"
)

print(
    "[OK] Duplicated triangle cleanup completed."
)


# ============================================================
# 7. REMOVE DEGENERATE TRIANGLES
# ============================================================

print_section(
    7,
    "Removing degenerate triangles"
)

before = len(mesh.triangles)

mesh.remove_degenerate_triangles()

after = len(mesh.triangles)

removed = before - after

print(
    f"Triangles before : {before}"
)

print(
    f"Triangles after  : {after}"
)

print(
    f"Degenerate triangles removed : {removed}"
)

print(
    "[OK] Degenerate triangle cleanup completed."
)


# ============================================================
# 8. REMOVE UNREFERENCED VERTICES
# ============================================================

print_section(
    8,
    "Removing unreferenced vertices"
)

before = len(mesh.vertices)

mesh.remove_unreferenced_vertices()

after = len(mesh.vertices)

removed = before - after

print(
    f"Vertices before : {before}"
)

print(
    f"Vertices after  : {after}"
)

print(
    f"Unreferenced vertices removed : {removed}"
)

print(
    "[OK] Unreferenced vertex cleanup completed."
)


# ============================================================
# 9. NON-MANIFOLD EDGE ANALYSIS
# ============================================================

print_section(
    9,
    "Analyzing mesh topology"
)

try:

    non_manifold_edges_before = len(
        mesh.get_non_manifold_edges(
            allow_boundary_edges=True
        )
    )

except Exception:

    non_manifold_edges_before = -1

try:

    non_manifold_vertices_before = len(
        mesh.get_non_manifold_vertices()
    )

except Exception:

    non_manifold_vertices_before = -1


print(
    f"Non-manifold edges detected   : "
    f"{non_manifold_edges_before}"
)

print(
    f"Non-manifold vertices detected: "
    f"{non_manifold_vertices_before}"
)


# ============================================================
# 10. REMOVE NON-MANIFOLD EDGES
# ============================================================

print_section(
    10,
    "Removing non-manifold edges"
)

if REMOVE_NON_MANIFOLD:

    if non_manifold_edges_before > 0:

        print(
            "Non-manifold edges detected."
        )

        print(
            "Removing problematic triangles..."
        )

        mesh.remove_non_manifold_edges()

        print(
            "[OK] Non-manifold edge cleanup completed."
        )

    else:

        print(
            "[OK] No non-manifold edges require removal."
        )

else:

    print(
        "[INFO] Non-manifold cleanup disabled."
    )


# ============================================================
# 11. FINAL BASIC CLEANUP
# ============================================================

print_section(
    11,
    "Running final mesh cleanup"
)

mesh.remove_duplicated_vertices()

mesh.remove_duplicated_triangles()

mesh.remove_degenerate_triangles()

mesh.remove_unreferenced_vertices()

print(
    "[OK] Final cleanup pass completed."
)


# ============================================================
# 12. ORIENT TRIANGLES
# ============================================================

print_section(
    12,
    "Orienting mesh triangles"
)

try:

    oriented = mesh.orient_triangles()

    if oriented:

        print(
            "[OK] Mesh triangles oriented consistently."
        )

    else:

        print(
            "[WARNING] Mesh could not be fully oriented."
        )

except Exception as exc:

    print(
        "[WARNING] Triangle orientation failed:"
    )

    print(
        f"          {exc}"
    )


# ============================================================
# 13. COMPUTE NORMALS
# ============================================================

print_section(
    13,
    "Recomputing mesh normals"
)

if RECOMPUTE_NORMALS:

    try:

        mesh.compute_triangle_normals()

        mesh.compute_vertex_normals()

        mesh.normalize_normals()

        print(
            "[OK] Triangle and vertex normals computed."
        )

    except Exception as exc:

        print(
            "[WARNING] Normal computation failed:"
        )

        print(
            f"          {exc}"
        )

else:

    print(
        "[INFO] Normal recomputation disabled."
    )


# ============================================================
# 14. FINAL VALIDATION
# ============================================================

print_section(
    14,
    "Validating cleaned mesh"
)

if not mesh_geometry_valid(mesh):

    print(
        "[ERROR] Cleaned mesh failed geometry validation."
    )

    raise SystemExit(1)

print(
    "[OK] Cleaned mesh geometry is finite."
)

clean_vertices = np.asarray(
    mesh.vertices
)

clean_triangles = np.asarray(
    mesh.triangles
)

print(
    "[OK] Triangle indices are valid."
)


# ============================================================
# 15. FINAL TOPOLOGY CHECK
# ============================================================

print_section(
    15,
    "Final topology validation"
)

try:

    final_non_manifold_edges = len(
        mesh.get_non_manifold_edges(
            allow_boundary_edges=True
        )
    )

except Exception:

    final_non_manifold_edges = -1


try:

    final_non_manifold_vertices = len(
        mesh.get_non_manifold_vertices()
    )

except Exception:

    final_non_manifold_vertices = -1


print(
    f"Final non-manifold edges   : "
    f"{final_non_manifold_edges}"
)

print(
    f"Final non-manifold vertices: "
    f"{final_non_manifold_vertices}"
)


final_repeated_triangles = (
    count_repeated_vertex_triangles(mesh)
)

print(
    f"Repeated-vertex triangles  : "
    f"{final_repeated_triangles}"
)


final_areas = triangle_areas(
    mesh
)

if len(final_areas) > 0:

    final_zero_area = int(
        np.count_nonzero(
            final_areas <= MIN_TRIANGLE_AREA
        )
    )

else:

    final_zero_area = 0


print(
    f"Near-zero-area triangles   : "
    f"{final_zero_area}"
)


if final_repeated_triangles == 0:

    print(
        "[OK] No repeated-vertex triangles remain."
    )

else:

    print(
        "[WARNING] Repeated-vertex triangles remain."
    )


if final_zero_area == 0:

    print(
        "[OK] No near-zero-area triangles remain."
    )

else:

    print(
        "[WARNING] Near-zero-area triangles remain."
    )


# ============================================================
# 16. CLEAN GEOMETRY
# ============================================================

print_section(
    16,
    "Computing cleaned mesh geometry"
)

clean_stats = mesh_statistics(
    mesh
)

print(
    f"X range : "
    f"{clean_stats['x_min']:.6f} "
    f"to "
    f"{clean_stats['x_max']:.6f}"
)

print(
    f"Y range : "
    f"{clean_stats['y_min']:.6f} "
    f"to "
    f"{clean_stats['y_max']:.6f}"
)

print(
    f"Z range : "
    f"{clean_stats['z_min']:.6f} "
    f"to "
    f"{clean_stats['z_max']:.6f}"
)


# ============================================================
# 17. CLEANUP STATISTICS
# ============================================================

print_section(
    17,
    "Mesh cleanup statistics"
)

cleaned_vertices = clean_stats["vertices"]
cleaned_triangles = clean_stats["triangles"]

vertices_removed = (
    original_vertices -
    cleaned_vertices
)

triangles_removed = (
    original_triangles -
    cleaned_triangles
)

if original_vertices > 0:

    vertex_retention = (
        cleaned_vertices /
        original_vertices
    ) * 100.0

else:

    vertex_retention = 0.0


if original_triangles > 0:

    triangle_retention = (
        cleaned_triangles /
        original_triangles
    ) * 100.0

else:

    triangle_retention = 0.0


print(
    f"Original vertices     : "
    f"{original_vertices}"
)

print(
    f"Cleaned vertices      : "
    f"{cleaned_vertices}"
)

print(
    f"Vertices removed      : "
    f"{vertices_removed}"
)

print(
    f"Vertex retention      : "
    f"{vertex_retention:.2f}%"
)

print()

print(
    f"Original triangles    : "
    f"{original_triangles}"
)

print(
    f"Cleaned triangles     : "
    f"{cleaned_triangles}"
)

print(
    f"Triangles removed     : "
    f"{triangles_removed}"
)

print(
    f"Triangle retention    : "
    f"{triangle_retention:.2f}%"
)


# ============================================================
# 18. SAVE CLEANED PLY
# ============================================================

print_section(
    18,
    "Saving cleaned PLY mesh"
)

ply_success = o3d.io.write_triangle_mesh(
    CLEANED_PLY,
    mesh,
    write_ascii=False,
    compressed=False,
    write_vertex_normals=True,
    write_vertex_colors=True,
    write_triangle_uvs=True
)

if not ply_success:

    print(
        "[ERROR] Failed to save cleaned PLY."
    )

    raise SystemExit(1)

print(
    "[OK] Cleaned PLY saved:"
)

print(
    f"     {CLEANED_PLY}"
)


# ============================================================
# 19. SAVE CLEANED OBJ
# ============================================================

print_section(
    19,
    "Saving cleaned OBJ mesh"
)

obj_success = o3d.io.write_triangle_mesh(
    CLEANED_OBJ,
    mesh,
    write_ascii=False,
    compressed=False,
    write_vertex_normals=True,
    write_vertex_colors=True,
    write_triangle_uvs=True
)

if obj_success:

    print(
        "[OK] Cleaned OBJ saved:"
    )

    print(
        f"     {CLEANED_OBJ}"
    )

else:

    print(
        "[WARNING] OBJ export failed."
    )


# ============================================================
# 20. SAVE STATISTICS
# ============================================================

print_section(
    20,
    "Saving cleanup statistics"
)

with open(
    STATS_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "ARIA-S3D | PHASE 5.5\n"
    )

    f.write(
        "MESH CLEANUP AND VALIDATION\n"
    )

    f.write(
        "=" * 70 + "\n\n"
    )

    f.write(
        "INPUT\n"
    )

    f.write(
        f"Input mesh: {INPUT_MESH}\n\n"
    )

    f.write(
        "ORIGINAL MESH\n"
    )

    f.write(
        f"Vertices: "
        f"{original_stats['vertices']}\n"
    )

    f.write(
        f"Triangles: "
        f"{original_stats['triangles']}\n"
    )

    f.write(
        f"Zero-area triangles: "
        f"{original_stats['zero_area_triangles']}\n"
    )

    f.write(
        f"Repeated-vertex triangles: "
        f"{original_stats['repeated_vertex_triangles']}\n"
    )

    f.write(
        f"Non-manifold edges: "
        f"{original_stats['non_manifold_edges']}\n"
    )

    f.write(
        f"Non-manifold vertices: "
        f"{original_stats['non_manifold_vertices']}\n"
    )

    f.write(
        f"Watertight: "
        f"{original_stats['watertight']}\n\n"
    )

    f.write(
        "CLEANED MESH\n"
    )

    f.write(
        f"Vertices: "
        f"{clean_stats['vertices']}\n"
    )

    f.write(
        f"Triangles: "
        f"{clean_stats['triangles']}\n"
    )

    f.write(
        f"Zero-area triangles: "
        f"{clean_stats['zero_area_triangles']}\n"
    )

    f.write(
        f"Repeated-vertex triangles: "
        f"{clean_stats['repeated_vertex_triangles']}\n"
    )

    f.write(
        f"Non-manifold edges: "
        f"{final_non_manifold_edges}\n"
    )

    f.write(
        f"Non-manifold vertices: "
        f"{final_non_manifold_vertices}\n"
    )

    f.write(
        f"Watertight: "
        f"{clean_stats['watertight']}\n\n"
    )

    f.write(
        "REMOVAL STATISTICS\n"
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

    f.write(
        "OUTPUTS\n"
    )

    f.write(
        f"PLY: {CLEANED_PLY}\n"
    )

    f.write(
        f"OBJ: {CLEANED_OBJ}\n"
    )

print(
    "[OK] Statistics saved:"
)

print(
    f"     {STATS_FILE}"
)


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 70)
print("ARIA-S3D | PHASE 5.5 COMPLETE")
print("=" * 70)

print()

print(
    f"Original vertices     : "
    f"{original_vertices}"
)

print(
    f"Cleaned vertices      : "
    f"{cleaned_vertices}"
)

print(
    f"Original triangles    : "
    f"{original_triangles}"
)

print(
    f"Cleaned triangles     : "
    f"{cleaned_triangles}"
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

print(
    f"Final non-manifold edges   : "
    f"{final_non_manifold_edges}"
)

print(
    f"Final non-manifold vertices: "
    f"{final_non_manifold_vertices}"
)

print(
    f"Near-zero-area triangles   : "
    f"{final_zero_area}"
)

print()

print(
    "Generated outputs:"
)

print(
    f"  {CLEANED_PLY}"
)

print(
    f"  {CLEANED_OBJ}"
)

print(
    f"  {STATS_FILE}"
)

print()

print(
    "NEXT STEP:"
)

print(
    "Phase 5.6 - Mesh validation"
)

print()

print(
    "IMPORTANT:"
)

print(
    "The reconstruction remains monocular."
)

print(
    "Absolute metric scale remains arbitrary."
)

print(
    "Mesh cleanup removes redundant and problematic "
    "topological elements while preserving the reconstructed geometry."
)

print(
    "=" * 70
)

