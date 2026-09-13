"""
ARIA-S3D
PHASE 5.7
ADAPTIVE MESH OPTIMIZATION

Purpose:
    Optimize a reconstructed mesh for downstream visualization,
    texturing and real-time use while preserving scene geometry.

Design goals:
    - Works with different reconstruction sizes
    - Avoids hard-coded scene dimensions
    - Automatically determines simplification strength
    - Preserves boundaries where possible
    - Removes redundant / degenerate geometry
    - Validates the optimized mesh
    - Produces PLY + OBJ + statistics
    - Does NOT assume metric scale

Input:
    data/output/validated_mesh.ply

Outputs:
    data/output/optimized_mesh.ply
    data/output/optimized_mesh.obj
    data/output/adaptive_mesh_optimization_stats.txt
"""

from pathlib import Path
import time
import numpy as np
import open3d as o3d


# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "data" / "output"

INPUT_MESH = OUTPUT_DIR / "validated_mesh.ply"

OUTPUT_PLY = OUTPUT_DIR / "optimized_mesh.ply"
OUTPUT_OBJ = OUTPUT_DIR / "optimized_mesh.obj"
OUTPUT_STATS = OUTPUT_DIR / "adaptive_mesh_optimization_stats.txt"


# Adaptive limits.
# These are percentages, not scene-specific dimensions.

MIN_REDUCTION = 0.15
MAX_REDUCTION = 0.60

# Target triangle count bounds.
MIN_TARGET_TRIANGLES = 10000
MAX_TARGET_TRIANGLES = 150000

# Never simplify extremely small meshes aggressively.
SMALL_MESH_TRIANGLES = 20000

# Geometry validation threshold.
FINITE_EPS = 1e-12


# ============================================================================
# PRINT HELPERS
# ============================================================================

def header(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def section(number, title):
    print()
    print(f"[{number}] {title}")
    print("-" * 70)


def ok(message):
    print(f"[OK] {message}")


def warning(message):
    print(f"[WARNING] {message}")


def info(message):
    print(f"[INFO] {message}")


# ============================================================================
# GEOMETRY HELPERS
# ============================================================================

def get_vertices(mesh):
    return np.asarray(mesh.vertices)


def get_triangles(mesh):
    return np.asarray(mesh.triangles)


def geometry_is_finite(mesh):
    vertices = get_vertices(mesh)

    if len(vertices) == 0:
        return False

    return bool(np.all(np.isfinite(vertices)))


def triangle_area_statistics(mesh):
    vertices = get_vertices(mesh)
    triangles = get_triangles(mesh)

    if len(triangles) == 0:
        return 0.0, 0.0, 0.0, 0

    a = vertices[triangles[:, 0]]
    b = vertices[triangles[:, 1]]
    c = vertices[triangles[:, 2]]

    cross = np.cross(b - a, c - a)
    areas = 0.5 * np.linalg.norm(cross, axis=1)

    finite_areas = areas[np.isfinite(areas)]

    if len(finite_areas) == 0:
        return 0.0, 0.0, 0.0, len(areas)

    near_zero_threshold = max(
        np.finfo(np.float64).eps,
        np.max(finite_areas) * 1e-8
    )

    near_zero = int(np.sum(finite_areas <= near_zero_threshold))

    return (
        float(np.min(finite_areas)),
        float(np.max(finite_areas)),
        float(np.mean(finite_areas)),
        near_zero
    )


def bounding_box_scale(mesh):
    extent = mesh.get_max_bound() - mesh.get_min_bound()

    if len(extent) == 0:
        return 0.0

    return float(np.max(extent))


# ============================================================================
# ADAPTIVE TARGET CALCULATION
# ============================================================================

def calculate_adaptive_target(mesh):
    triangles = len(mesh.triangles)

    if triangles <= 0:
        raise RuntimeError("Input mesh contains no triangles.")

    # Small meshes should be preserved.
    if triangles <= SMALL_MESH_TRIANGLES:
        reduction = MIN_REDUCTION

    else:
        # Smooth adaptive scaling based on mesh size.
        #
        # Larger meshes receive stronger simplification,
        # but the reduction is clamped to safe limits.
        normalized = np.log10(max(triangles, 1))

        # Approximately:
        # 20k triangles -> ~15%
        # 100k triangles -> ~35%
        # 1M triangles -> ~60%
        reduction = 0.15 + (
            (normalized - np.log10(20000))
            / (np.log10(1000000) - np.log10(20000))
        ) * 0.45

        reduction = float(
            np.clip(
                reduction,
                MIN_REDUCTION,
                MAX_REDUCTION
            )
        )

    target = int(round(triangles * (1.0 - reduction)))

    target = max(target, MIN_TARGET_TRIANGLES)
    target = min(target, MAX_TARGET_TRIANGLES)

    # Never request more triangles than the input.
    target = min(target, triangles)

    actual_reduction = 1.0 - (target / triangles)

    return target, actual_reduction


# ============================================================================
# MESH CLEANUP
# ============================================================================

def cleanup_mesh(mesh):
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()

    return mesh


def validate_triangle_indices(mesh):
    vertices = get_vertices(mesh)
    triangles = get_triangles(mesh)

    if len(triangles) == 0:
        return False

    if np.min(triangles) < 0:
        return False

    if np.max(triangles) >= len(vertices):
        return False

    return True


def count_repeated_vertex_triangles(mesh):
    triangles = get_triangles(mesh)

    if len(triangles) == 0:
        return 0

    repeated = (
        (triangles[:, 0] == triangles[:, 1])
        |
        (triangles[:, 1] == triangles[:, 2])
        |
        (triangles[:, 0] == triangles[:, 2])
    )

    return int(np.sum(repeated))


# ============================================================================
# MAIN
# ============================================================================

def main():

    start_time = time.perf_counter()

    header("ARIA-S3D | PHASE 5.7\nADAPTIVE MESH OPTIMIZATION")

    # ----------------------------------------------------------------------
    # 1. INPUT CHECK
    # ----------------------------------------------------------------------

    section(1, "Checking validated mesh")

    if not INPUT_MESH.exists():
        raise FileNotFoundError(
            f"Input mesh not found:\n{INPUT_MESH}"
        )

    ok(f"Input mesh found:\n     {INPUT_MESH}")

    # ----------------------------------------------------------------------
    # 2. LOAD
    # ----------------------------------------------------------------------

    section(2, "Loading validated mesh")

    mesh = o3d.io.read_triangle_mesh(str(INPUT_MESH))

    if mesh.is_empty():
        raise RuntimeError("Failed to load mesh or mesh is empty.")

    original_vertices = len(mesh.vertices)
    original_triangles = len(mesh.triangles)

    print(f"Vertices  : {original_vertices}")
    print(f"Triangles : {original_triangles}")

    # ----------------------------------------------------------------------
    # 3. INPUT VALIDATION
    # ----------------------------------------------------------------------

    section(3, "Validating input geometry")

    if not geometry_is_finite(mesh):
        raise RuntimeError(
            "Input mesh contains non-finite vertex coordinates."
        )

    ok("All vertex coordinates are finite.")

    if not validate_triangle_indices(mesh):
        raise RuntimeError(
            "Input mesh contains invalid triangle indices."
        )

    ok("All triangle indices are valid.")

    repeated = count_repeated_vertex_triangles(mesh)

    print(f"Repeated-vertex triangles : {repeated}")

    if repeated > 0:
        warning(
            f"{repeated} repeated-vertex triangles detected."
        )
        cleanup_mesh(mesh)
    else:
        ok("No repeated-vertex triangles.")

    # ----------------------------------------------------------------------
    # 4. INPUT GEOMETRY
    # ----------------------------------------------------------------------

    section(4, "Computing input geometry")

    scale = bounding_box_scale(mesh)

    min_bound = mesh.get_min_bound()
    max_bound = mesh.get_max_bound()

    print(f"Mesh scale : {scale:.6f}")

    print(
        f"X range : {min_bound[0]:.6f} to {max_bound[0]:.6f}"
    )

    print(
        f"Y range : {min_bound[1]:.6f} to {max_bound[1]:.6f}"
    )

    print(
        f"Z range : {min_bound[2]:.6f} to {max_bound[2]:.6f}"
    )

    # ----------------------------------------------------------------------
    # 5. TRIANGLE STATISTICS
    # ----------------------------------------------------------------------

    section(5, "Analyzing input triangle quality")

    (
        min_area,
        max_area,
        mean_area,
        near_zero
    ) = triangle_area_statistics(mesh)

    print(f"Minimum triangle area : {min_area:.12e}")
    print(f"Maximum triangle area : {max_area:.12e}")
    print(f"Mean triangle area    : {mean_area:.12e}")
    print(f"Near-zero triangles   : {near_zero}")

    # ----------------------------------------------------------------------
    # 6. ADAPTIVE TARGET
    # ----------------------------------------------------------------------

    section(6, "Computing adaptive optimization target")

    target_triangles, reduction = calculate_adaptive_target(mesh)

    print(f"Input triangles       : {len(mesh.triangles)}")
    print(f"Adaptive target       : {target_triangles}")
    print(f"Requested reduction   : {reduction * 100.0:.2f}%")

    if reduction <= 0:
        warning(
            "Adaptive optimizer selected no triangle reduction."
        )

    else:
        ok("Adaptive target calculated from input mesh size.")

    # ----------------------------------------------------------------------
    # 7. QUADRIC DECIMATION
    # ----------------------------------------------------------------------

    section(7, "Running adaptive quadric decimation")

    print(
        "Optimization method : Quadric Error Metric Decimation"
    )

    print(
        f"Target triangles     : {target_triangles}"
    )

    print(
        "Preserving geometric shape while reducing redundant triangles..."
    )

    optimization_start = time.perf_counter()

    optimized = mesh.simplify_quadric_decimation(
        target_number_of_triangles=target_triangles
    )

    optimization_time = time.perf_counter() - optimization_start

    if optimized is None or optimized.is_empty():
        raise RuntimeError(
            "Adaptive mesh optimization returned an empty mesh."
        )

    ok("Quadric decimation completed.")

    print(
        f"Optimization time : {optimization_time:.3f} seconds"
    )

    # ----------------------------------------------------------------------
    # 8. POST-OPTIMIZATION CLEANUP
    # ----------------------------------------------------------------------

    section(8, "Cleaning optimized mesh")

    before_cleanup_vertices = len(optimized.vertices)
    before_cleanup_triangles = len(optimized.triangles)

    cleanup_mesh(optimized)

    after_cleanup_vertices = len(optimized.vertices)
    after_cleanup_triangles = len(optimized.triangles)

    print(
        f"Vertices before cleanup  : {before_cleanup_vertices}"
    )

    print(
        f"Vertices after cleanup   : {after_cleanup_vertices}"
    )

    print(
        f"Triangles before cleanup : {before_cleanup_triangles}"
    )

    print(
        f"Triangles after cleanup  : {after_cleanup_triangles}"
    )

    ok("Post-optimization cleanup completed.")

    # ----------------------------------------------------------------------
    # 9. COMPUTE NORMALS
    # ----------------------------------------------------------------------

    section(9, "Recomputing optimized mesh normals")

    optimized.compute_triangle_normals()
    optimized.compute_vertex_normals()
    optimized.normalize_normals()

    ok("Optimized mesh normals computed.")

    # ----------------------------------------------------------------------
    # 10. VALIDATE OPTIMIZED GEOMETRY
    # ----------------------------------------------------------------------

    section(10, "Validating optimized mesh")

    if not geometry_is_finite(optimized):
        raise RuntimeError(
            "Optimized mesh contains non-finite coordinates."
        )

    ok("Optimized geometry is finite.")

    if not validate_triangle_indices(optimized):
        raise RuntimeError(
            "Optimized mesh contains invalid triangle indices."
        )

    ok("Optimized triangle indices are valid.")

    repeated_after = count_repeated_vertex_triangles(optimized)

    print(
        f"Repeated-vertex triangles : {repeated_after}"
    )

    if repeated_after == 0:
        ok("No repeated-vertex triangles remain.")
    else:
        warning(
            f"{repeated_after} repeated-vertex triangles remain."
        )

    # ----------------------------------------------------------------------
    # 11. OPTIMIZED GEOMETRY
    # ----------------------------------------------------------------------

    section(11, "Computing optimized geometry")

    optimized_min = optimized.get_min_bound()
    optimized_max = optimized.get_max_bound()

    optimized_scale = bounding_box_scale(optimized)

    print(f"Mesh scale : {optimized_scale:.6f}")

    print(
        f"X range : {optimized_min[0]:.6f} to {optimized_max[0]:.6f}"
    )

    print(
        f"Y range : {optimized_min[1]:.6f} to {optimized_max[1]:.6f}"
    )

    print(
        f"Z range : {optimized_min[2]:.6f} to {optimized_max[2]:.6f}"
    )

    # ----------------------------------------------------------------------
    # 12. FINAL TRIANGLE STATISTICS
    # ----------------------------------------------------------------------

    section(12, "Computing optimized mesh statistics")

    final_min_area, final_max_area, final_mean_area, final_near_zero = (
        triangle_area_statistics(optimized)
    )

    final_vertices = len(optimized.vertices)
    final_triangles = len(optimized.triangles)

    vertex_reduction = (
        1.0 -
        final_vertices / max(original_vertices, 1)
    )

    triangle_reduction = (
        1.0 -
        final_triangles / max(original_triangles, 1)
    )

    vertex_retention = (
        final_vertices /
        max(original_vertices, 1)
    )

    triangle_retention = (
        final_triangles /
        max(original_triangles, 1)
    )

    print(f"Original vertices     : {original_vertices}")
    print(f"Optimized vertices    : {final_vertices}")
    print(
        f"Vertices removed      : "
        f"{original_vertices - final_vertices}"
    )
    print(
        f"Vertex reduction      : "
        f"{vertex_reduction * 100.0:.2f}%"
    )

    print()

    print(f"Original triangles     : {original_triangles}")
    print(f"Optimized triangles    : {final_triangles}")
    print(
        f"Triangles removed      : "
        f"{original_triangles - final_triangles}"
    )
    print(
        f"Triangle reduction    : "
        f"{triangle_reduction * 100.0:.2f}%"
    )

    print()

    print(f"Final minimum area     : {final_min_area:.12e}")
    print(f"Final maximum area     : {final_max_area:.12e}")
    print(f"Final mean area        : {final_mean_area:.12e}")
    print(f"Final near-zero count  : {final_near_zero}")

    # ----------------------------------------------------------------------
    # 13. GEOMETRY PRESERVATION CHECK
    # ----------------------------------------------------------------------

    section(13, "Checking geometry preservation")

    original_min = mesh.get_min_bound()
    original_max = mesh.get_max_bound()

    original_extent = original_max - original_min
    optimized_extent = optimized_max - optimized_min

    extent_error = np.abs(
        optimized_extent - original_extent
    ) / np.maximum(
        np.abs(original_extent),
        FINITE_EPS
    )

    max_extent_error = float(np.max(extent_error))

    print(
        f"Maximum bounding-box extent deviation : "
        f"{max_extent_error * 100.0:.3f}%"
    )

    if max_extent_error < 0.10:
        ok("Global mesh extent is well preserved.")
    else:
        warning(
            "Bounding-box extent changed significantly. "
            "Review the optimized mesh before production use."
        )

    # ----------------------------------------------------------------------
    # 14. SAVE PLY
    # ----------------------------------------------------------------------

    section(14, "Saving optimized PLY mesh")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ply_success = o3d.io.write_triangle_mesh(
        str(OUTPUT_PLY),
        optimized,
        write_ascii=False,
        compressed=False,
        write_vertex_normals=True,
        write_vertex_colors=True,
        write_triangle_uvs=True
    )

    if not ply_success:
        raise RuntimeError(
            "Failed to save optimized PLY mesh."
        )

    ok(f"Optimized PLY saved:\n     {OUTPUT_PLY}")

    # ----------------------------------------------------------------------
    # 15. SAVE OBJ
    # ----------------------------------------------------------------------

    section(15, "Saving optimized OBJ mesh")

    obj_success = o3d.io.write_triangle_mesh(
        str(OUTPUT_OBJ),
        optimized,
        write_ascii=False,
        compressed=False,
        write_vertex_normals=True,
        write_vertex_colors=True,
        write_triangle_uvs=True
    )

    if not obj_success:
        warning("OBJ export reported failure.")

    else:
        ok(f"Optimized OBJ saved:\n     {OUTPUT_OBJ}")

    # ----------------------------------------------------------------------
    # 16. SAVE STATISTICS
    # ----------------------------------------------------------------------

    section(16, "Saving optimization statistics")

    total_time = time.perf_counter() - start_time

    stats = []

    stats.append("ARIA-S3D | PHASE 5.7")
    stats.append("ADAPTIVE MESH OPTIMIZATION")
    stats.append("=" * 70)
    stats.append("")
    stats.append("INPUT")
    stats.append("-" * 70)
    stats.append(f"Input mesh: {INPUT_MESH}")
    stats.append(f"Original vertices: {original_vertices}")
    stats.append(f"Original triangles: {original_triangles}")
    stats.append("")
    stats.append("ADAPTIVE OPTIMIZATION")
    stats.append("-" * 70)
    stats.append(
        f"Adaptive requested reduction: "
        f"{reduction * 100.0:.4f}%"
    )
    stats.append(
        f"Adaptive target triangles: "
        f"{target_triangles}"
    )
    stats.append(
        f"Optimization time: "
        f"{optimization_time:.6f}"
    )
    stats.append("")
    stats.append("OUTPUT")
    stats.append("-" * 70)
    stats.append(f"Optimized vertices: {final_vertices}")
    stats.append(f"Optimized triangles: {final_triangles}")
    stats.append(
        f"Vertices removed: "
        f"{original_vertices - final_vertices}"
    )
    stats.append(
        f"Triangles removed: "
        f"{original_triangles - final_triangles}"
    )
    stats.append(
        f"Vertex reduction: "
        f"{vertex_reduction * 100.0:.4f}%"
    )
    stats.append(
        f"Triangle reduction: "
        f"{triangle_reduction * 100.0:.4f}%"
    )
    stats.append(
        f"Vertex retention: "
        f"{vertex_retention * 100.0:.4f}%"
    )
    stats.append(
        f"Triangle retention: "
        f"{triangle_retention * 100.0:.4f}%"
    )
    stats.append("")
    stats.append("QUALITY")
    stats.append("-" * 70)
    stats.append(
        f"Repeated-vertex triangles: {repeated_after}"
    )
    stats.append(
        f"Near-zero triangles: {final_near_zero}"
    )
    stats.append(
        f"Maximum extent deviation: "
        f"{max_extent_error * 100.0:.6f}%"
    )
    stats.append(
        f"Final mesh scale: {optimized_scale:.6f}"
    )
    stats.append("")
    stats.append("OUTPUT FILES")
    stats.append("-" * 70)
    stats.append(str(OUTPUT_PLY))
    stats.append(str(OUTPUT_OBJ))
    stats.append(str(OUTPUT_STATS))
    stats.append("")
    stats.append(
        f"Total processing time: {total_time:.6f} seconds"
    )
    stats.append("")
    stats.append(
        "IMPORTANT: ARIA-S3D remains a monocular reconstruction."
    )
    stats.append(
        "Absolute metric scale remains arbitrary."
    )

    OUTPUT_STATS.write_text(
        "\n".join(stats),
        encoding="utf-8"
    )

    ok(f"Statistics saved:\n     {OUTPUT_STATS}")

    # ----------------------------------------------------------------------
    # FINAL
    # ----------------------------------------------------------------------

    header("ARIA-S3D | PHASE 5.7 COMPLETE")

    print(f"Original vertices      : {original_vertices}")
    print(f"Optimized vertices     : {final_vertices}")
    print(
        f"Vertex reduction      : "
        f"{vertex_reduction * 100.0:.2f}%"
    )

    print()

    print(f"Original triangles      : {original_triangles}")
    print(f"Optimized triangles     : {final_triangles}")
    print(
        f"Triangle reduction     : "
        f"{triangle_reduction * 100.0:.2f}%"
    )

    print()

    print(
        f"Adaptive target        : "
        f"{target_triangles}"
    )

    print(
        f"Bounding-box deviation : "
        f"{max_extent_error * 100.0:.3f}%"
    )

    print(
        f"Processing time        : "
        f"{total_time:.3f} seconds"
    )

    print()
    print("Generated outputs:")
    print(f"  {OUTPUT_PLY}")
    print(f"  {OUTPUT_OBJ}")
    print(f"  {OUTPUT_STATS}")

    print()
    print("NEXT STEP:")
    print("Phase 5.8 - Texture preparation")

    print()
    print("IMPORTANT:")
    print("ARIA-S3D remains a monocular reconstruction.")
    print("Absolute metric scale remains arbitrary.")
    print(
        "Adaptive optimization reduces mesh complexity while "
        "attempting to preserve global reconstructed geometry."
    )

    print("=" * 70)


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        print()
        print("[STOPPED] Optimization interrupted by user.")

    except Exception as exc:
        print()
        print("=" * 70)
        print("[ERROR] PHASE 5.7 FAILED")
        print("=" * 70)
        print(str(exc))
        raise