"""
ARIA-S3D
PHASE 5.8
TEXTURE PREPARATION

Purpose:
    Prepare the optimized drone-scene mesh for texture mapping.

Pipeline:
    1. Load optimized mesh
    2. Validate geometry
    3. Compute normals
    4. Attempt Open3D UV atlas generation
    5. If atlas generation is unsuitable for an outdoor/open mesh,
       generate a robust geometry-based UV projection
    6. Validate UV coordinates
    7. Save texture-ready mesh
    8. Save UV data
    9. Save statistics

Important:
    ARIA-S3D remains monocular.
    Absolute metric scale remains arbitrary.
"""

from pathlib import Path
import time
import sys

import numpy as np
import open3d as o3d


# ======================================================================
# CONFIGURATION
# ======================================================================

ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = ROOT / "data" / "output"

INPUT_MESH = OUTPUT_DIR / "optimized_mesh.ply"

OUTPUT_PLY = OUTPUT_DIR / "texture_ready_mesh.ply"
OUTPUT_OBJ = OUTPUT_DIR / "texture_ready_mesh.obj"
OUTPUT_UVS = OUTPUT_DIR / "texture_uvs.npy"
OUTPUT_STATS = OUTPUT_DIR / "texture_preparation_stats.txt"

# UV atlas configuration
UV_SIZE = 2048
UV_GUTTER = 4.0
UV_MAX_STRETCH = 0.1666667

# Fallback projection
UV_PADDING = 0.005

# Open3D UV atlas can require a manifold mesh.
# We allow it to be attempted, but never make it mandatory.
TRY_UV_ATLAS = True


# ======================================================================
# DISPLAY HELPERS
# ======================================================================

def banner(title):
    print("=" * 70)
    print(f"ARIA-S3D | PHASE 5.8")
    print(title)
    print("=" * 70)
    print()


def section(number, title):
    print(f"[{number}] {title}")
    print("-" * 70)


def ok(message):
    print(f"[OK] {message}")


def warning(message):
    print(f"[WARNING] {message}")


def info(message):
    print(f"[INFO] {message}")


def fail(message):
    print(f"[ERROR] {message}")


# ======================================================================
# VALIDATION
# ======================================================================

def validate_mesh(mesh):
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("Invalid vertex array shape.")

    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("Invalid triangle array shape.")

    if not np.isfinite(vertices).all():
        raise ValueError("Mesh contains non-finite vertex coordinates.")

    if len(triangles) > 0:
        if triangles.min() < 0:
            raise ValueError("Mesh contains negative triangle indices.")

        if triangles.max() >= len(vertices):
            raise ValueError("Mesh contains out-of-range triangle indices.")

    repeated = np.sum(
        (triangles[:, 0] == triangles[:, 1])
        | (triangles[:, 1] == triangles[:, 2])
        | (triangles[:, 0] == triangles[:, 2])
    ) if len(triangles) else 0

    return int(repeated)


# ======================================================================
# GEOMETRY STATISTICS
# ======================================================================

def geometry_stats(vertices):
    minimum = vertices.min(axis=0)
    maximum = vertices.max(axis=0)

    extent = maximum - minimum
    scale = float(np.linalg.norm(extent))

    return minimum, maximum, extent, scale


# ======================================================================
# FALLBACK UV PROJECTION
# ======================================================================

def generate_geometry_uvs(mesh):
    """
    Generate triangle UVs using a robust dominant-axis projection.

    Each triangle receives UV coordinates independently.

    This avoids assumptions about:
        - watertightness
        - manifold topology
        - closed surfaces
        - connected components

    It is therefore suitable as a fallback for outdoor drone scenes.
    """

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    triangle_count = len(triangles)

    if triangle_count == 0:
        raise ValueError("Mesh contains no triangles.")

    uv = np.zeros((triangle_count * 3, 2), dtype=np.float64)

    tri_vertices = vertices[triangles]

    # Triangle normals
    edge1 = tri_vertices[:, 1] - tri_vertices[:, 0]
    edge2 = tri_vertices[:, 2] - tri_vertices[:, 0]

    normals = np.cross(edge1, edge2)

    normal_lengths = np.linalg.norm(normals, axis=1)

    # Avoid division by zero
    valid = normal_lengths > 1e-12

    normals[valid] /= normal_lengths[valid, None]

    abs_normals = np.abs(normals)

    # Dominant projection axis:
    #
    # X dominant -> project onto YZ
    # Y dominant -> project onto XZ
    # Z dominant -> project onto XY
    axes = np.argmax(abs_normals, axis=1)

    for i in range(triangle_count):

        tri = tri_vertices[i]

        axis = axes[i]

        if axis == 0:
            # Project Y/Z
            local = tri[:, [1, 2]]

        elif axis == 1:
            # Project X/Z
            local = tri[:, [0, 2]]

        else:
            # Project X/Y
            local = tri[:, [0, 1]]

        uv[i * 3:(i + 1) * 3] = local

    # ------------------------------------------------------------------
    # Global normalization
    # ------------------------------------------------------------------

    finite = np.isfinite(uv).all(axis=1)

    if not finite.all():
        uv[~finite] = 0.0

    uv_min = uv.min(axis=0)
    uv_max = uv.max(axis=0)

    uv_extent = uv_max - uv_min

    # Protect against flat projections
    if uv_extent[0] < 1e-12:
        uv_extent[0] = 1.0

    if uv_extent[1] < 1e-12:
        uv_extent[1] = 1.0

    uv = (uv - uv_min) / uv_extent

    # Padding
    uv = (
        UV_PADDING
        + uv * (1.0 - 2.0 * UV_PADDING)
    )

    # Clamp numerical noise
    uv = np.clip(uv, 0.0, 1.0)

    return uv


# ======================================================================
# OPEN3D UV ATLAS
# ======================================================================

def try_compute_uv_atlas(mesh):
    """
    Attempt Open3D's tensor UV atlas.

    Returns:
        uv_array, atlas_info

    or:
        None, reason
    """

    try:
        tensor_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)

        result = tensor_mesh.compute_uvatlas(
            size=UV_SIZE,
            gutter=UV_GUTTER,
            max_stretch=UV_MAX_STRETCH,
            parallel_partitions=1,
            nthreads=0
        )

        # texture_uvs is a triangle attribute.
        if "texture_uvs" not in tensor_mesh.triangle:
            return None, "UV atlas did not produce texture_uvs."

        uv_tensor = tensor_mesh.triangle["texture_uvs"]

        uv = uv_tensor.numpy()

        uv = np.asarray(uv, dtype=np.float64)

        if uv.ndim != 2 or uv.shape[1] != 2:
            return None, f"Unexpected UV shape: {uv.shape}"

        expected = len(mesh.triangles) * 3

        if len(uv) != expected:
            return None, (
                f"UV count mismatch. "
                f"Expected {expected}, got {len(uv)}."
            )

        if not np.isfinite(uv).all():
            return None, "UV atlas contains non-finite values."

        # Open3D documents atlas coordinates in [0, 1].
        # Clamp tiny floating point deviations.
        uv = np.clip(uv, 0.0, 1.0)

        atlas_info = {
            "max_stretch": float(result[0]),
            "charts": int(result[1]),
            "partitions": int(result[2]),
        }

        return uv, atlas_info

    except Exception as exc:
        return None, str(exc)


# ======================================================================
# UV VALIDATION
# ======================================================================

def validate_uvs(uvs, triangle_count):

    expected = triangle_count * 3

    if uvs.shape != (expected, 2):
        raise ValueError(
            f"Invalid UV shape {uvs.shape}; "
            f"expected {(expected, 2)}."
        )

    if not np.isfinite(uvs).all():
        raise ValueError("UV coordinates contain non-finite values.")

    uv_min = uvs.min(axis=0)
    uv_max = uvs.max(axis=0)

    outside = np.sum(
        (uvs[:, 0] < 0.0)
        | (uvs[:, 0] > 1.0)
        | (uvs[:, 1] < 0.0)
        | (uvs[:, 1] > 1.0)
    )

    return uv_min, uv_max, int(outside)


# ======================================================================
# ATTACH UVS TO LEGACY OPEN3D MESH
# ======================================================================

def attach_legacy_uvs(mesh, uvs):
    """
    Legacy Open3D stores UVs as:
        3 UV coordinates per triangle.

    Shape:
        (3 * triangle_count, 2)
    """

    mesh.triangle_uvs = o3d.utility.Vector2dVector(uvs)

    return mesh


# ======================================================================
# WRITE OBJ WITH EXPLICIT UVS
# ======================================================================

def write_obj_with_uv(mesh, uvs, path):
    """
    Explicit OBJ writer.

    This guarantees that the UV coordinates are written as:
        v
        vt
        vn
        f v/vt/vn

    rather than relying on legacy Open3D's OBJ behavior.
    """

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    vertex_normals = np.asarray(mesh.vertex_normals)

    if len(vertex_normals) != len(vertices):
        mesh.compute_vertex_normals()
        vertex_normals = np.asarray(mesh.vertex_normals)

    with open(path, "w", encoding="utf-8") as f:

        f.write("# ARIA-S3D Phase 5.8 Texture Ready Mesh\n")
        f.write("# Generated by texture_preparation.py\n\n")

        # --------------------------------------------------------------
        # Vertices
        # --------------------------------------------------------------

        for v in vertices:
            f.write(
                f"v {v[0]:.9f} {v[1]:.9f} {v[2]:.9f}\n"
            )

        f.write("\n")

        # --------------------------------------------------------------
        # UVs
        # --------------------------------------------------------------

        for uv in uvs:
            f.write(
                f"vt {uv[0]:.9f} {uv[1]:.9f}\n"
            )

        f.write("\n")

        # --------------------------------------------------------------
        # Normals
        # --------------------------------------------------------------

        for n in vertex_normals:
            f.write(
                f"vn {n[0]:.9f} {n[1]:.9f} {n[2]:.9f}\n"
            )

        f.write("\n")

        # --------------------------------------------------------------
        # Faces
        # --------------------------------------------------------------

        for i, tri in enumerate(triangles):

            uv_base = i * 3

            v0 = int(tri[0]) + 1
            v1 = int(tri[1]) + 1
            v2 = int(tri[2]) + 1

            uv0 = uv_base + 1
            uv1 = uv_base + 2
            uv2 = uv_base + 3

            # Vertex-normal indices correspond to vertex indices.
            n0 = v0
            n1 = v1
            n2 = v2

            f.write(
                f"f "
                f"{v0}/{uv0}/{n0} "
                f"{v1}/{uv1}/{n1} "
                f"{v2}/{uv2}/{n2}\n"
            )


# ======================================================================
# SAVE STATISTICS
# ======================================================================

def save_statistics(stats):

    with open(OUTPUT_STATS, "w", encoding="utf-8") as f:

        f.write("ARIA-S3D | PHASE 5.8\n")
        f.write("TEXTURE PREPARATION\n")
        f.write("=" * 70 + "\n\n")

        for key, value in stats.items():
            f.write(f"{key}: {value}\n")


# ======================================================================
# MAIN
# ======================================================================

def main():

    start_time = time.time()

    banner("TEXTURE PREPARATION")

    # ==================================================================
    # 1. INPUT
    # ==================================================================

    section(1, "Checking optimized mesh")

    if not INPUT_MESH.exists():
        fail(f"Input mesh not found:\n     {INPUT_MESH}")
        sys.exit(1)

    ok(f"Input mesh found:\n     {INPUT_MESH}")

    # ==================================================================
    # 2. LOAD
    # ==================================================================

    section(2, "Loading optimized mesh")

    mesh = o3d.io.read_triangle_mesh(str(INPUT_MESH))

    if mesh.is_empty():
        fail("Loaded mesh is empty.")
        sys.exit(1)

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    print(f"Vertices  : {len(vertices)}")
    print(f"Triangles : {len(triangles)}")
    print()

    # ==================================================================
    # 3. VALIDATE
    # ==================================================================

    section(3, "Validating input geometry")

    repeated = validate_mesh(mesh)

    if repeated == 0:
        ok("All vertex coordinates are finite.")
        ok("All triangle indices are valid.")
        ok("No repeated-vertex triangles.")
    else:
        warning(
            f"Repeated-vertex triangles detected: {repeated}"
        )

    # ==================================================================
    # 4. GEOMETRY
    # ==================================================================

    section(4, "Computing mesh geometry")

    minimum, maximum, extent, scale = geometry_stats(vertices)

    print(
        f"X range : {minimum[0]:.6f} to {maximum[0]:.6f}"
    )

    print(
        f"Y range : {minimum[1]:.6f} to {maximum[1]:.6f}"
    )

    print(
        f"Z range : {minimum[2]:.6f} to {maximum[2]:.6f}"
    )

    print(f"Mesh scale : {scale:.6f}")
    print()

    # ==================================================================
    # 5. NORMALS
    # ==================================================================

    section(5, "Preparing mesh normals")

    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()

    normals = np.asarray(mesh.vertex_normals)

    if not np.isfinite(normals).all():
        fail("Mesh normals contain non-finite values.")
        sys.exit(1)

    ok(f"Vertex normals computed : {len(normals)}")

    # ==================================================================
    # 6. UV ATLAS
    # ==================================================================

    section(6, "Generating UV coordinates")

    uv_method = "geometry_projection"
    atlas_info = None

    uvs = None

    if TRY_UV_ATLAS:

        print("Attempting Open3D UV atlas...")
        print(
            f"Texture size : {UV_SIZE} x {UV_SIZE}"
        )
        print(
            f"Gutter       : {UV_GUTTER} px"
        )
        print(
            f"Max stretch  : {UV_MAX_STRETCH}"
        )
        print()

        uvs, atlas_result = try_compute_uv_atlas(mesh)

        if uvs is not None:

            uv_method = "open3d_uv_atlas"
            atlas_info = atlas_result

            ok("Open3D UV atlas generated.")

            print(
                f"Atlas max stretch : "
                f"{atlas_info['max_stretch']:.6f}"
            )

            print(
                f"UV charts         : "
                f"{atlas_info['charts']}"
            )

            print(
                f"Partitions        : "
                f"{atlas_info['partitions']}"
            )

        else:

            warning("Automatic UV atlas unavailable.")
            warning(f"Reason: {atlas_result}")
            print()
            print(
                "Falling back to geometry-based UV projection..."
            )

    # ==================================================================
    # 7. FALLBACK UV
    # ==================================================================

    if uvs is None:

        uvs = generate_geometry_uvs(mesh)

        uv_method = "geometry_projection"

        ok("Fallback geometry-based UV projection generated.")

    # ==================================================================
    # 8. UV VALIDATION
    # ==================================================================

    section(8, "Validating UV coordinates")

    uv_min, uv_max, outside = validate_uvs(
        uvs,
        len(triangles)
    )

    print(
        f"UV coordinates       : {len(uvs)}"
    )

    print(
        f"U range              : "
        f"{uv_min[0]:.6f} to {uv_max[0]:.6f}"
    )

    print(
        f"V range              : "
        f"{uv_min[1]:.6f} to {uv_max[1]:.6f}"
    )

    print(
        f"Out-of-range UVs     : {outside}"
    )

    if outside != 0:
        fail("UV coordinates outside [0,1].")
        sys.exit(1)

    ok("All UV coordinates are finite.")
    ok("All UV coordinates are inside [0,1].")

    # ==================================================================
    # 9. ATTACH UVS
    # ==================================================================

    section(9, "Attaching UV coordinates")

    mesh = attach_legacy_uvs(mesh, uvs)

    if not mesh.has_triangle_uvs():
        fail("Open3D mesh does not contain triangle UVs.")
        sys.exit(1)

    loaded_uvs = np.asarray(mesh.triangle_uvs)

    if loaded_uvs.shape != uvs.shape:
        fail("Attached UV data has unexpected shape.")
        sys.exit(1)

    ok("Triangle UV coordinates attached.")

    # ==================================================================
    # 10. SAVE UV DATA
    # ==================================================================

    section(10, "Saving UV data")

    np.save(
        OUTPUT_UVS,
        uvs
    )

    if not OUTPUT_UVS.exists():
        fail("UV data file was not created.")
        sys.exit(1)

    ok(f"UV data saved:\n     {OUTPUT_UVS}")

    # ==================================================================
    # 11. SAVE PLY
    # ==================================================================

    section(11, "Saving texture-ready PLY")

    # Legacy PLY does not reliably preserve UV coordinates.
    # We still save it as the geometry/normals companion output.

    ply_success = o3d.io.write_triangle_mesh(
        str(OUTPUT_PLY),
        mesh,
        write_ascii=False,
        compressed=False,
        write_vertex_normals=True,
        write_vertex_colors=True,
        write_triangle_uvs=False
    )

    if not ply_success:
        fail("Failed to save texture-ready PLY.")
        sys.exit(1)

    ok(f"Texture-ready PLY saved:\n     {OUTPUT_PLY}")

    # ==================================================================
    # 12. SAVE OBJ WITH UVS
    # ==================================================================

    section(12, "Saving UV-enabled OBJ")

    write_obj_with_uv(
        mesh,
        uvs,
        OUTPUT_OBJ
    )

    if not OUTPUT_OBJ.exists():
        fail("Failed to save UV-enabled OBJ.")
        sys.exit(1)

    ok(f"UV-enabled OBJ saved:\n     {OUTPUT_OBJ}")

    # ==================================================================
    # 13. RELOAD OBJ VALIDATION
    # ==================================================================

    section(13, "Validating exported OBJ")

    # We verify that the OBJ exists and is non-empty.
    obj_size = OUTPUT_OBJ.stat().st_size

    if obj_size <= 0:
        fail("Exported OBJ is empty.")
        sys.exit(1)

    print(
        f"OBJ file size : {obj_size / (1024 * 1024):.2f} MB"
    )

    ok("OBJ export completed.")

    # ==================================================================
    # 14. FINAL STATISTICS
    # ==================================================================

    section(14, "Texture preparation statistics")

    elapsed = time.time() - start_time

    stats = {
        "Input mesh": str(INPUT_MESH),
        "Vertices": len(vertices),
        "Triangles": len(triangles),
        "Mesh scale": f"{scale:.6f}",
        "UV method": uv_method,
        "UV coordinates": len(uvs),
        "UV range U": f"{uv_min[0]:.6f} to {uv_max[0]:.6f}",
        "UV range V": f"{uv_min[1]:.6f} to {uv_max[1]:.6f}",
        "Out-of-range UVs": outside,
        "Texture resolution": f"{UV_SIZE} x {UV_SIZE}",
        "UV gutter": UV_GUTTER,
        "UV max stretch": UV_MAX_STRETCH,
        "Processing time": f"{elapsed:.3f} seconds",
    }

    if atlas_info is not None:

        stats["Atlas max stretch"] = (
            f"{atlas_info['max_stretch']:.6f}"
        )

        stats["UV charts"] = atlas_info["charts"]
        stats["UV partitions"] = atlas_info["partitions"]

    for key, value in stats.items():
        print(f"{key:<28}: {value}")

    # ==================================================================
    # 15. SAVE STATS
    # ==================================================================

    section(15, "Saving preparation statistics")

    save_statistics(stats)

    ok(
        f"Statistics saved:\n     {OUTPUT_STATS}"
    )

    # ==================================================================
    # COMPLETE
    # ==================================================================

    print()
    print("=" * 70)
    print("ARIA-S3D | PHASE 5.8 COMPLETE")
    print("=" * 70)

    print()
    print(f"Input vertices        : {len(vertices)}")
    print(f"Input triangles       : {len(triangles)}")
    print(f"UV method             : {uv_method}")
    print(f"UV coordinates        : {len(uvs)}")
    print(f"Out-of-range UVs      : {outside}")
    print(f"Processing time       : {elapsed:.3f} seconds")

    print()
    print("Generated outputs:")

    print(f"  {OUTPUT_PLY}")
    print(f"  {OUTPUT_OBJ}")
    print(f"  {OUTPUT_UVS}")
    print(f"  {OUTPUT_STATS}")

    print()
    print("-" * 70)
    print("NEXT STEP:")
    print("Phase 5.9 - Texture projection / generation")
    print()
    print("IMPORTANT:")
    print("ARIA-S3D remains a monocular reconstruction.")
    print("Absolute metric scale remains arbitrary.")
    print()
    print(
        "The OBJ contains explicit triangle UV coordinates and "
        "is the primary texture-ready mesh."
    )
    print("=" * 70)


# ======================================================================
# ENTRY POINT
# ======================================================================

if __name__ == "__main__":
    main()