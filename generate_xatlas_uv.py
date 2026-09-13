"""
ARIA-S3D
PHASE 5.9.2 – XATLAS UV ATLAS GENERATION

Purpose
-------
Generate a proper UV atlas for the optimized ARIA-S3D mesh using XAtlas.

Why this exists
---------------
The previous fallback UV projection produced extremely poor atlas utilization.
This script replaces that fallback with a real chart-based XAtlas unwrap.

Inputs
------
data/output/optimized_mesh.ply

Outputs
-------
data/output/xatlas_uv_mesh.ply
data/output/xatlas_uv_mesh.obj
data/output/xatlas_uvs.npy
data/output/xatlas_uv_indices.npy
data/output/xatlas_vertex_mapping.npy
data/output/xatlas_uv_stats.txt

Important
---------
The XAtlas binding differs between Python package versions.
This script therefore sets optional ChartOptions / PackOptions attributes
only when the installed version supports them.

The OBJ output contains the real UV mapping.
The PLY output contains seam-expanded geometry, but PLY should not be relied
upon to preserve texture UV/material information.
"""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

import numpy as np
import open3d as o3d

try:
    import xatlas
except ImportError:
    print("[FATAL] xatlas is not installed.")
    print()
    print("Install it inside your ARIA-S3D virtual environment:")
    print("    pip install xatlas")
    sys.exit(1)


# ============================================================================
# CONFIGURATION
# ============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "data" / "output"

INPUT_MESH = OUTPUT_DIR / "optimized_mesh.ply"

OUTPUT_PLY = OUTPUT_DIR / "xatlas_uv_mesh.ply"
OUTPUT_OBJ = OUTPUT_DIR / "xatlas_uv_mesh.obj"

OUTPUT_UVS = OUTPUT_DIR / "xatlas_uvs.npy"
OUTPUT_UV_INDICES = OUTPUT_DIR / "xatlas_uv_indices.npy"
OUTPUT_VERTEX_MAPPING = OUTPUT_DIR / "xatlas_vertex_mapping.npy"

OUTPUT_STATS = OUTPUT_DIR / "xatlas_uv_stats.txt"

TARGET_TEXTURE_RESOLUTION = 2048
PADDING_PIXELS = 4

EPS = 1e-12


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def heading(title: str) -> None:
    print()
    print(title)


def safe_set(obj, attribute: str, value) -> bool:
    """
    Set a Python binding option only if the installed xatlas version exposes it.

    This avoids crashes such as:
        AttributeError:
        'xatlas.PackOptions' object has no attribute 'block_align'
    """

    if hasattr(obj, attribute):
        try:
            setattr(obj, attribute, value)
            print(f"  {attribute:<28}: {value}")
            return True
        except Exception as exc:
            print(
                f"  {attribute:<28}: supported but could not set "
                f"({type(exc).__name__}: {exc})"
            )
            return False

    print(f"  {attribute:<28}: unsupported - skipped")
    return False


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def finite_ratio(arr: np.ndarray) -> float:
    if arr.size == 0:
        return 0.0

    return float(np.isfinite(arr).sum()) / float(arr.size)


def triangle_areas(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    p0 = vertices[triangles[:, 0]]
    p1 = vertices[triangles[:, 1]]
    p2 = vertices[triangles[:, 2]]

    cross = np.cross(p1 - p0, p2 - p0)

    return 0.5 * np.linalg.norm(cross, axis=1)


def uv_triangle_areas(
    uv_vertices: np.ndarray,
    uv_triangles: np.ndarray,
) -> np.ndarray:
    a = uv_vertices[uv_triangles[:, 0]]
    b = uv_vertices[uv_triangles[:, 1]]
    c = uv_vertices[uv_triangles[:, 2]]

    ab = b - a
    ac = c - a

    signed_double_area = (
        ab[:, 0] * ac[:, 1]
        - ab[:, 1] * ac[:, 0]
    )

    return 0.5 * np.abs(signed_double_area)


def get_atlas_utilization(atlas) -> tuple[float | None, str]:
    """
    Try to obtain XAtlas' own utilization metric.

    Different bindings expose this differently:
    - float
    - list/tuple
    - NumPy array
    """

    if not hasattr(atlas, "utilization"):
        return None, "not exposed by installed xatlas"

    try:
        util = atlas.utilization

        if callable(util):
            util = util()

        arr = np.asarray(util, dtype=np.float64).reshape(-1)

        if arr.size == 0:
            return None, "empty utilization result"

        value = float(np.mean(arr))

        # Some bindings could theoretically expose percent rather than fraction.
        if value > 1.0 and value <= 100.0:
            value /= 100.0

        return value, "xatlas"
    except Exception as exc:
        return None, f"could not read utilization: {exc}"


def estimate_uv_area_utilization(
    uv_vertices: np.ndarray,
    uv_triangles: np.ndarray,
) -> float:
    """
    Sum UV triangle area as a secondary atlas-utilization metric.

    XAtlas charts should normally not overlap, so this is useful as a
    sanity/diagnostic statistic.

    It is not considered as authoritative as atlas.utilization because
    padding and rasterization behavior are not represented here.
    """

    areas = uv_triangle_areas(uv_vertices, uv_triangles)

    areas = areas[np.isfinite(areas)]

    if len(areas) == 0:
        return 0.0

    return float(np.sum(areas))


def write_obj_with_uvs(
    path: Path,
    vertices: np.ndarray,
    triangles: np.ndarray,
    vertex_uvs: np.ndarray,
) -> None:
    """
    Write an OBJ with one vt for each seam-expanded XAtlas vertex.

    Because XAtlas has already duplicated vertices at UV seams,
    the vertex index and texture-coordinate index can safely match.
    """

    require(
        len(vertices) == len(vertex_uvs),
        "OBJ export requires one UV coordinate per seam-expanded vertex.",
    )

    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("# ARIA-S3D XAtlas UV mesh\n")
        f.write("# Generated by generate_xatlas_uv.py\n")
        f.write("# Phase 5.9.2\n")
        f.write("\n")

        for vertex in vertices:
            f.write(
                f"v "
                f"{float(vertex[0]):.9f} "
                f"{float(vertex[1]):.9f} "
                f"{float(vertex[2]):.9f}\n"
            )

        f.write("\n")

        # OBJ UV convention normally uses lower-left origin.
        # We preserve XAtlas coordinates directly.
        for uv in vertex_uvs:
            f.write(
                f"vt "
                f"{float(uv[0]):.9f} "
                f"{float(uv[1]):.9f}\n"
            )

        f.write("\n")

        for tri in triangles:
            i0 = int(tri[0]) + 1
            i1 = int(tri[1]) + 1
            i2 = int(tri[2]) + 1

            f.write(
                f"f "
                f"{i0}/{i0} "
                f"{i1}/{i1} "
                f"{i2}/{i2}\n"
            )


def parse_obj_counts(path: Path) -> tuple[int, int, int]:
    vertices = 0
    texture_vertices = 0
    faces = 0

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                vertices += 1
            elif line.startswith("vt "):
                texture_vertices += 1
            elif line.startswith("f "):
                faces += 1

    return vertices, texture_vertices, faces


def try_get_chart_count(atlas) -> int | None:
    """
    XAtlas Python versions differ in metadata exposure.
    Attempt several non-destructive possibilities.
    """

    candidates = (
        "chart_count",
        "chartCount",
    )

    for name in candidates:
        if hasattr(atlas, name):
            try:
                value = getattr(atlas, name)

                if callable(value):
                    value = value()

                return int(value)
            except Exception:
                pass

    return None


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:

    total_start = time.perf_counter()

    print("=" * 72)
    print("ARIA-S3D | PHASE 5.9.2 – XATLAS UV ATLAS GENERATION")
    print("=" * 72)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------------
    # 1. Check input
    # ----------------------------------------------------------------------

    heading("[1] Checking input mesh")

    print(f"Input mesh : {INPUT_MESH}")

    require(
        INPUT_MESH.exists(),
        f"Input mesh not found:\n{INPUT_MESH}",
    )

    # ----------------------------------------------------------------------
    # 2. Load mesh
    # ----------------------------------------------------------------------

    heading("[2] Loading optimized mesh")

    legacy_mesh = o3d.io.read_triangle_mesh(str(INPUT_MESH))

    require(
        legacy_mesh is not None,
        "Open3D failed to create the input mesh object.",
    )

    require(
        len(legacy_mesh.vertices) > 0,
        "Input mesh contains no vertices.",
    )

    require(
        len(legacy_mesh.triangles) > 0,
        "Input mesh contains no triangles.",
    )

    vertices64 = np.asarray(
        legacy_mesh.vertices,
        dtype=np.float64,
    )

    triangles64 = np.asarray(
        legacy_mesh.triangles,
        dtype=np.int64,
    )

    original_vertex_count = len(vertices64)
    original_triangle_count = len(triangles64)

    print(f"Vertices   : {original_vertex_count:,}")
    print(f"Triangles  : {original_triangle_count:,}")

    # ----------------------------------------------------------------------
    # 3. Validate geometry
    # ----------------------------------------------------------------------

    heading("[3] Validating mesh geometry")

    require(
        vertices64.ndim == 2 and vertices64.shape[1] == 3,
        f"Unexpected vertex array shape: {vertices64.shape}",
    )

    require(
        triangles64.ndim == 2 and triangles64.shape[1] == 3,
        f"Unexpected triangle array shape: {triangles64.shape}",
    )

    require(
        np.all(np.isfinite(vertices64)),
        "Input mesh contains non-finite vertex coordinates.",
    )

    require(
        np.all(triangles64 >= 0),
        "Input mesh contains negative triangle indices.",
    )

    require(
        int(triangles64.max()) < original_vertex_count,
        "Triangle index exceeds available vertex count.",
    )

    geom_areas = triangle_areas(vertices64, triangles64)

    zero_geom_triangles = int(np.count_nonzero(geom_areas <= EPS))

    if zero_geom_triangles:
        print(
            f"[WARN] Near-zero geometry triangles : "
            f"{zero_geom_triangles:,}"
        )
    else:
        print("[OK] No zero-area geometry triangles detected.")

    # Critical for xatlas binding:
    vertices = np.ascontiguousarray(
        vertices64.astype(np.float32)
    )

    triangles = np.ascontiguousarray(
        triangles64.astype(np.uint32)
    )

    print("[OK] Geometry is valid.")
    print(f"Position dtype : {vertices.dtype}")
    print(f"Index dtype    : {triangles.dtype}")

    bbox_min = vertices64.min(axis=0)
    bbox_max = vertices64.max(axis=0)
    bbox_extent = bbox_max - bbox_min

    print(f"BBox min       : {bbox_min}")
    print(f"BBox max       : {bbox_max}")
    print(f"BBox extent    : {bbox_extent}")

    # ----------------------------------------------------------------------
    # 4. Create XAtlas atlas
    # ----------------------------------------------------------------------

    heading("[4] Creating XAtlas atlas")

    atlas = xatlas.Atlas()

    atlas.add_mesh(vertices, triangles)

    print("[OK] Mesh added to XAtlas.")

    # ----------------------------------------------------------------------
    # 5. Chart options
    # ----------------------------------------------------------------------

    heading("[5] Configuring XAtlas chart options")

    chart_options = xatlas.ChartOptions()

    # Keep settings conservative.
    #
    # Different xatlas-python builds expose different properties.
    # safe_set() makes this portable.

    safe_set(
        chart_options,
        "max_iterations",
        1,
    )

    print("[OK] Compatible chart options configured.")

    # ----------------------------------------------------------------------
    # 6. Packing options
    # ----------------------------------------------------------------------

    heading("[6] Configuring XAtlas packing")

    pack_options = xatlas.PackOptions()

    # IMPORTANT:
    # Never directly use:
    #
    #     pack_options.block_align = False
    #
    # because the user's installed xatlas package does not expose that
    # property.

    safe_set(
        pack_options,
        "resolution",
        TARGET_TEXTURE_RESOLUTION,
    )

    safe_set(
        pack_options,
        "padding",
        PADDING_PIXELS,
    )

    safe_set(
        pack_options,
        "brute_force",
        False,
    )

    safe_set(
        pack_options,
        "block_align",
        False,
    )

    safe_set(
        pack_options,
        "create_image",
        False,
    )

    safe_set(
        pack_options,
        "rotate_charts",
        True,
    )

    safe_set(
        pack_options,
        "rotate_charts_to_axis",
        True,
    )

    safe_set(
        pack_options,
        "max_chart_size",
        0,
    )

    print("[OK] Compatible packing options configured.")

    # ----------------------------------------------------------------------
    # 7. Generate UV atlas
    # ----------------------------------------------------------------------

    heading("[7] Generating XAtlas UV atlas")

    print("This is the computationally expensive step.")
    print(
        f"Processing {original_triangle_count:,} triangles..."
    )

    atlas_start = time.perf_counter()

    atlas.generate(
        chart_options=chart_options,
        pack_options=pack_options,
    )

    atlas_seconds = time.perf_counter() - atlas_start

    print(
        f"[OK] XAtlas generation completed in "
        f"{atlas_seconds:.3f}s."
    )

    # ----------------------------------------------------------------------
    # 8. Retrieve XAtlas results
    # ----------------------------------------------------------------------

    heading("[8] Retrieving XAtlas results")

    vmapping, atlas_indices, atlas_uvs = atlas[0]

    vmapping = np.asarray(vmapping)
    atlas_indices = np.asarray(atlas_indices)
    atlas_uvs = np.asarray(atlas_uvs)

    print(f"Vertex mapping shape : {vmapping.shape}")
    print(f"Atlas indices shape  : {atlas_indices.shape}")
    print(f"Atlas UV shape       : {atlas_uvs.shape}")

    # Normalize arrays.

    vmapping = np.ascontiguousarray(
        vmapping.astype(np.uint32).reshape(-1)
    )

    if atlas_indices.ndim == 1:
        require(
            atlas_indices.size % 3 == 0,
            "XAtlas returned an index array not divisible by 3.",
        )

        atlas_indices = atlas_indices.reshape(-1, 3)

    atlas_indices = np.ascontiguousarray(
        atlas_indices.astype(np.uint32)
    )

    atlas_uvs = np.ascontiguousarray(
        atlas_uvs.astype(np.float32)
    )

    require(
        atlas_uvs.ndim == 2 and atlas_uvs.shape[1] == 2,
        f"Unexpected XAtlas UV shape: {atlas_uvs.shape}",
    )

    require(
        len(vmapping) == len(atlas_uvs),
        (
            "XAtlas vertex mapping and UV arrays have different "
            "vertex counts."
        ),
    )

    require(
        len(atlas_indices) == original_triangle_count,
        (
            "XAtlas changed triangle count unexpectedly: "
            f"{len(atlas_indices):,} vs "
            f"{original_triangle_count:,}"
        ),
    )

    require(
        int(atlas_indices.max()) < len(atlas_uvs),
        "XAtlas generated an invalid UV/vertex index.",
    )

    require(
        int(vmapping.max()) < original_vertex_count,
        "XAtlas vertex mapping references invalid source vertices.",
    )

    xatlas_vertex_count = len(atlas_uvs)

    print(f"Source vertices      : {original_vertex_count:,}")
    print(f"XAtlas vertices      : {xatlas_vertex_count:,}")
    print(f"Source triangles     : {original_triangle_count:,}")
    print(f"XAtlas triangles     : {len(atlas_indices):,}")

    duplicated_vertices = (
        xatlas_vertex_count - original_vertex_count
    )

    duplication_ratio = (
        float(xatlas_vertex_count)
        / float(original_vertex_count)
    )

    print(f"Seam extra vertices  : {duplicated_vertices:,}")
    print(f"Vertex expansion     : {duplication_ratio:.3f}x")

    # ----------------------------------------------------------------------
    # 9. Validate UV data
    # ----------------------------------------------------------------------

    heading("[9] Validating UV atlas")

    require(
        np.all(np.isfinite(atlas_uvs)),
        "XAtlas UV array contains NaN or infinity.",
    )

    uv_min = atlas_uvs.min(axis=0)
    uv_max = atlas_uvs.max(axis=0)

    print(f"UV minimum           : {uv_min}")
    print(f"UV maximum           : {uv_max}")
    print(
        f"UV scalar range      : "
        f"{float(atlas_uvs.min()):.8f} -> "
        f"{float(atlas_uvs.max()):.8f}"
    )

    uv_outside_mask = (
        (atlas_uvs[:, 0] < -1e-6)
        | (atlas_uvs[:, 0] > 1.0 + 1e-6)
        | (atlas_uvs[:, 1] < -1e-6)
        | (atlas_uvs[:, 1] > 1.0 + 1e-6)
    )

    uv_outside = int(np.count_nonzero(uv_outside_mask))

    print(f"UVs outside [0,1]    : {uv_outside:,}")

    require(
        uv_outside == 0,
        "XAtlas produced UV coordinates outside the expected [0,1] range.",
    )

    uv_areas = uv_triangle_areas(
        atlas_uvs,
        atlas_indices,
    )

    zero_uv_triangles = int(
        np.count_nonzero(uv_areas <= EPS)
    )

    print(
        f"Zero-area UV faces   : "
        f"{zero_uv_triangles:,}"
    )

    if zero_uv_triangles > 0:
        print(
            "[WARN] Some triangles are nearly degenerate "
            "in UV space."
        )
    else:
        print("[OK] All triangles have usable UV area.")

    # ----------------------------------------------------------------------
    # 10. UV atlas utilization
    # ----------------------------------------------------------------------

    heading("[10] Measuring UV atlas utilization")

    atlas_utilization, util_source = get_atlas_utilization(atlas)

    area_utilization = estimate_uv_area_utilization(
        atlas_uvs,
        atlas_indices,
    )

    if atlas_utilization is not None:
        print(
            f"XAtlas utilization   : "
            f"{atlas_utilization * 100.0:.4f}%"
        )
        print(f"Utilization source   : {util_source}")
    else:
        print(
            "XAtlas utilization   : unavailable from "
            "installed binding"
        )
        print(f"Reason               : {util_source}")

    print(
        f"UV triangle area sum : "
        f"{area_utilization * 100.0:.4f}%"
    )

    chart_count = try_get_chart_count(atlas)

    if chart_count is not None:
        print(f"Chart count          : {chart_count:,}")
    else:
        print(
            "Chart count          : not exposed by "
            "installed binding"
        )

    # ----------------------------------------------------------------------
    # 11. Build seam-expanded mesh
    # ----------------------------------------------------------------------

    heading("[11] Building seam-expanded XAtlas mesh")

    expanded_vertices = vertices64[
        vmapping.astype(np.int64)
    ]

    expanded_triangles = atlas_indices.astype(np.int64)

    require(
        len(expanded_vertices) == len(atlas_uvs),
        "Expanded geometry and UV vertex counts do not match.",
    )

    expanded_mesh = o3d.geometry.TriangleMesh()

    expanded_mesh.vertices = o3d.utility.Vector3dVector(
        expanded_vertices
    )

    expanded_mesh.triangles = o3d.utility.Vector3iVector(
        expanded_triangles
    )

    expanded_mesh.compute_vertex_normals()

    print(f"Expanded vertices    : {len(expanded_vertices):,}")
    print(f"Expanded triangles   : {len(expanded_triangles):,}")

    print("[OK] Seam-expanded mesh constructed.")

    # ----------------------------------------------------------------------
    # 12. Save NumPy UV data
    # ----------------------------------------------------------------------

    heading("[12] Saving UV mapping arrays")

    # Per-XAtlas-vertex UV coordinates.
    np.save(
        OUTPUT_UVS,
        atlas_uvs.astype(np.float32),
    )

    np.save(
        OUTPUT_UV_INDICES,
        atlas_indices.astype(np.uint32),
    )

    np.save(
        OUTPUT_VERTEX_MAPPING,
        vmapping.astype(np.uint32),
    )

    print(f"[OK] {OUTPUT_UVS.name}")
    print(f"[OK] {OUTPUT_UV_INDICES.name}")
    print(f"[OK] {OUTPUT_VERTEX_MAPPING.name}")

    # ----------------------------------------------------------------------
    # 13. Save seam-expanded PLY
    # ----------------------------------------------------------------------

    heading("[13] Saving seam-expanded PLY mesh")

    ply_ok = o3d.io.write_triangle_mesh(
        str(OUTPUT_PLY),
        expanded_mesh,
        write_ascii=False,
        compressed=False,
        write_vertex_normals=True,
    )

    require(
        ply_ok,
        "Open3D failed to write the XAtlas PLY mesh.",
    )

    print(f"[OK] {OUTPUT_PLY.name}")
    print(
        "NOTE: PLY preserves geometry/normals here, "
        "not the texture UV material."
    )

    # ----------------------------------------------------------------------
    # 14. Save OBJ with proper UV coordinates
    # ----------------------------------------------------------------------

    heading("[14] Saving UV-mapped OBJ")

    write_obj_with_uvs(
        OUTPUT_OBJ,
        expanded_vertices,
        expanded_triangles,
        atlas_uvs,
    )

    require(
        OUTPUT_OBJ.exists()
        and OUTPUT_OBJ.stat().st_size > 0,
        "OBJ export failed.",
    )

    print(f"[OK] {OUTPUT_OBJ.name}")

    # ----------------------------------------------------------------------
    # 15. OBJ round-trip validation
    # ----------------------------------------------------------------------

    heading("[15] Validating OBJ round-trip")

    obj_v, obj_vt, obj_f = parse_obj_counts(
        OUTPUT_OBJ
    )

    print(f"OBJ vertices         : {obj_v:,}")
    print(f"OBJ texture vertices : {obj_vt:,}")
    print(f"OBJ faces            : {obj_f:,}")

    require(
        obj_v == xatlas_vertex_count,
        (
            "OBJ vertex count differs from generated "
            "XAtlas mesh."
        ),
    )

    require(
        obj_vt == xatlas_vertex_count,
        (
            "OBJ texture-coordinate count differs "
            "from XAtlas UV count."
        ),
    )

    require(
        obj_f == original_triangle_count,
        (
            "OBJ triangle count differs from source "
            "triangle count."
        ),
    )

    print("[OK] OBJ structure matches XAtlas output.")

    # Open3D secondary readability check.

    print()
    print("Open3D OBJ import check...")

    obj_mesh = o3d.io.read_triangle_mesh(
        str(OUTPUT_OBJ)
    )

    require(
        obj_mesh is not None
        and len(obj_mesh.vertices) > 0
        and len(obj_mesh.triangles) > 0,
        "Open3D could not read the generated OBJ.",
    )

    imported_vertices = len(obj_mesh.vertices)
    imported_triangles = len(obj_mesh.triangles)

    print(f"Imported vertices    : {imported_vertices:,}")
    print(f"Imported triangles   : {imported_triangles:,}")

    require(
        imported_triangles == original_triangle_count,
        (
            "Open3D OBJ round-trip did not preserve "
            "triangle count."
        ),
    )

    print("[OK] Open3D OBJ round-trip succeeded.")

    # ----------------------------------------------------------------------
    # 16. Produce per-corner UV data
    # ----------------------------------------------------------------------

    heading("[16] Preparing texture-projection UV layout")

    # Our previous ARIA texture pipeline used one UV for every triangle corner.
    #
    # Shape must therefore become:
    #
    #     (triangle_count * 3, 2)
    #
    # = 129470 * 3
    # = 388410 UV coordinates for the current mesh.

    flat_uv_indices = atlas_indices.reshape(-1)

    corner_uvs = atlas_uvs[
        flat_uv_indices.astype(np.int64)
    ]

    expected_corner_count = (
        original_triangle_count * 3
    )

    require(
        corner_uvs.shape == (
            expected_corner_count,
            2,
        ),
        (
            f"Unexpected corner UV shape: "
            f"{corner_uvs.shape}"
        ),
    )

    CORNER_UV_OUTPUT = OUTPUT_DIR / "xatlas_corner_uvs.npy"

    np.save(
        CORNER_UV_OUTPUT,
        corner_uvs.astype(np.float32),
    )

    print(
        f"Per-corner UV shape  : "
        f"{corner_uvs.shape}"
    )

    print(
        f"Expected coordinates : "
        f"{expected_corner_count:,}"
    )

    print(f"[OK] {CORNER_UV_OUTPUT.name}")

    # ----------------------------------------------------------------------
    # 17. Final validation
    # ----------------------------------------------------------------------

    heading("[17] Final validation")

    require(
        OUTPUT_PLY.exists(),
        "Missing output PLY.",
    )

    require(
        OUTPUT_OBJ.exists(),
        "Missing output OBJ.",
    )

    require(
        OUTPUT_UVS.exists(),
        "Missing UV coordinate array.",
    )

    require(
        OUTPUT_UV_INDICES.exists(),
        "Missing UV index array.",
    )

    require(
        OUTPUT_VERTEX_MAPPING.exists(),
        "Missing vertex mapping array.",
    )

    require(
        CORNER_UV_OUTPUT.exists(),
        "Missing per-corner UV array.",
    )

    require(
        finite_ratio(atlas_uvs) == 1.0,
        "UV array contains non-finite values.",
    )

    require(
        len(atlas_indices) == original_triangle_count,
        "Triangle count changed.",
    )

    print("[OK] All required XAtlas artifacts exist.")
    print("[OK] All UV coordinates are finite.")
    print("[OK] Triangle count preserved.")
    print("[OK] XAtlas seam mapping preserved.")
    print("[OK] Per-corner texture UV layout generated.")

    # ----------------------------------------------------------------------
    # 18. Statistics report
    # ----------------------------------------------------------------------

    heading("[18] Writing XAtlas statistics")

    total_seconds = time.perf_counter() - total_start

    stats_lines = [
        "ARIA-S3D | PHASE 5.9.2 – XATLAS UV ATLAS GENERATION",
        "=" * 72,
        "",
        f"Input mesh: {INPUT_MESH}",
        "",
        "SOURCE GEOMETRY",
        f"Vertices: {original_vertex_count}",
        f"Triangles: {original_triangle_count}",
        f"Zero/near-zero geometry triangles: {zero_geom_triangles}",
        "",
        "XATLAS OUTPUT",
        f"Vertices after UV seam expansion: {xatlas_vertex_count}",
        f"Triangles: {len(atlas_indices)}",
        f"Extra seam vertices: {duplicated_vertices}",
        f"Vertex expansion ratio: {duplication_ratio:.6f}",
        "",
        "UV INFORMATION",
        f"UV shape: {atlas_uvs.shape}",
        f"Corner UV shape: {corner_uvs.shape}",
        f"UV minimum: {uv_min.tolist()}",
        f"UV maximum: {uv_max.tolist()}",
        f"UVs outside [0,1]: {uv_outside}",
        f"Zero-area UV triangles: {zero_uv_triangles}",
        "",
        "UTILIZATION",
        (
            f"XAtlas utilization: "
            f"{atlas_utilization:.8f}"
            if atlas_utilization is not None
            else "XAtlas utilization: unavailable"
        ),
        (
            f"XAtlas utilization percent: "
            f"{atlas_utilization * 100.0:.4f}%"
            if atlas_utilization is not None
            else "XAtlas utilization percent: unavailable"
        ),
        f"UV triangle area sum: {area_utilization:.8f}",
        f"UV triangle area percent: {area_utilization * 100.0:.4f}%",
        (
            f"Chart count: {chart_count}"
            if chart_count is not None
            else "Chart count: unavailable"
        ),
        "",
        "CONFIGURATION",
        f"Requested texture resolution: {TARGET_TEXTURE_RESOLUTION}",
        f"Requested padding: {PADDING_PIXELS}",
        "",
        "TIMING",
        f"XAtlas generation time: {atlas_seconds:.3f} sec",
        f"Total processing time: {total_seconds:.3f} sec",
        "",
        "OUTPUTS",
        str(OUTPUT_PLY),
        str(OUTPUT_OBJ),
        str(OUTPUT_UVS),
        str(OUTPUT_UV_INDICES),
        str(OUTPUT_VERTEX_MAPPING),
        str(CORNER_UV_OUTPUT),
        "",
        "RESULT",
        "PHASE 5.9.2 PASS",
        "",
    ]

    OUTPUT_STATS.write_text(
        "\n".join(stats_lines),
        encoding="utf-8",
    )

    print(f"[OK] {OUTPUT_STATS.name}")

    # ----------------------------------------------------------------------
    # Final summary
    # ----------------------------------------------------------------------

    print()
    print("=" * 72)
    print("ARIA-S3D | PHASE 5.9.2 PASS")
    print("=" * 72)

    print(
        f"Source vertices       : "
        f"{original_vertex_count:,}"
    )

    print(
        f"XAtlas vertices       : "
        f"{xatlas_vertex_count:,}"
    )

    print(
        f"Triangles             : "
        f"{original_triangle_count:,}"
    )

    print(
        f"Per-corner UVs        : "
        f"{len(corner_uvs):,}"
    )

    print(
        f"UV range              : "
        f"{float(atlas_uvs.min()):.6f} -> "
        f"{float(atlas_uvs.max()):.6f}"
    )

    if atlas_utilization is not None:
        print(
            f"XAtlas utilization    : "
            f"{atlas_utilization * 100.0:.4f}%"
        )
    else:
        print(
            f"UV area utilization   : "
            f"{area_utilization * 100.0:.4f}%"
        )

    print(
        f"XAtlas generation     : "
        f"{atlas_seconds:.3f}s"
    )

    print(
        f"Total processing      : "
        f"{total_seconds:.3f}s"
    )

    print()
    print("Generated:")
    print(f"  {OUTPUT_PLY.name}")
    print(f"  {OUTPUT_OBJ.name}")
    print(f"  {OUTPUT_UVS.name}")
    print(f"  {OUTPUT_UV_INDICES.name}")
    print(f"  {OUTPUT_VERTEX_MAPPING.name}")
    print(f"  {CORNER_UV_OUTPUT.name}")
    print(f"  {OUTPUT_STATS.name}")

    print()
    print("NEXT STEP:")
    print(
        "  Use the XAtlas UV atlas for calibrated "
        "photographic texture projection."
    )
    print(
        "  Do NOT use the previous fallback geometry UVs."
    )


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":

    try:
        main()

    except KeyboardInterrupt:
        print()
        print("[ABORTED] User interrupted XAtlas generation.")
        sys.exit(130)

    except Exception as exc:
        print()
        print("=" * 72)
        print("ARIA-S3D | PHASE 5.9.2 FAILED")
        print("=" * 72)
        print()
        print(f"Reason: {exc}")
        print()
        print("Full traceback:")
        traceback.print_exc()
        sys.exit(1)