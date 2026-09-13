"""
ARIA-S3D | PHASE 5.9.1
Projection Geometry Diagnostic

Tests:
1. BA world->camera extrinsics
2. Inverted camera->world extrinsics
3. Mesh visibility inside the source video frame
4. Camera-to-mesh distances
5. UV atlas occupancy

No files are modified.
"""

from pathlib import Path
import re
import numpy as np
import cv2
import open3d as o3d
import time


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

INPUT_DIR = ROOT / "data" / "input"
OUTPUT_DIR = ROOT / "data" / "output"

LEGACY_MESH_PATH = OUTPUT_DIR / "texture_ready_mesh.ply"
LEGACY_UV_PATH = OUTPUT_DIR / "texture_uvs.npy"
XATLAS_MESH_PATH = OUTPUT_DIR / "xatlas_uv_mesh.ply"
XATLAS_UV_PATH = OUTPUT_DIR / "xatlas_corner_uvs.npy"
XATLAS_STATS_PATH = OUTPUT_DIR / "xatlas_uv_stats.txt"

if XATLAS_MESH_PATH.exists() and XATLAS_UV_PATH.exists():
    MESH_PATH = XATLAS_MESH_PATH
    UV_PATH = XATLAS_UV_PATH
else:
    MESH_PATH = LEGACY_MESH_PATH
    UV_PATH = LEGACY_UV_PATH
BA_PATH = OUTPUT_DIR / "bundle_adjusted_cameras.npz"
VIDEO_PATH_FILE = INPUT_DIR / "source_video_path.txt"

WIDTH = 1280
HEIGHT = 720

MIN_DEPTH = 0.01


# ============================================================
# HELPERS
# ============================================================

def rotation_matrix_from_rvec(rvec):
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))
    return R


def make_world_to_camera(camera_params):
    rvec = camera_params[:3]
    t = camera_params[3:6]

    R = rotation_matrix_from_rvec(rvec)

    Rt = np.eye(4, dtype=np.float64)
    Rt[:3, :3] = R
    Rt[:3, 3] = t

    return Rt


def invert_rt(Rt):
    R = Rt[:3, :3]
    t = Rt[:3, 3]

    inv = np.eye(4, dtype=np.float64)
    inv[:3, :3] = R.T
    inv[:3, 3] = -R.T @ t

    return inv


def project_points(points, K, Rt):
    """
    World -> camera -> image.
    """

    R = Rt[:3, :3]
    t = Rt[:3, 3]

    cam = (R @ points.T).T + t

    z = cam[:, 2]

    valid_depth = z > MIN_DEPTH

    u = np.full(len(points), np.nan)
    v = np.full(len(points), np.nan)

    u[valid_depth] = (
        K[0, 0] * cam[valid_depth, 0] / z[valid_depth]
        + K[0, 2]
    )

    v[valid_depth] = (
        K[1, 1] * cam[valid_depth, 1] / z[valid_depth]
        + K[1, 2]
    )

    inside = (
        valid_depth
        & (u >= 0)
        & (u < WIDTH)
        & (v >= 0)
        & (v < HEIGHT)
    )

    return cam, u, v, valid_depth, inside


def camera_center(Rt):
    R = Rt[:3, :3]
    t = Rt[:3, 3]
    return -R.T @ t


def percentile(values, p):
    values = np.asarray(values)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan

    return float(np.percentile(values, p))


# ============================================================
# LOAD
# ============================================================

start = time.time()

print()
print("=" * 72)
print("ARIA-S3D | PHASE 5.9.1 - PROJECTION GEOMETRY DIAGNOSTIC")
print("=" * 72)

print()
print("[1] Loading mesh")

mesh = o3d.io.read_triangle_mesh(str(MESH_PATH))

if mesh.is_empty():
    raise RuntimeError("Mesh is empty.")

vertices = np.asarray(mesh.vertices, dtype=np.float64)

print(f"Mesh vertices       : {len(vertices):,}")
print(f"Mesh triangles      : {len(mesh.triangles):,}")

bbox = mesh.get_axis_aligned_bounding_box()

print(f"Bounding box min    : {bbox.min_bound}")
print(f"Bounding box max    : {bbox.max_bound}")
print(f"Bounding extent     : {bbox.get_extent()}")


print()
print("[2] Loading UVs")

uvs = np.load(UV_PATH)

print(f"UV array shape      : {uvs.shape}")

uv_flat = uvs.reshape(-1, 2)

finite_uv = np.isfinite(uv_flat).all(axis=1)

print(f"Finite UVs          : {finite_uv.sum():,}/{len(uv_flat):,}")

if finite_uv.any():
    valid_uv = uv_flat[finite_uv]

    print(
        f"UV range            : "
        f"{valid_uv.min(axis=0)} -> {valid_uv.max(axis=0)}"
    )


print()
print("[3] Loading Bundle Adjustment cameras")

ba = np.load(BA_PATH)

camera_params = ba["camera_params"]
frame_ids = ba["frame_ids"]
K = ba["K"]

print(f"BA cameras          : {len(camera_params)}")
print(f"Camera parameter shape: {camera_params.shape}")
print(f"Frame ID range      : {frame_ids.min()} -> {frame_ids.max()}")

print()
print("Intrinsic matrix:")
print(K)

if K.shape != (3, 3):
    raise RuntimeError("Invalid intrinsic matrix.")

print()
print("[4] Loading source video")

video_path = Path(VIDEO_PATH_FILE.read_text().strip())

print(f"Video               : {video_path}")

cap = cv2.VideoCapture(str(video_path))

if not cap.isOpened():
    raise RuntimeError("Could not open source video.")

video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
video_fps = cap.get(cv2.CAP_PROP_FPS)

cap.release()

print(f"Video resolution    : {video_width} x {video_height}")
print(f"Video frames        : {video_frames}")
print(f"Video FPS           : {video_fps:.3f}")


# ============================================================
# SAMPLE MESH
# ============================================================

print()
print("[5] Sampling mesh")

MAX_POINTS = 20000

if len(vertices) > MAX_POINTS:
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(
        len(vertices),
        size=MAX_POINTS,
        replace=False
    )
    sample_points = vertices[sample_idx]
else:
    sample_points = vertices

print(f"Diagnostic points   : {len(sample_points):,}")


# ============================================================
# BUILD EXTRINSICS
# ============================================================

world_to_camera = [
    make_world_to_camera(p)
    for p in camera_params
]

camera_to_world = [
    invert_rt(Rt)
    for Rt in world_to_camera
]


# ============================================================
# TEST FUNCTION
# ============================================================

def evaluate(label, extrinsics):

    print()
    print("=" * 72)
    print(label)
    print("=" * 72)

    visible_percentages = []
    depth_percentages = []
    distances = []

    camera_results = []

    for i, Rt in enumerate(extrinsics):

        cam, u, v, valid_depth, inside = project_points(
            sample_points,
            K,
            Rt
        )

        depth_pct = 100.0 * valid_depth.mean()
        inside_pct = 100.0 * inside.mean()

        C = camera_center(Rt)

        distance = np.linalg.norm(
            sample_points - C[None, :],
            axis=1
        )

        median_distance = float(np.median(distance))

        visible_percentages.append(inside_pct)
        depth_percentages.append(depth_pct)
        distances.append(median_distance)

        camera_results.append(
            (
                i,
                int(frame_ids[i]),
                inside_pct,
                depth_pct,
                median_distance
            )
        )

    visible_percentages = np.asarray(visible_percentages)
    depth_percentages = np.asarray(depth_percentages)
    distances = np.asarray(distances)

    print(
        f"Median points in front     : "
        f"{np.median(depth_percentages):.2f}%"
    )

    print(
        f"Median points inside image : "
        f"{np.median(visible_percentages):.2f}%"
    )

    print(
        f"Mean points inside image   : "
        f"{np.mean(visible_percentages):.2f}%"
    )

    print(
        f"Best camera coverage       : "
        f"{np.max(visible_percentages):.2f}%"
    )

    print(
        f"Worst camera coverage      : "
        f"{np.min(visible_percentages):.2f}%"
    )

    print(
        f"Camera median distance     : "
        f"{np.median(distances):.4f}"
    )

    thresholds = [0.1, 1.0, 5.0, 10.0, 25.0]

    for threshold in thresholds:

        count = int(
            np.sum(visible_percentages >= threshold)
        )

        print(
            f"Cameras >= {threshold:>4.1f}% visibility : "
            f"{count}/{len(extrinsics)}"
        )

    print()
    print("Top 10 cameras by visibility:")

    camera_results.sort(
        key=lambda x: x[2],
        reverse=True
    )

    for row in camera_results[:10]:

        i, frame_id, inside_pct, depth_pct, distance = row

        print(
            f"  cam={i:3d} "
            f"frame={frame_id:5d} "
            f"inside={inside_pct:7.2f}% "
            f"front={depth_pct:7.2f}% "
            f"distance={distance:10.4f}"
        )

    return {
        "visible": visible_percentages,
        "depth": depth_percentages,
        "distance": distances,
        "results": camera_results,
    }


# ============================================================
# TEST A
# ============================================================

result_a = evaluate(
    "EXTRINSIC TEST A - BA WORLD -> CAMERA",
    world_to_camera
)


# ============================================================
# TEST B
# ============================================================

result_b = evaluate(
    "EXTRINSIC TEST B - INVERTED CAMERA -> WORLD",
    camera_to_world
)


# ============================================================
# DECISION
# ============================================================

print()
print("=" * 72)
print("EXTRINSIC COMPARISON")
print("=" * 72)

score_a = (
    np.median(result_a["visible"])
    + 0.25 * np.median(result_a["depth"])
)

score_b = (
    np.median(result_b["visible"])
    + 0.25 * np.median(result_b["depth"])
)

print(f"World -> Camera score : {score_a:.4f}")
print(f"Inverted score       : {score_b:.4f}")
print("Convention authority : Bundle Adjustment reprojection model")

# The inverted transform can score higher on the coarse "inside image"
# heuristic while still destroying frame/feature correspondence. The BA
# optimizer defines [R|t] as world-to-camera, so that convention is the only
# production choice for photographic projection.
selected = "BA WORLD -> CAMERA"
selected_result = result_a

print()
print(f"SELECTED CONVENTION  : {selected}")


# ============================================================
# UV OCCUPANCY
# ============================================================

print()
print("=" * 72)
print("UV ATLAS DIAGNOSTIC")
print("=" * 72)

TEXTURE_SIZE = 2048

uv_valid = uv_flat[
    np.isfinite(uv_flat).all(axis=1)
]

uv_valid = uv_valid[
    (uv_valid[:, 0] >= 0)
    & (uv_valid[:, 0] <= 1)
    & (uv_valid[:, 1] >= 0)
    & (uv_valid[:, 1] <= 1)
]

print(f"Valid UV coordinates : {len(uv_valid):,}")

# Prefer XAtlas's packed-area utilization. Counting only UV corner pixels
# underestimates a valid atlas because it ignores triangle interiors.
occupancy = None
if XATLAS_STATS_PATH.exists():
    stats_text = XATLAS_STATS_PATH.read_text(encoding="utf-8", errors="ignore")
    match = re.search(
        r"XAtlas utilization percent:\s*([0-9.]+)%",
        stats_text,
    )
    if match:
        occupancy = float(match.group(1))
        print(f"XAtlas packed utilization : {occupancy:.4f}%")

if occupancy is None:
    if len(uv_valid) > 0:
        atlas = np.zeros(
            (TEXTURE_SIZE, TEXTURE_SIZE),
            dtype=np.uint8
        )

        px = np.clip(
            (uv_valid[:, 0] * (TEXTURE_SIZE - 1)).astype(np.int32),
            0,
            TEXTURE_SIZE - 1
        )

        py = np.clip(
            ((1.0 - uv_valid[:, 1]) * (TEXTURE_SIZE - 1)).astype(np.int32),
            0,
            TEXTURE_SIZE - 1
        )

        atlas[py, px] = 255
        occupancy = 100.0 * np.count_nonzero(atlas) / atlas.size

        print(f"UV point occupancy      : {occupancy:.4f}%")
        unique_uv_pixels = len(
            np.unique(np.stack([px, py], axis=1), axis=0)
        )
        print(f"Unique atlas pixels     : {unique_uv_pixels:,}")
    else:
        occupancy = 0.0
        print("No valid UV coordinates.")


# ============================================================
# FINAL DIAGNOSIS
# ============================================================

print()
print("=" * 72)
print("DIAGNOSIS")
print("=" * 72)

median_visibility = float(
    np.median(selected_result["visible"])
)

best_visibility = float(
    np.max(selected_result["visible"])
)

median_depth = float(
    np.median(selected_result["depth"])
)

if median_depth > 80 and median_visibility > 20:
    print("[PASS] Camera geometry has strong image visibility.")

elif median_depth > 50 and median_visibility > 5:
    print("[WARN] Camera geometry is usable but weak.")

else:
    print("[FAIL] Camera geometry appears inconsistent.")


if occupancy > 20:
    print("[PASS] UV atlas has reasonable coordinate utilization.")

elif occupancy > 5:
    print("[WARN] UV atlas utilization is low.")

else:
    print("[FAIL] UV atlas utilization is extremely low.")


if median_visibility < 5:
    print()
    print("ROOT CAUSE CANDIDATE:")
    print("  Camera/mesh projection geometry.")

elif occupancy < 5:
    print()
    print("ROOT CAUSE CANDIDATE:")
    print("  UV atlas layout.")

else:
    print()
    print("ROOT CAUSE CANDIDATE:")
    print("  Camera <-> source-video frame correspondence.")


# ============================================================
# REPORT
# ============================================================

report_path = OUTPUT_DIR / "projection_geometry_diagnostic.txt"

with open(report_path, "w", encoding="utf-8") as f:

    f.write("ARIA-S3D | PHASE 5.9.1\n")
    f.write("Projection Geometry Diagnostic\n")
    f.write("=" * 60 + "\n\n")

    f.write(f"Mesh vertices: {len(vertices)}\n")
    f.write(f"Mesh triangles: {len(mesh.triangles)}\n")
    f.write(f"BA cameras: {len(camera_params)}\n")
    f.write(f"Video frames: {video_frames}\n")
    f.write(f"Video resolution: {video_width}x{video_height}\n\n")

    f.write(
        f"World->Camera median visibility: "
        f"{np.median(result_a['visible']):.4f}%\n"
    )

    f.write(
        f"Inverted median visibility: "
        f"{np.median(result_b['visible']):.4f}%\n"
    )

    f.write(
        f"World->Camera median front: "
        f"{np.median(result_a['depth']):.4f}%\n"
    )

    f.write(
        f"Inverted median front: "
        f"{np.median(result_b['depth']):.4f}%\n"
    )

    f.write(
        f"\nSelected convention: {selected}\n"
    )

    f.write(
        f"UV occupancy: {occupancy:.4f}%\n"
    )

    f.write(
        f"\nProcessing time: "
        f"{time.time() - start:.3f}s\n"
    )

print()
print(f"[OK] Report written:")
print(f"     {report_path}")

print()
print("=" * 72)
print("PHASE 5.9.1 DIAGNOSTIC COMPLETE")
print("=" * 72)
