"""
ARIA-S3D
PHASE 5.9
UNIVERSAL TEXTURE PROJECTION ENGINE

Purpose
-------
Project calibrated source-video frames onto the reconstructed mesh using
the optimized Bundle Adjustment camera poses. This implementation is
input-driven and does not assume a particular video filename, frame count,
trajectory length, or camera motion pattern.

Primary inputs
--------------
data/input/source_video_path.txt
data/input/video_metadata.npz                  (optional)
data/output/xatlas_uv_mesh.ply
data/output/xatlas_corner_uvs.npy
data/output/bundle_adjusted_cameras.npz        (preferred)
data/output/ba_data.npz                        (fallback)

Camera convention
-----------------
bundle_adjustment.py stores camera parameters as:

    [rx, ry, rz, tx, ty, tz]

with OpenCV-style Rodrigues rotation and:

    X_camera = R @ X_world + t

Open3D's project_images_to_albedo() expects world-to-camera 4x4 extrinsics,
so the BA [R|t] transform is used directly. No trajectory-based orientation
guessing is performed.

Outputs
-------
data/output/aria_albedo_texture.png
data/output/aria_textured_mesh.glb
data/output/aria_textured_mesh.obj
data/output/texture_projection_stats.txt

Notes
-----
- ARIA-S3D remains monocular; absolute metric scale is arbitrary.
- Photographic projection is required for PASS. No synthetic gradient
  fallback is used.
- Open3D 0.19+ is expected.
"""

from pathlib import Path
import sys
import time
import math

import cv2
import numpy as np
import open3d as o3d


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = PROJECT_ROOT / "data" / "input"
QUALITY_FRAME_DIR = PROJECT_ROOT / "data" / "quality_frames"
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"

VIDEO_PATH_FILE = INPUT_DIR / "source_video_path.txt"
VIDEO_METADATA_FILE = INPUT_DIR / "video_metadata.npz"

LEGACY_MESH_FILE = OUTPUT_DIR / "texture_ready_mesh.ply"
LEGACY_UV_FILE = OUTPUT_DIR / "texture_uvs.npy"

# Phase 5.9.2 produces a seam-expanded mesh and one UV per triangle corner.
# Prefer those artifacts so photographic projection uses the real XAtlas
# layout. The legacy pair remains a compatibility fallback for older runs.
XATLAS_MESH_FILE = OUTPUT_DIR / "xatlas_uv_mesh.ply"
XATLAS_UV_FILE = OUTPUT_DIR / "xatlas_corner_uvs.npy"

if XATLAS_MESH_FILE.exists() and XATLAS_UV_FILE.exists():
    MESH_FILE = XATLAS_MESH_FILE
    UV_FILE = XATLAS_UV_FILE
    UV_LAYOUT_SOURCE = "XAtlas Phase 5.9.2"
else:
    MESH_FILE = LEGACY_MESH_FILE
    UV_FILE = LEGACY_UV_FILE
    UV_LAYOUT_SOURCE = "legacy texture-preparation fallback"

BA_CAMERAS_FILE = OUTPUT_DIR / "bundle_adjusted_cameras.npz"
BA_DATA_FILE = OUTPUT_DIR / "ba_data.npz"

OUTPUT_TEXTURE = OUTPUT_DIR / "aria_albedo_texture.png"
OUTPUT_GLTF = OUTPUT_DIR / "aria_textured_mesh.glb"
OUTPUT_OBJ = OUTPUT_DIR / "aria_textured_mesh.obj"
OUTPUT_STATS = OUTPUT_DIR / "texture_projection_stats.txt"

TEXTURE_SIZE = 2048

# Maximum source images supplied to Open3D in one projection call.
MAX_TEXTURE_FRAMES = 250

# Sharpness threshold is deliberately conservative. If too few frames
# survive, the selector falls back to percentile-based selection.
MIN_SHARPNESS = 20.0

# Avoid nearly identical neighbouring frames.
MIN_FRAME_GAP = 2

# A frame must be inside this image margin to be considered useful.
IMAGE_BORDER_MARGIN = 2

# If a BA frame id does not directly match a video frame index, these
# candidate mapping strategies are tested automatically.
MAX_MAPPING_CANDIDATES = 12

# Minimum amount of non-background texture expected from projection.
MIN_TEXTURE_NONZERO_RATIO = 0.002

# Fill small gaps left by discrete back-projection using nearby photographic
# texels. This is standard atlas padding, not a synthetic texture fallback.
TEXTURE_PADDING_PIXELS = 8

# Minimum fraction of selected cameras that must be valid.
MIN_VALID_CAMERA_RATIO = 0.50

# Phase 6.1 visibility-aware atlas baking. The existing Open3D projection is
# still run as a compatibility/reference path, while this baker explicitly
# chooses calibrated, front-facing, depth-consistent views per mesh triangle.
VISIBILITY_BAKE_TOP_VIEWS = 3
VISIBILITY_BAKE_DEPTH_SCALE = 4
VISIBILITY_BAKE_DEPTH_TOLERANCE = 0.12
VISIBILITY_BAKE_MIN_FACING = 0.05


# ============================================================
# BASIC UTILITIES
# ============================================================

def print_header(title):
    print("=" * 78)
    print(title)
    print("=" * 78)


def check_file(path, description):
    if not path.exists():
        print(f"[ERROR] Missing {description}:")
        print(f"        {path}")
        return False

    print(f"[OK] {description}:")
    print(f"     {path}")
    return True


def safe_float(value, default=0.0):
    try:
        x = float(value)
        return x if np.isfinite(x) else default
    except Exception:
        return default


def compute_sharpness(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def load_npz_dict(path):
    try:
        data = np.load(path, allow_pickle=True)
        result = {key: data[key] for key in data.files}
        data.close()
        return result
    except Exception as exc:
        raise RuntimeError(f"Could not load {path}: {exc}") from exc


def as_1d_int(values, name):
    arr = np.asarray(values).reshape(-1)

    if arr.size == 0:
        raise RuntimeError(f"{name} is empty.")

    if not np.all(np.isfinite(arr)):
        raise RuntimeError(f"{name} contains non-finite values.")

    return np.rint(arr).astype(np.int64)


# ============================================================
# VIDEO DISCOVERY / METADATA
# ============================================================

def load_video_path():
    if not VIDEO_PATH_FILE.exists():
        raise RuntimeError(
            f"Source-video registration file not found: {VIDEO_PATH_FILE}"
        )

    text = VIDEO_PATH_FILE.read_text(
        encoding="utf-8"
    ).strip()

    if not text:
        raise RuntimeError(
            "source_video_path.txt is empty."
        )

    path = Path(text)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    if not path.exists():
        raise RuntimeError(
            f"Registered source video does not exist: {path}"
        )

    return path


def load_video_info(video_path):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open source video: {video_path}"
        )

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    fps = safe_float(
        cap.get(cv2.CAP_PROP_FPS),
        0.0,
    )

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    cap.release()

    if width <= 0 or height <= 0 or total_frames <= 0:
        raise RuntimeError(
            "Invalid source-video metadata."
        )

    duration = (
        total_frames / fps
        if fps > 0
        else 0.0
    )

    return {
        "width": width,
        "height": height,
        "fps": fps,
        "total_frames": total_frames,
        "duration": duration,
    }


def load_video_metadata():
    if not VIDEO_METADATA_FILE.exists():
        return {}

    try:
        return load_npz_dict(
            VIDEO_METADATA_FILE
        )

    except Exception as exc:
        print(
            f"[WARNING] Could not load optional "
            f"video metadata: {exc}"
        )

        return {}


# ============================================================
# CAMERA CALIBRATION
# ============================================================

def scalar_from_metadata(metadata, names):
    for name in names:

        if name in metadata:

            value = np.asarray(
                metadata[name]
            ).reshape(-1)

            if value.size:

                x = safe_float(
                    value[0],
                    math.nan,
                )

                if np.isfinite(x):
                    return x

    return None


def build_intrinsics(
    width,
    height,
    metadata,
    ba_data=None,
):
    """
    Prefer calibration stored by BA. Fall back to registered metadata.
    Final fallback is the historical ARIA initialization.
    """

    if (
        ba_data is not None
        and "K" in ba_data
    ):

        K = np.asarray(
            ba_data["K"],
            dtype=np.float64,
        )

        if (
            K.shape == (3, 3)
            and np.all(np.isfinite(K))
        ):

            if (
                K[0, 0] > 0
                and K[1, 1] > 0
            ):

                # If dimensions differ from the calibration image, scale
                # the principal/focal parameters to the actual source size.
                calib_w = None
                calib_h = None

                for key in (
                    "image_width",
                    "width",
                    "IMAGE_WIDTH",
                ):

                    if key in ba_data:

                        calib_w = safe_float(
                            np.asarray(
                                ba_data[key]
                            ).reshape(-1)[0],
                            0,
                        )

                        break

                for key in (
                    "image_height",
                    "height",
                    "IMAGE_HEIGHT",
                ):

                    if key in ba_data:

                        calib_h = safe_float(
                            np.asarray(
                                ba_data[key]
                            ).reshape(-1)[0],
                            0,
                        )

                        break

                if (
                    calib_w
                    and calib_h
                    and calib_w > 0
                    and calib_h > 0
                ):

                    sx = width / calib_w
                    sy = height / calib_h

                    K = K.copy()

                    K[0, :] *= sx
                    K[1, :] *= sy

                return (
                    K,
                    "bundle_adjusted_cameras/ba_data",
                )

    fx = scalar_from_metadata(
        metadata,
        (
            "fx",
            "focal_x",
            "focal_length_x",
        ),
    )

    fy = scalar_from_metadata(
        metadata,
        (
            "fy",
            "focal_y",
            "focal_length_y",
        ),
    )

    cx = scalar_from_metadata(
        metadata,
        (
            "cx",
            "principal_x",
        ),
    )

    cy = scalar_from_metadata(
        metadata,
        (
            "cy",
            "principal_y",
        ),
    )

    fx = (
        fx
        if fx and fx > 0
        else float(width)
    )

    fy = (
        fy
        if fy and fy > 0
        else float(width)
    )

    cx = (
        cx
        if cx is not None
        else width / 2.0
    )

    cy = (
        cy
        if cy is not None
        else height / 2.0
    )

    K = np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    return (
        K,
        "video_metadata/ARIA_initialization",
    )


# ============================================================
# BA CAMERA LOADING
# ============================================================

def load_ba_cameras():
    """
    Preferred source: bundle_adjusted_cameras.npz

    Expected:
        camera_params : N x 6
        frame_ids     : N
        K             : 3 x 3

    Fallback:
        ba_data.npz
    """

    source = None
    data = None

    if BA_CAMERAS_FILE.exists():

        source = BA_CAMERAS_FILE

        data = load_npz_dict(
            BA_CAMERAS_FILE
        )

    elif BA_DATA_FILE.exists():

        source = BA_DATA_FILE

        data = load_npz_dict(
            BA_DATA_FILE
        )

    else:

        raise RuntimeError(
            "Neither bundle_adjusted_cameras.npz "
            "nor ba_data.npz exists."
        )

    if "camera_params" not in data:

        raise RuntimeError(
            f"{source} does not contain camera_params."
        )

    camera_params = np.asarray(
        data["camera_params"],
        dtype=np.float64,
    )

    if (
        camera_params.ndim != 2
        or camera_params.shape[1] < 6
    ):

        raise RuntimeError(
            f"camera_params has invalid shape: "
            f"{camera_params.shape}"
        )

    camera_params = camera_params[:, :6]

    if not np.all(
        np.isfinite(camera_params)
    ):

        raise RuntimeError(
            "BA camera_params contain "
            "non-finite values."
        )

    if "frame_ids" in data:

        frame_ids = as_1d_int(
            data["frame_ids"],
            "BA frame_ids",
        )

    elif "frame_id" in data:

        frame_ids = as_1d_int(
            data["frame_id"],
            "BA frame_id",
        )

    else:

        # Last-resort indexing. This is explicitly reported.
        frame_ids = np.arange(
            len(camera_params),
            dtype=np.int64,
        )

    if len(frame_ids) != len(camera_params):

        raise RuntimeError(
            "BA frame_ids and camera_params "
            "have different lengths."
        )

    if len(np.unique(frame_ids)) != len(frame_ids):

        raise RuntimeError(
            "BA frame_ids contain duplicates."
        )

    K = None

    if "K" in data:

        candidate = np.asarray(
            data["K"],
            dtype=np.float64,
        )

        if (
            candidate.shape == (3, 3)
            and np.all(np.isfinite(candidate))
        ):

            K = candidate

    return {
        "source": source,
        "camera_params": camera_params,
        "frame_ids": frame_ids,
        "K": K,
        "data": data,
    }


def rodrigues_to_matrix(rvec):

    rvec = np.asarray(
        rvec,
        dtype=np.float64,
    ).reshape(3, 1)

    R, _ = cv2.Rodrigues(
        rvec
    )

    if (
        R.shape != (3, 3)
        or not np.all(np.isfinite(R))
    ):

        raise RuntimeError(
            "Invalid Rodrigues rotation."
        )

    return R


def ba_camera_to_extrinsic(
    camera_params,
):
    """
    Convert BA [rx, ry, rz, tx, ty, tz]
    into Open3D's world-to-camera 4x4 extrinsic matrix.
    """

    p = np.asarray(
        camera_params,
        dtype=np.float64,
    ).reshape(-1)

    if (
        p.size < 6
        or not np.all(
            np.isfinite(p[:6])
        )
    ):

        raise RuntimeError(
            "Invalid BA camera parameters."
        )

    R = rodrigues_to_matrix(
        p[:3]
    )

    t = p[3:6]

    ext = np.eye(
        4,
        dtype=np.float64,
    )

    ext[:3, :3] = R
    ext[:3, 3] = t

    return ext


def build_ba_extrinsics(
    camera_params,
):
    extrinsics = []

    for idx, params in enumerate(
        camera_params
    ):

        try:

            ext = ba_camera_to_extrinsic(
                params
            )

        except Exception as exc:

            raise RuntimeError(
                f"Invalid BA camera at index "
                f"{idx}: {exc}"
            ) from exc

        extrinsics.append(ext)

    return extrinsics


# ============================================================
# FRAME-ID → VIDEO-FRAME MAPPING
# ============================================================

def metadata_frame_interval(
    metadata,
):

    for key in (
        "frame_interval",
        "recommended_frame_interval",
        "sample_interval",
        "sampling_interval",
        "frame_step",
    ):

        if key in metadata:

            value = safe_float(
                np.asarray(
                    metadata[key]
                ).reshape(-1)[0],
                0.0,
            )

            if value >= 1:

                return int(
                    round(value)
                )

    return None


def candidate_frame_mappings(
    ba_frame_ids,
    total_video_frames,
    metadata,
):
    """
    Generate multiple plausible mappings without assuming that the BA
    frame numbering is identical to source-video frame numbering.

    Candidates are tuples:
        (name, mapped_video_ids)

    The final selector scores them using video readability and temporal
    consistency. Exact identity is always tested first.
    """

    ids = np.asarray(
        ba_frame_ids,
        dtype=np.int64,
    )

    candidates = []

    def add(name, mapped):

        mapped = np.asarray(
            mapped,
            dtype=np.int64,
        )

        if mapped.shape != ids.shape:
            return

        if (
            np.any(mapped < 0)
            or np.any(
                mapped >= total_video_frames
            )
        ):
            return

        if (
            len(np.unique(mapped))
            != len(mapped)
        ):
            return

        candidates.append(
            (name, mapped)
        )

    add(
        "identity",
        ids,
    )

    interval = metadata_frame_interval(
        metadata
    )

    if interval:

        add(
            f"metadata_interval_{interval}",
            ids * interval,
        )

        add(
            f"metadata_interval_{interval}_plus_one",
            ids * interval + 1,
        )

    # If IDs are a zero-based reconstruction sequence, infer a uniform
    # scale from its maximum ID to the video frame range.
    max_id = (
        int(ids.max())
        if len(ids)
        else 0
    )

    if max_id > 0:

        ratio = (
            total_video_frames - 1
        ) / max_id

        for rounded in {
            max(
                1,
                int(round(ratio)),
            ),
            max(
                1,
                int(math.floor(ratio)),
            ),
            max(
                1,
                int(math.ceil(ratio)),
            ),
        }:

            add(
                f"inferred_stride_{rounded}",
                ids * rounded,
            )

    # A linear endpoint mapping handles cases where the BA IDs are not
    # zero-based but still span the sampled source sequence.
    id_min = int(ids.min())
    id_max = int(ids.max())

    if id_max > id_min:

        mapped = np.rint(
            (
                (ids - id_min)
                / (id_max - id_min)
                * (total_video_frames - 1)
            )
        ).astype(np.int64)

        add(
            "linear_endpoint_mapping",
            mapped,
        )

    # Common reconstruction conventions:
    # frame id may be one-based.
    add(
        "identity_minus_one",
        ids - 1,
    )

    add(
        "identity_plus_one",
        ids + 1,
    )

    # Remove duplicate mappings by content.
    unique = []
    seen = set()

    for name, mapped in candidates:

        key = tuple(
            mapped.tolist()
        )

        if key not in seen:

            unique.append(
                (name, mapped)
            )

            seen.add(key)

    return unique[
        :MAX_MAPPING_CANDIDATES
    ]


def frame_correspondence(frame, reference):
    """Return normalized grayscale correlation for two corresponding frames."""

    if frame is None or reference is None:
        return None

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)

    size = (160, 90)
    frame_gray = cv2.resize(frame_gray, size, interpolation=cv2.INTER_AREA)
    reference_gray = cv2.resize(reference_gray, size, interpolation=cv2.INTER_AREA)

    a = frame_gray.astype(np.float32)
    b = reference_gray.astype(np.float32)
    a -= float(a.mean())
    b -= float(b.mean())

    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= 1e-8:
        return 0.0

    return float(np.sum(a * b) / denominator)


def score_frame_mapping(
    video_path,
    ba_frame_ids,
    mapped_ids,
    max_probe=24,
):
    """
    Score a candidate mapping by actually decoding a small set of frames.
    This avoids blindly assuming that BA frame IDs equal video indices.
    """

    if len(mapped_ids) == 0:
        return -1.0, 0.0, 0, -1.0

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():
        return -1.0, 0.0, 0, -1.0

    probe_indices = np.linspace(
        0,
        len(mapped_ids) - 1,
        min(
            max_probe,
            len(mapped_ids),
        ),
        dtype=np.int64,
    )

    sharpness = []
    correspondence = []
    readable = 0

    for idx in probe_indices:

        frame_id = int(
            mapped_ids[idx]
        )

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_id,
        )

        ok, frame = cap.read()

        if (
            ok
            and frame is not None
            and frame.size
        ):

            readable += 1

            sharpness.append(
                compute_sharpness(frame)
            )

            quality_path = QUALITY_FRAME_DIR / (
                f"frame_{int(ba_frame_ids[idx]):05d}.jpg"
            )
            if quality_path.exists():
                reference = cv2.imread(
                    str(quality_path),
                    cv2.IMREAD_COLOR,
                )
                score = frame_correspondence(
                    frame,
                    reference,
                )
                if score is not None:
                    correspondence.append(score)

    cap.release()

    if not sharpness:
        return -1.0, 0.0, readable, -1.0

    mean_sharpness = float(
        np.mean(sharpness)
    )

    readable_ratio = (
        readable
        / max(
            1,
            len(probe_indices),
        )
    )

    mean_correspondence = (
        float(np.mean(correspondence))
        if correspondence
        else -1.0
    )

    # Readability remains mandatory. When quality-frame references exist,
    # visual correspondence is the primary discriminator; sharpness is only
    # a tie-breaker between otherwise plausible mappings.
    score = (
        readable_ratio * 1000.0
        + (
            max(mean_correspondence, -1.0) * 1000.0
            if mean_correspondence >= 0.0
            else 0.0
        )
        + min(mean_sharpness, 5000.0) / 100.0
    )

    return (
        score,
        mean_sharpness,
        readable,
        mean_correspondence,
    )


def select_best_frame_mapping(
    video_path,
    ba_frame_ids,
    total_video_frames,
    metadata,
):

    candidates = candidate_frame_mappings(
        ba_frame_ids,
        total_video_frames,
        metadata,
    )

    if not candidates:

        raise RuntimeError(
            "Could not generate any "
            "BA→video frame mapping."
        )

    results = []

    print()
    print(
        "Testing BA frame -> "
        "source-video mappings:"
    )

    for name, mapped in candidates:

        (
            score,
            mean_sharpness,
            readable,
            mean_correspondence,
        ) = score_frame_mapping(
            video_path,
            ba_frame_ids,
            mapped,
        )

        results.append(
            {
                "name": name,
                "mapped": mapped,
                "score": score,
                "mean_sharpness": mean_sharpness,
                "readable": readable,
                "mean_correspondence": mean_correspondence,
            }
        )

        print(
            f"  {name:<32} "
            f"readable={readable:>2} "
            f"probe={min(24, len(mapped)):>2} "
            f"corr={mean_correspondence:>7.4f} "
            f"sharpness={mean_sharpness:>9.2f} "
            f"score={score:>10.2f}"
        )

    results.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    best = results[0]

    if best["readable"] < max(
        1,
        int(
            math.ceil(
                0.75
                * min(
                    24,
                    len(best["mapped"]),
                )
            )
        ),
    ):

        raise RuntimeError(
            "No BA→video mapping decoded "
            "reliably enough."
        )

    print()
    print(
        f"[OK] Selected mapping: "
        f"{best['name']}"
    )

    print(
        f"     BA frames   : "
        f"{len(ba_frame_ids)}"
    )

    print(
        f"     Video frames: "
        f"{len(best['mapped'])}"
    )

    return (
        best["mapped"],
        best["name"],
        results,
    )


# ============================================================
# FRAME SELECTION
# ============================================================

def uniform_subsample(
    items,
    max_count,
):

    if len(items) <= max_count:
        return list(items)

    indices = np.linspace(
        0,
        len(items) - 1,
        max_count,
        dtype=np.int64,
    )

    return [
        items[int(i)]
        for i in indices
    ]


def read_video_frame(
    cap,
    frame_id,
):

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        int(frame_id),
    )

    ok, frame = cap.read()

    if (
        not ok
        or frame is None
    ):
        return None

    return frame


def select_texture_observations(
    video_path,
    video_frame_ids,
    width,
    height,
):
    """
    Select sharp, temporally distributed source frames.

    The BA cameras and selected video frames remain one-to-one.
    """

    if len(video_frame_ids) == 0:
        return []

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():
        raise RuntimeError(
            "Could not reopen source video."
        )

    observations = []

    # Decode a candidate subset first.
    # If BA has many cameras, this is still bounded and preserves
    # temporal coverage.
    candidate_ids = uniform_subsample(
        list(video_frame_ids),
        max_count=min(
            max(
                MAX_TEXTURE_FRAMES * 2,
                300,
            ),
            len(video_frame_ids),
        ),
    )

    last_selected = -10**18

    for video_id in candidate_ids:

        video_id = int(video_id)

        if (
            video_id - last_selected
            < MIN_FRAME_GAP
        ):
            continue

        frame = read_video_frame(
            cap,
            video_id,
        )

        if frame is None:
            continue

        if (
            frame.shape[1] != width
            or frame.shape[0] != height
        ):

            frame = cv2.resize(
                frame,
                (width, height),
                interpolation=cv2.INTER_AREA,
            )

        sharpness = compute_sharpness(
            frame
        )

        observations.append(
            {
                "video_frame_id": video_id,
                "frame": frame,
                "sharpness": sharpness,
            }
        )

        last_selected = video_id

    cap.release()

    if not observations:
        return []

    # Robust threshold:
    #
    # - honour MIN_SHARPNESS when possible
    # - if the source is generally soft, use its 25th percentile
    #   instead of rejecting almost everything.
    values = np.asarray(
        [
            x["sharpness"]
            for x in observations
        ],
        dtype=np.float64,
    )

    percentile_threshold = float(
        np.percentile(
            values,
            25.0,
        )
    )

    threshold = max(
        MIN_SHARPNESS,
        percentile_threshold * 0.35,
    )

    usable = [
        x
        for x in observations
        if x["sharpness"] >= threshold
    ]

    if len(usable) < min(
        8,
        len(observations),
    ):

        usable = sorted(
            observations,
            key=lambda x: x["sharpness"],
            reverse=True,
        )[
            :min(
                8,
                len(observations),
            )
        ]

    # Rank quality, then preserve temporal diversity by taking bins.
    usable.sort(
        key=lambda x: x["video_frame_id"]
    )

    if len(usable) > MAX_TEXTURE_FRAMES:

        bins = np.linspace(
            0,
            len(usable),
            MAX_TEXTURE_FRAMES + 1,
        ).astype(np.int64)

        selected = []

        for a, b in zip(
            bins[:-1],
            bins[1:],
        ):

            group = usable[
                int(a):int(b)
            ]

            if not group:
                continue

            best = max(
                group,
                key=lambda x: x["sharpness"],
            )

            selected.append(
                best
            )

        usable = selected

    # Final temporal sort.
    usable.sort(
        key=lambda x: x["video_frame_id"]
    )

    print(
        f"[OK] Selected "
        f"{len(usable)} texture observations."
    )

    print(
        f"     Sharpness threshold: "
        f"{threshold:.3f}"
    )

    return usable


# ============================================================
# UV / MESH VALIDATION
# ============================================================

def validate_uvs(mesh):

    if not UV_FILE.exists():

        raise RuntimeError(
            f"UV file missing: {UV_FILE}"
        )

    uvs = np.asarray(
        np.load(UV_FILE),
        dtype=np.float32,
    )

    triangles = np.asarray(
        mesh.triangles
    )

    if (
        uvs.ndim != 2
        or uvs.shape[1] != 2
    ):

        raise RuntimeError(
            f"UV array must have shape "
            f"(N, 2), got {uvs.shape}"
        )

    expected = len(triangles) * 3

    if len(uvs) != expected:

        raise RuntimeError(
            f"UV count mismatch: "
            f"got {len(uvs)}, "
            f"expected {expected}"
        )

    if not np.all(
        np.isfinite(uvs)
    ):

        raise RuntimeError(
            "UVs contain non-finite values."
        )

    if (
        np.min(uvs) < -1e-5
        or np.max(uvs) > 1.00001
    ):

        raise RuntimeError(
            "UV coordinates are outside "
            "the supported [0,1] range."
        )

    uvs = np.clip(
        uvs,
        0.0,
        1.0,
    )

    return uvs


def create_tensor_mesh(
    legacy_mesh,
    uvs,
):
    """
    Build the Open3D tensor mesh through an OBJ round-trip so that the UVs
    are imported through Open3D's supported TriangleMeshModel path.

    Directly assigning triangle["texture_uvs"] can be visible from Python,
    yet ProjectImagesToAlbedo() may still reject the mesh because its C++
    path expects the UV face attribute to be populated as a model-imported
    texture coordinate set.

    The official Open3D example uses:

        read_triangle_model()
        from_triangle_mesh_model()

    IMPORTANT:
    Open3D tensor representation of imported UVs is:

        [num_triangles, 3, 2]

    not:

        [num_triangles * 3, 2]
    """

    uvs = np.asarray(
        uvs,
        dtype=np.float64,
    )

    vertices = np.asarray(
        legacy_mesh.vertices,
        dtype=np.float64,
    )

    triangles = np.asarray(
        legacy_mesh.triangles,
        dtype=np.int64,
    )

    expected = len(triangles) * 3

    if uvs.shape != (
        expected,
        2,
    ):

        raise RuntimeError(
            f"UV array must have shape "
            f"({expected}, 2), got {uvs.shape}"
        )

    if not np.all(
        np.isfinite(uvs)
    ):

        raise RuntimeError(
            "UVs contain non-finite values."
        )

    # Keep OBJ UV coordinates in the documented [0,1] range.
    uvs = np.clip(
        uvs,
        0.0,
        1.0,
    )

    import contextlib

    # Open3D may retain a handle to the imported OBJ for the lifetime of the
    # tensor mesh. Keep a stable bridge inside the project output directory so
    # Windows temp-folder ACLs and cleanup of an open file cannot interrupt
    # projection.
    tmp = OUTPUT_DIR / ".aria_uv_bridge_runtime"
    tmp.mkdir(parents=True, exist_ok=True)
    with contextlib.nullcontext(tmp):

        obj_path = (
            Path(tmp)
            / "mesh_with_uvs.obj"
        )

        # ----------------------------------------------------
        # WRITE TEMPORARY OBJ
        # ----------------------------------------------------

        # OBJ is used only as a standards-compliant bridge
        # carrying the already-computed ARIA UVs.
        #
        # One vt entry is written per triangle corner, so
        # arbitrary UV seams are preserved exactly.

        with obj_path.open(
            "w",
            encoding="utf-8",
            newline="\n",
        ) as f:

            f.write(
                "# ARIA-S3D UV bridge\n"
            )

            # Geometry vertices
            for v in vertices:

                f.write(
                    f"v {v[0]:.17g} "
                    f"{v[1]:.17g} "
                    f"{v[2]:.17g}\n"
                )

            # UV coordinates
            for uv in uvs:

                f.write(
                    f"vt {uv[0]:.17g} "
                    f"{uv[1]:.17g}\n"
                )

            # Faces
            #
            # OBJ indices are 1-based.
            #
            # Geometry vertices use the original mesh vertex
            # indices.
            #
            # UV indices follow the flattened per-corner UV
            # array.

            for tri_idx, tri in enumerate(
                triangles
            ):

                a = (
                    3 * tri_idx
                    + 1
                )

                f.write(
                    f"f "
                    f"{int(tri[0]) + 1}/{a} "
                    f"{int(tri[1]) + 1}/{a + 1} "
                    f"{int(tri[2]) + 1}/{a + 2}\n"
                )

        # ----------------------------------------------------
        # IMPORT THROUGH OPEN3D TRIANGLEMESHMODEL
        # ----------------------------------------------------

        model = o3d.io.read_triangle_model(
            str(obj_path),
            print_progress=False,
        )

        meshes = (
            o3d.t.geometry.TriangleMesh
            .from_triangle_mesh_model(
                model
            )
        )

        if not meshes:

            raise RuntimeError(
                "Open3D OBJ bridge produced "
                "no tensor meshes."
            )

        # A single-material bridge is expected.
        #
        # If an OBJ importer ever creates multiple mesh parts,
        # refuse to silently discard geometry.

        parts = list(
            meshes.values()
        )

        tensor_mesh = parts[0]

        if len(parts) > 1:

            raise RuntimeError(
                "Open3D OBJ bridge produced "
                "multiple mesh parts; refusing "
                "to silently discard geometry."
            )

        # ----------------------------------------------------
        # VERIFY TEXTURE UV ATTRIBUTE
        # ----------------------------------------------------

        if (
            "texture_uvs"
            not in tensor_mesh.triangle
        ):

            raise RuntimeError(
                "OBJ round-trip did not produce "
                "Open3D triangle texture_uvs."
            )

        stored = (
            tensor_mesh
            .triangle[
                "texture_uvs"
            ]
        )

        # IMPORTANT:
        #
        # Open3D tensor representation is:
        #
        #     [num_triangles, 3, 2]
        #
        # NOT:
        #
        #     [num_triangles * 3, 2]

        expected_shape = [
            len(triangles),
            3,
            2,
        ]

        if list(
            stored.shape
        ) != expected_shape:

            raise RuntimeError(
                "Open3D imported texture_uvs "
                "has unexpected shape: "
                f"{stored.shape}; "
                f"expected {expected_shape}."
            )

        # Flatten only for comparison with
        # our original UV array.

        imported = (
            stored
            .numpy()
            .reshape(-1, 2)
        )

        if not np.all(
            np.isfinite(imported)
        ):

            raise RuntimeError(
                "Imported Open3D texture_uvs "
                "contain non-finite values."
            )

        # ----------------------------------------------------
        # UV ROUND-TRIP VALIDATION
        # ----------------------------------------------------

        uv_error = (
            float(
                np.max(
                    np.abs(
                        imported - uvs
                    )
                )
            )
            if imported.size
            else 0.0
        )

        if uv_error > 1e-5:

            raise RuntimeError(
                "Open3D OBJ UV round-trip "
                "changed the UV coordinates "
                "too much: "
                f"max error={uv_error:.8g}"
            )

        print(
            "[OK] OBJ UV bridge -> "
            "tensor texture_uvs verified."
        )

        print(
            f"     Imported UV shape : "
            f"{stored.shape}"
        )

        print(
            f"     UV round-trip err : "
            f"{uv_error:.8g}"
        )

        print(
            f"     Imported UV range: "
            f"{float(imported.min()):.6f} .. "
            f"{float(imported.max()):.6f}"
        )

        return tensor_mesh


# ============================================================
# IMAGE / CAMERA ARRAY CONSTRUCTION
# ============================================================

def build_projection_inputs(
    observations,
    ba_frame_ids,
    mapped_video_ids,
    camera_extrinsics,
):
    """
    Keep source image, BA camera and frame-id association intact.

    observations are selected by source video frame ID.
    """

    by_video_id = {
        int(obs["video_frame_id"]): obs
        for obs in observations
    }

    images = []
    extrinsics = []
    selected_ba_ids = []
    selected_video_ids = []
    sharpness = []

    for (
        ba_id,
        video_id,
        ext,
    ) in zip(
        ba_frame_ids,
        mapped_video_ids,
        camera_extrinsics,
    ):

        video_id = int(
            video_id
        )

        if video_id not in by_video_id:
            continue

        obs = by_video_id[
            video_id
        ]

        rgb = cv2.cvtColor(
            obs["frame"],
            cv2.COLOR_BGR2RGB,
        )

        image = o3d.t.geometry.Image(
            rgb
        )

        images.append(
            image
        )

        extrinsics.append(
            ext
        )

        selected_ba_ids.append(
            int(ba_id)
        )

        selected_video_ids.append(
            video_id
        )

        sharpness.append(
            float(
                obs["sharpness"]
            )
        )

    return (
        images,
        extrinsics,
        selected_ba_ids,
        selected_video_ids,
        sharpness,
    )


def tensor_list(
    array_list,
):

    return [
        o3d.core.Tensor(
            np.asarray(
                x,
                dtype=np.float32,
            ),
            dtype=o3d.core.Dtype.Float32,
        )
        for x in array_list
    ]


# ============================================================
# TEXTURE VALIDATION
# ============================================================

def validate_texture(
    texture,
):

    texture = np.asarray(
        texture
    )

    if texture.ndim == 2:

        texture = np.repeat(
            texture[:, :, None],
            3,
            axis=2,
        )

    if (
        texture.ndim != 3
        or texture.shape[2] < 3
    ):

        raise RuntimeError(
            f"Projected texture has "
            f"invalid shape: {texture.shape}"
        )

    texture = texture[:, :, :3]

    if not np.all(
        np.isfinite(
            texture.astype(
                np.float32
            )
        )
    ):

        raise RuntimeError(
            "Projected texture contains "
            "non-finite values."
        )

    texture = np.clip(
        texture,
        0,
        255,
    ).astype(np.uint8)

    nonzero_ratio = float(
        np.count_nonzero(
            np.any(
                texture > 2,
                axis=2,
            )
        )
        / texture.shape[0]
        / texture.shape[1]
    )

    if (
        nonzero_ratio
        < MIN_TEXTURE_NONZERO_RATIO
    ):

        raise RuntimeError(
            "Projected texture contains "
            "too little non-zero image data."
        )

    return (
        texture,
        nonzero_ratio,
    )


def pad_projected_texture(
    texture,
    mesh,
    uvs,
):
    """Pad small empty gaps inside the packed XAtlas triangle footprint.

    Open3D back-projects photographic samples into the atlas but can leave
    isolated empty texels between samples. Copying the nearest already
    projected color for a bounded radius is conventional texture padding and
    preserves photographic colors without inventing a gradient or fallback.
    """

    triangles = np.asarray(mesh.triangles, dtype=np.int64)
    uvs = np.asarray(uvs, dtype=np.float32)

    if (
        uvs.shape != (len(triangles) * 3, 2)
        or len(triangles) == 0
    ):
        print("[WARN] UV/triangle shape invalid; skipping texture padding.")
        return texture, 0

    height, width = texture.shape[:2]
    uv_triangles = uvs.reshape(-1, 3, 2)
    polygons = np.rint(
        np.stack(
            (
                uv_triangles[:, :, 0] * (width - 1),
                (1.0 - uv_triangles[:, :, 1]) * (height - 1),
            ),
            axis=2,
        )
    ).astype(np.int32)

    atlas_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(atlas_mask, list(polygons), 255)

    known = (
        np.any(texture > 2, axis=2)
        & (atlas_mask > 0)
    )

    if not np.any(known):
        return texture, 0

    filled = known.copy()
    filled_pixels = 0
    kernel = np.ones((3, 3), dtype=np.uint8)
    height_pad = height + 2
    width_pad = width + 2
    offsets = (
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    )

    for _ in range(TEXTURE_PADDING_PIXELS):
        frontier = (
            cv2.dilate(filled.astype(np.uint8), kernel, iterations=1) > 0
        )
        frontier &= (atlas_mask > 0) & ~filled

        count = int(np.count_nonzero(frontier))
        if count == 0:
            break

        filled_pad = np.pad(filled, 1, mode="constant")
        texture_pad = np.pad(
            texture,
            ((1, 1), (1, 1), (0, 0)),
            mode="constant",
        )
        assigned = np.zeros((height, width), dtype=bool)

        for dy, dx in offsets:
            y0 = 1 + dy
            y1 = y0 + height
            x0 = 1 + dx
            x1 = x0 + width
            source = filled_pad[y0:y1, x0:x1]
            target = frontier & source & ~assigned
            if np.any(target):
                texture[target] = texture_pad[y0:y1, x0:x1][target]
                assigned[target] = True

        filled |= frontier
        filled_pixels += int(np.count_nonzero(assigned))

    return texture, filled_pixels


def bake_visibility_aware_texture(
    mesh,
    corner_uvs,
    images,
    extrinsics,
    K,
    sharpness,
):
    """Bake photographic colors into UV space using calibrated visibility.

    Each triangle is evaluated against every selected camera. A camera is
    eligible only when the triangle centroid is in front of the camera, lies
    inside the source image, faces the camera, and is consistent with a coarse
    mesh depth buffer. The top few eligible views are bilinearly sampled and
    blended while rasterizing the triangle in UV space.

    This is intentionally photographic-only: triangles without a valid view
    stay empty and are handled only by the bounded atlas-padding step.
    """

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.triangles, dtype=np.int64)
    uv_triangles = np.asarray(corner_uvs, dtype=np.float64).reshape(-1, 3, 2)

    if len(triangles) != len(uv_triangles):
        raise RuntimeError("Visibility baker received mismatched triangle/UV counts.")
    if not images or not extrinsics:
        raise RuntimeError("Visibility baker received no images or camera poses.")

    image_arrays = []
    for image in images:
        if hasattr(image, "as_tensor"):
            array = image.as_tensor().numpy()
        else:
            array = np.asarray(image)
        array = np.asarray(array)
        if array.ndim != 3 or array.shape[2] < 3:
            raise RuntimeError(f"Visibility baker received invalid image shape: {array.shape}")
        image_arrays.append(array[..., :3].astype(np.uint8, copy=False))

    height, width = image_arrays[0].shape[:2]
    if any(array.shape[:2] != (height, width) for array in image_arrays):
        raise RuntimeError("Visibility baker requires equally sized source images.")

    triangle_points = vertices[triangles]
    centers = triangle_points.mean(axis=1)
    edge_a = triangle_points[:, 1] - triangle_points[:, 0]
    edge_b = triangle_points[:, 2] - triangle_points[:, 0]
    normals = np.cross(edge_a, edge_b)
    normal_lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals /= np.maximum(normal_lengths, 1e-12)

    face_count = len(triangles)
    top_scores = np.full((face_count, VISIBILITY_BAKE_TOP_VIEWS), -np.inf)
    top_camera_ids = np.full((face_count, VISIBILITY_BAKE_TOP_VIEWS), -1, dtype=np.int32)

    sharpness = np.asarray(sharpness, dtype=np.float64).reshape(-1)
    sharpness_scale = float(np.percentile(sharpness, 95)) if len(sharpness) else 1.0
    sharpness_scale = max(sharpness_scale, 1.0)

    depth_scale = VISIBILITY_BAKE_DEPTH_SCALE
    depth_width = max(1, int(math.ceil(width / depth_scale)))
    depth_height = max(1, int(math.ceil(height / depth_scale)))
    valid_camera_count = 0
    depth_reject_count = 0

    for camera_id, (ext, image) in enumerate(zip(extrinsics, image_arrays)):
        R = np.asarray(ext, dtype=np.float64)[:3, :3]
        t = np.asarray(ext, dtype=np.float64)[:3, 3]

        camera_vertices = (R @ vertices.T).T + t
        vertex_depth = camera_vertices[:, 2]
        vertex_projection = (K @ camera_vertices.T).T
        vertex_valid = vertex_depth > 1e-8
        vertex_uv = np.zeros((len(vertices), 2), dtype=np.float64)
        vertex_uv[vertex_valid] = (
            vertex_projection[vertex_valid, :2]
            / vertex_projection[vertex_valid, 2:3]
        )

        depth_buffer = np.full((depth_height, depth_width), np.inf, dtype=np.float32)
        depth_valid = (
            vertex_valid
            & (vertex_uv[:, 0] >= 0)
            & (vertex_uv[:, 0] < width)
            & (vertex_uv[:, 1] >= 0)
            & (vertex_uv[:, 1] < height)
        )
        depth_x = np.clip((vertex_uv[depth_valid, 0] / depth_scale).astype(np.int32), 0, depth_width - 1)
        depth_y = np.clip((vertex_uv[depth_valid, 1] / depth_scale).astype(np.int32), 0, depth_height - 1)
        np.minimum.at(
            depth_buffer,
            (depth_y, depth_x),
            vertex_depth[depth_valid].astype(np.float32),
        )
        finite_depth = np.isfinite(depth_buffer)
        if np.any(finite_depth):
            # A small erosion makes the sparse vertex z-buffer usable for face
            # centroids while preserving the nearest visible surface.
            depth_work = depth_buffer.copy()
            fill_value = float(np.max(depth_work[finite_depth]))
            depth_work[~finite_depth] = fill_value
            depth_buffer = cv2.erode(depth_work, np.ones((3, 3), np.uint8))

        camera_center = -R.T @ t
        to_camera = camera_center[None, :] - centers
        to_camera /= np.maximum(np.linalg.norm(to_camera, axis=1, keepdims=True), 1e-12)
        facing = np.sum(normals * to_camera, axis=1)

        camera_centers = (R @ centers.T).T + t
        center_depth = camera_centers[:, 2]
        center_projection = (K @ camera_centers.T).T
        center_uv = np.zeros((face_count, 2), dtype=np.float64)
        center_valid = center_depth > 1e-8
        center_uv[center_valid] = (
            center_projection[center_valid, :2]
            / center_projection[center_valid, 2:3]
        )
        inside = (
            center_valid
            & (center_uv[:, 0] >= IMAGE_BORDER_MARGIN)
            & (center_uv[:, 0] < width - IMAGE_BORDER_MARGIN)
            & (center_uv[:, 1] >= IMAGE_BORDER_MARGIN)
            & (center_uv[:, 1] < height - IMAGE_BORDER_MARGIN)
        )

        depth_x = np.clip((center_uv[:, 0] / depth_scale).astype(np.int32), 0, depth_width - 1)
        depth_y = np.clip((center_uv[:, 1] / depth_scale).astype(np.int32), 0, depth_height - 1)
        center_z_ref = depth_buffer[depth_y, depth_x]
        depth_consistent = (
            np.isfinite(center_z_ref)
            & (center_depth <= center_z_ref * (1.0 + VISIBILITY_BAKE_DEPTH_TOLERANCE) + 1e-3)
        )

        if np.any(inside & depth_consistent):
            valid_camera_count += 1
        depth_reject_count += int(np.count_nonzero(inside & ~depth_consistent))

        projected_triangle = vertex_uv[triangles]
        projected_area = 0.5 * np.abs(
            (projected_triangle[:, 1, 0] - projected_triangle[:, 0, 0])
            * (projected_triangle[:, 2, 1] - projected_triangle[:, 0, 1])
            - (projected_triangle[:, 1, 1] - projected_triangle[:, 0, 1])
            * (projected_triangle[:, 2, 0] - projected_triangle[:, 0, 0])
        )
        footprint_score = np.clip(np.log1p(projected_area) / 8.0, 0.25, 1.0)
        quality_score = 0.5 + 0.5 * min(
            safe_float(sharpness[camera_id] if camera_id < len(sharpness) else 0.0),
            sharpness_scale,
        ) / sharpness_scale
        scores = facing * footprint_score * quality_score
        eligible = inside & depth_consistent & (facing >= VISIBILITY_BAKE_MIN_FACING)
        scores[~eligible] = -np.inf

        for slot in range(VISIBILITY_BAKE_TOP_VIEWS):
            better = scores > top_scores[:, slot]
            if not np.any(better):
                continue
            if slot + 1 < VISIBILITY_BAKE_TOP_VIEWS:
                top_scores[better, slot + 1:] = top_scores[better, slot:-1]
                top_camera_ids[better, slot + 1:] = top_camera_ids[better, slot:-1]
            top_scores[better, slot] = scores[better]
            top_camera_ids[better, slot] = camera_id

    atlas = np.zeros((TEXTURE_SIZE, TEXTURE_SIZE, 3), dtype=np.float32)
    atlas_weight = np.zeros((TEXTURE_SIZE, TEXTURE_SIZE), dtype=np.float32)
    faces_with_views = int(np.count_nonzero(top_camera_ids[:, 0] >= 0))
    faces_rasterized = 0

    for face_id, face_uv in enumerate(uv_triangles):
        if top_camera_ids[face_id, 0] < 0:
            continue

        uv_pixels = face_uv.copy()
        uv_pixels[:, 0] *= TEXTURE_SIZE - 1
        uv_pixels[:, 1] = (1.0 - uv_pixels[:, 1]) * (TEXTURE_SIZE - 1)
        min_x = max(0, int(np.floor(np.min(uv_pixels[:, 0]))))
        max_x = min(TEXTURE_SIZE - 1, int(np.ceil(np.max(uv_pixels[:, 0]))))
        min_y = max(0, int(np.floor(np.min(uv_pixels[:, 1]))))
        max_y = min(TEXTURE_SIZE - 1, int(np.ceil(np.max(uv_pixels[:, 1]))))
        if max_x < min_x or max_y < min_y:
            continue

        grid_y, grid_x = np.mgrid[min_y:max_y + 1, min_x:max_x + 1]
        p = np.stack((grid_x, grid_y), axis=-1).astype(np.float64)
        a = uv_pixels[0]
        b = uv_pixels[1]
        c = uv_pixels[2]
        denominator = ((b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1]))
        if abs(denominator) < 1e-12:
            continue
        bary_a = ((b[1] - c[1]) * (p[..., 0] - c[0]) + (c[0] - b[0]) * (p[..., 1] - c[1])) / denominator
        bary_b = ((c[1] - a[1]) * (p[..., 0] - c[0]) + (a[0] - c[0]) * (p[..., 1] - c[1])) / denominator
        bary_c = 1.0 - bary_a - bary_b
        footprint = (bary_a >= -1e-5) & (bary_b >= -1e-5) & (bary_c >= -1e-5)
        if not np.any(footprint):
            continue

        bary = np.stack((bary_a, bary_b, bary_c), axis=-1)
        world = np.einsum("...k,kd->...d", bary, triangle_points[face_id])
        flat_world = world.reshape(-1, 3)
        flat_footprint = footprint.reshape(-1)
        color_sum = np.zeros((len(flat_world), 3), dtype=np.float32)
        weight_sum = np.zeros(len(flat_world), dtype=np.float32)

        for slot in range(VISIBILITY_BAKE_TOP_VIEWS):
            camera_id = int(top_camera_ids[face_id, slot])
            score = float(top_scores[face_id, slot])
            if camera_id < 0 or not np.isfinite(score):
                continue
            ext = np.asarray(extrinsics[camera_id], dtype=np.float64)
            R = ext[:3, :3]
            t = ext[:3, 3]
            camera_world = (R @ flat_world.T).T + t
            depth = camera_world[:, 2]
            projection = (K @ camera_world.T).T
            valid = flat_footprint & (depth > 1e-8)
            projected = np.zeros((len(flat_world), 2), dtype=np.float32)
            projected[valid] = projection[valid, :2] / projection[valid, 2:3]
            valid &= (
                (projected[:, 0] >= 0)
                & (projected[:, 0] < width - 1)
                & (projected[:, 1] >= 0)
                & (projected[:, 1] < height - 1)
            )
            if not np.any(valid):
                continue
            sampled = cv2.remap(
                image_arrays[camera_id],
                projected[:, 0].reshape(-1, 1),
                projected[:, 1].reshape(-1, 1),
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            ).reshape(-1, 3)
            weight = max(score, 1e-6)
            color_sum[valid] += sampled[valid].astype(np.float32) * weight
            weight_sum[valid] += weight

        valid_colors = weight_sum > 0
        if not np.any(valid_colors):
            continue
        color_sum[valid_colors] /= weight_sum[valid_colors, None]
        local_y = grid_y.reshape(-1)[valid_colors]
        local_x = grid_x.reshape(-1)[valid_colors]
        atlas[local_y, local_x] += color_sum[valid_colors]
        atlas_weight[local_y, local_x] += 1.0
        faces_rasterized += 1

    populated = atlas_weight > 0
    atlas[populated] /= atlas_weight[populated, None]
    atlas = np.clip(atlas, 0, 255).astype(np.uint8)
    stats = {
        "cameras_considered": len(extrinsics),
        "cameras_with_visible_faces": valid_camera_count,
        "faces_with_views": faces_with_views,
        "faces_rasterized": faces_rasterized,
        "depth_rejects": depth_reject_count,
        "raw_coverage": float(np.mean(np.any(atlas > 2, axis=2))),
    }
    return atlas, stats


def save_texture(
    texture,
):

    ok = cv2.imwrite(
        str(OUTPUT_TEXTURE),
        cv2.cvtColor(
            texture,
            cv2.COLOR_RGB2BGR,
        ),
    )

    if not ok:

        raise RuntimeError(
            f"Could not save texture: "
            f"{OUTPUT_TEXTURE}"
        )


# ============================================================
# MATERIAL / MODEL EXPORT
# ============================================================

def attach_material(
    tensor_mesh,
    texture,
):

    tensor_mesh.material.material_name = (
        "defaultLit"
    )

    tensor_mesh.material.texture_maps[
        "albedo"
    ] = o3d.t.geometry.Image(
        texture
    )


def save_glb(
    tensor_mesh,
):

    ok = o3d.t.io.write_triangle_mesh(
        str(OUTPUT_GLTF),
        tensor_mesh,
        write_vertex_normals=True,
        write_vertex_colors=True,
        write_triangle_uvs=True,
        print_progress=False,
    )

    if not ok:

        raise RuntimeError(
            f"Open3D GLB writer returned False: "
            f"{OUTPUT_GLTF}"
        )


def save_obj(
    tensor_mesh,
):

    ok = o3d.t.io.write_triangle_mesh(
        str(OUTPUT_OBJ),
        tensor_mesh,
        write_vertex_normals=True,
        write_vertex_colors=True,
        write_triangle_uvs=True,
        print_progress=False,
    )

    if not ok:

        raise RuntimeError(
            f"Open3D OBJ writer returned False: "
            f"{OUTPUT_OBJ}"
        )

    # Open3D writes the OBJ and MTL but does not always emit the albedo map
    # binding. Add the standard relative map_Kd reference for portable OBJ
    # viewers; the GLB remains the primary textured deliverable.
    mtl_path = OUTPUT_OBJ.with_suffix(".mtl")
    mtl_path.write_text(
        "# ARIA-S3D textured OBJ material\n"
        "newmtl aria_textured_mesh_0\n"
        "Ka 1.000 1.000 1.000\n"
        "Kd 1.000 1.000 1.000\n"
        "Ks 0.000 0.000 0.000\n"
        f"map_Kd {OUTPUT_TEXTURE.name}\n",
        encoding="utf-8",
    )


# ============================================================
# REPORT
# ============================================================

def write_stats(
    *,
    start_time,
    video_path,
    video_info,
    mesh,
    uv_count,
    ba_source,
    ba_count,
    mapping_name,
    mapping_results,
    calibration_source,
    K,
    selected_ba_ids,
    selected_video_ids,
    sharpness,
    texture,
    projected_ratio,
    filled_pixels,
    nonzero_ratio,
    projection_method="visibility_aware_multiview_bake",
    bake_stats=None,
):

    elapsed = (
        time.time()
        - start_time
    )

    sharpness = np.asarray(
        sharpness,
        dtype=np.float64,
    )

    mean_sharpness = (
        float(
            np.mean(sharpness)
        )
        if len(sharpness)
        else 0.0
    )

    min_sharpness = (
        float(
            np.min(sharpness)
        )
        if len(sharpness)
        else 0.0
    )

    max_sharpness = (
        float(
            np.max(sharpness)
        )
        if len(sharpness)
        else 0.0
    )

    mapping_lines = []

    for result in mapping_results:

        mapping_lines.append(
            f"  {result['name']}: "
            f"score={result['score']:.4f}, "
            f"readable_probe={result['readable']}, "
            f"frame_correspondence="
            f"{result['mean_correspondence']:.6f}, "
            f"probe_sharpness="
            f"{result['mean_sharpness']:.4f}"
        )

    bake_stats = bake_stats or {}

    text = f"""
ARIA-S3D | PHASE 5.9
UNIVERSAL TEXTURE PROJECTION ENGINE
======================================================================

STATUS
----------------------------------------------------------------------

Texture projection   : PASS
Projection method    : {projection_method}
Synthetic fallback   : DISABLED
Monocular scale      : arbitrary

---------------------------------------------------------------------- 
INPUT VIDEO
----------------------------------------------------------------------

Source video         : {video_path}
Resolution           : {video_info['width']} x {video_info['height']}
FPS                  : {video_info['fps']:.6f}
Frames               : {video_info['total_frames']}
Duration             : {video_info['duration']:.6f} seconds

---------------------------------------------------------------------- 
MESH
----------------------------------------------------------------------

Mesh                  : {MESH_FILE}
Vertices              : {len(np.asarray(mesh.vertices))}
Triangles             : {len(np.asarray(mesh.triangles))}
UV rows               : {uv_count}

---------------------------------------------------------------------- 
BUNDLE ADJUSTMENT CAMERA MODEL
----------------------------------------------------------------------

Camera source         : {ba_source}
BA camera count       : {ba_count}
Selected BA cameras   : {len(selected_ba_ids)}

BA convention:
  [rx, ry, rz, tx, ty, tz]
  X_camera = R @ X_world + t

Open3D extrinsic:
  world-to-camera 4x4 [R | t]

---------------------------------------------------------------------- 
FRAME MAPPING
----------------------------------------------------------------------

Selected mapping      : {mapping_name}
Selected video frames : {len(selected_video_ids)}

Candidate mapping scores:
{chr(10).join(mapping_lines)}

---------------------------------------------------------------------- 
INTRINSICS
----------------------------------------------------------------------

Calibration source:
{calibration_source}

K:
{K}

---------------------------------------------------------------------- 
TEXTURE
----------------------------------------------------------------------

Resolution            : {texture.shape[1]} x {texture.shape[0]}
Channels              : {texture.shape[2]}
Raw projected coverage: {projected_ratio:.6f}
Padding texels filled : {filled_pixels}
Non-zero texture ratio: {nonzero_ratio:.6f}

VISIBILITY-AWARE BAKE
  Cameras considered       : {bake_stats.get('cameras_considered', 0)}
  Cameras with visible faces: {bake_stats.get('cameras_with_visible_faces', 0)}
  Faces with valid views    : {bake_stats.get('faces_with_views', 0)}
  Faces rasterized          : {bake_stats.get('faces_rasterized', 0)}
  Depth-consistency rejects : {bake_stats.get('depth_rejects', 0)}

Mean sharpness        : {mean_sharpness:.6f}
Minimum sharpness     : {min_sharpness:.6f}
Maximum sharpness     : {max_sharpness:.6f}

---------------------------------------------------------------------- 
OUTPUTS
----------------------------------------------------------------------

Albedo texture        : {OUTPUT_TEXTURE}
Textured GLB          : {OUTPUT_GLTF}
Textured OBJ          : {OUTPUT_OBJ}
Statistics             : {OUTPUT_STATS}

---------------------------------------------------------------------- 
PROCESSING
----------------------------------------------------------------------

Processing time       : {elapsed:.6f} seconds

---------------------------------------------------------------------- 
UNIVERSALITY
----------------------------------------------------------------------

Input video filename is not hard-coded.
Camera orientation is not inferred from trajectory direction.
BA frame IDs are mapped to source-video frames automatically.
Synthetic geometry-gradient fallback is disabled.
Photographic texture projection is required for PASS.

IMPORTANT:
ARIA-S3D remains a monocular reconstruction pipeline.
Absolute metric scale remains arbitrary.
Texture quality depends on calibration, BA accuracy, mesh visibility,
camera coverage, source image quality, exposure and white balance.
"""

    OUTPUT_STATS.write_text(
        text.strip() + "\n",
        encoding="utf-8",
    )


# ============================================================
# MAIN
# ============================================================

def main():

    start_time = time.time()

    print()

    print_header(
        "ARIA-S3D | PHASE 5.9\n"
        "UNIVERSAL TEXTURE PROJECTION ENGINE"
    )

    print()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # 1. INPUT CHECK
    # --------------------------------------------------------

    print(
        "[1] Checking inputs"
    )

    print(
        "-" * 78
    )

    for path, description in (
        (
            MESH_FILE,
            "Texture-ready mesh",
        ),
        (
            UV_FILE,
            "UV coordinates",
        ),
        (
            VIDEO_PATH_FILE,
            "Source-video registration",
        ),
    ):

        if not check_file(
            path,
            description,
        ):

            sys.exit(1)

    video_path = load_video_path()

    print(
        f"[OK] Source video: "
        f"{video_path}"
    )

    print(
        f"[OK] UV layout source: "
        f"{UV_LAYOUT_SOURCE}"
    )

    # --------------------------------------------------------
    # 2. VIDEO
    # --------------------------------------------------------

    print()

    print(
        "[2] Reading source-video metadata"
    )

    print(
        "-" * 78
    )

    video_info = load_video_info(
        video_path
    )

    metadata = load_video_metadata()

    for key, value in video_info.items():

        if key != "duration":

            print(
                f"{key:<16}: {value}"
            )

    print(
        f"{'duration':<16}: "
        f"{video_info['duration']:.3f}s"
    )

    # --------------------------------------------------------
    # 3. MESH + UV
    # --------------------------------------------------------

    print()

    print(
        "[3] Loading mesh and validating UVs"
    )

    print(
        "-" * 78
    )

    mesh = o3d.io.read_triangle_mesh(
        str(MESH_FILE)
    )

    vertices = np.asarray(
        mesh.vertices
    )

    triangles = np.asarray(
        mesh.triangles
    )

    if (
        len(vertices) == 0
        or len(triangles) == 0
    ):

        raise RuntimeError(
            "Texture-ready mesh is empty."
        )

    if not np.all(
        np.isfinite(vertices)
    ):

        raise RuntimeError(
            "Mesh contains "
            "non-finite vertices."
        )

    uvs = validate_uvs(
        mesh
    )

    print(
        f"Vertices             : "
        f"{len(vertices)}"
    )

    print(
        f"Triangles            : "
        f"{len(triangles)}"
    )

    print(
        f"UV rows              : "
        f"{len(uvs)}"
    )

    print(
        f"UV range             : "
        f"{uvs.min():.6f} .. "
        f"{uvs.max():.6f}"
    )

    print(
        "[OK] Mesh and UV validation passed."
    )

    # --------------------------------------------------------
    # 4. BA CAMERAS
    # --------------------------------------------------------

    print()

    print(
        "[4] Loading Bundle Adjustment cameras"
    )

    print(
        "-" * 78
    )

    ba = load_ba_cameras()

    camera_params = ba[
        "camera_params"
    ]

    ba_frame_ids = ba[
        "frame_ids"
    ]

    print(
        f"Camera source        : "
        f"{ba['source']}"
    )

    print(
        f"Camera count         : "
        f"{len(camera_params)}"
    )

    print(
        f"First BA frame ID    : "
        f"{ba_frame_ids.min()}"
    )

    print(
        f"Last BA frame ID     : "
        f"{ba_frame_ids.max()}"
    )

    camera_extrinsics = (
        build_ba_extrinsics(
            camera_params
        )
    )

    print(
        "[OK] BA world-to-camera "
        "extrinsics generated "
        f"for {len(camera_extrinsics)} cameras."
    )

    # --------------------------------------------------------
    # 5. CAMERA INTRINSICS
    # --------------------------------------------------------

    print()

    print(
        "[5] Loading camera intrinsics"
    )

    print(
        "-" * 78
    )

    ba_data_for_K = (
        {
            "K": ba["K"]
        }
        if ba["K"] is not None
        else ba["data"]
    )

    K, calibration_source = (
        build_intrinsics(
            video_info["width"],
            video_info["height"],
            metadata,
            ba_data=ba_data_for_K,
        )
    )

    print(
        f"Calibration source   : "
        f"{calibration_source}"
    )

    print(
        "K ="
    )

    print(
        K
    )

    if not np.all(
        np.isfinite(K)
    ):

        raise RuntimeError(
            "Intrinsic matrix contains "
            "non-finite values."
        )

    # --------------------------------------------------------
    # 6. BA → VIDEO FRAME MAPPING
    # --------------------------------------------------------

    print()

    print(
        "[6] Mapping BA cameras to "
        "source-video frames"
    )

    print(
        "-" * 78
    )

    (
        mapped_video_ids,
        mapping_name,
        mapping_results,
    ) = select_best_frame_mapping(
        video_path,
        ba_frame_ids,
        video_info[
            "total_frames"
        ],
        metadata,
    )

    # --------------------------------------------------------
    # 7. SELECT TEXTURE OBSERVATIONS
    # --------------------------------------------------------

    print()

    print(
        "[7] Selecting high-quality "
        "texture observations"
    )

    print(
        "-" * 78
    )

    observations = (
        select_texture_observations(
            video_path,
            mapped_video_ids,
            video_info["width"],
            video_info["height"],
        )
    )

    if not observations:

        raise RuntimeError(
            "No usable source-video "
            "observations were selected."
        )

    # --------------------------------------------------------
    # 8. ALIGN IMAGES WITH BA CAMERAS
    # --------------------------------------------------------

    print()

    print(
        "[8] Aligning images with "
        "optimized BA poses"
    )

    print(
        "-" * 78
    )

    (
        images,
        extrinsics,
        selected_ba_ids,
        selected_video_ids,
        sharpness,
    ) = build_projection_inputs(
        observations,
        ba_frame_ids,
        mapped_video_ids,
        camera_extrinsics,
    )

    if not images:

        raise RuntimeError(
            "No selected video observations "
            "could be paired with BA cameras."
        )

    valid_ratio = (
        len(images)
        / max(
            1,
            len(observations),
        )
    )

    print(
        f"Paired images         : "
        f"{len(images)}"
    )

    print(
        f"Paired BA cameras     : "
        f"{len(extrinsics)}"
    )

    print(
        f"Pairing ratio         : "
        f"{valid_ratio:.3f}"
    )

    if (
        valid_ratio
        < MIN_VALID_CAMERA_RATIO
    ):

        raise RuntimeError(
            "Too few selected texture "
            "frames have matching "
            "BA cameras."
        )

    # --------------------------------------------------------
    # 9. CREATE TENSOR MESH
    # --------------------------------------------------------

    print()

    print(
        "[9] Preparing Open3D tensor mesh"
    )

    print(
        "-" * 78
    )

    tensor_mesh = create_tensor_mesh(
        mesh,
        uvs,
    )

    print(
        f"texture_uvs shape    : "
        f"{tensor_mesh.triangle['texture_uvs'].shape}"
    )

    print(
        "[OK] Open3D model-imported "
        "texture_uvs is ready for projection."
    )

    # --------------------------------------------------------
    # 10. PROJECT
    # --------------------------------------------------------

    print()

    print(
        "[10] Running calibrated "
        "photographic projection"
    )

    print(
        "-" * 78
    )

    intrinsic_tensors = [

        o3d.core.Tensor(
            K.astype(
                np.float32
            ),
            dtype=o3d.core.Dtype.Float32,
        )

        for _ in images

    ]

    extrinsic_tensors = (
        tensor_list(
            extrinsics
        )
    )

    print(
        f"Images supplied       : "
        f"{len(images)}"
    )

    print(
        f"Texture size          : "
        f"{TEXTURE_SIZE} x "
        f"{TEXTURE_SIZE}"
    )

    print(
        "Projection backend    : "
        "Open3D CPU"
    )

    projection_start = time.time()

    try:

        projected = (
            tensor_mesh
            .project_images_to_albedo(
                images,
                intrinsic_tensors,
                extrinsic_tensors,
                tex_size=TEXTURE_SIZE,
                update_material=True,
            )
        )

    except Exception as exc:

        raise RuntimeError(
            "Photographic texture "
            "projection failed. "
            "No synthetic fallback "
            "is permitted.\n"
            f"Open3D error: {exc}"
        ) from exc

    projection_time = (
        time.time()
        - projection_start
    )

    open3d_texture, open3d_ratio = (
        validate_texture(
            projected.as_tensor().numpy()
        )
    )

    print()
    print(
        "[10A] Running visibility-aware multi-view UV bake"
    )
    print(
        "-" * 78
    )

    visibility_bake_start = time.time()
    try:
        visibility_texture, bake_stats = bake_visibility_aware_texture(
            mesh,
            uvs,
            images,
            extrinsics,
            K,
            sharpness,
        )
        visibility_texture, visibility_ratio = validate_texture(
            visibility_texture
        )
        visibility_bake_time = time.time() - visibility_bake_start
        print(
            f"[OK] Visibility-aware bake completed in "
            f"{visibility_bake_time:.3f}s."
        )
        print(
            f"[OK] Faces with valid views: "
            f"{bake_stats['faces_with_views']:,}"
        )
        print(
            f"[OK] Visibility-aware raw coverage: "
            f"{visibility_ratio:.6f}"
        )
    except Exception as exc:
        print(
            f"[WARN] Visibility-aware bake failed; using Open3D result: {exc}"
        )
        visibility_texture = open3d_texture
        visibility_ratio = open3d_ratio
        bake_stats = {
            "cameras_considered": 0,
            "cameras_with_visible_faces": 0,
            "faces_with_views": 0,
            "faces_rasterized": 0,
            "depth_rejects": 0,
        }

    if visibility_ratio >= open3d_ratio:
        texture = visibility_texture
        projected_ratio = visibility_ratio
        projection_method = "visibility_aware_multiview_bake"
        print(
            f"[OK] Selected visibility-aware bake over Open3D reference "
            f"({visibility_ratio:.6f} >= {open3d_ratio:.6f})."
        )
    else:
        texture = open3d_texture
        projected_ratio = open3d_ratio
        projection_method = "calibrated_BA_video_projection"
        print(
            f"[WARN] Open3D reference retained "
            f"({open3d_ratio:.6f} > {visibility_ratio:.6f})."
        )

    texture, filled_pixels = pad_projected_texture(
        texture,
        mesh,
        uvs,
    )

    nonzero_ratio = float(
        np.mean(np.any(texture > 2, axis=2))
    )

    print(
        f"[OK] Photographic projection "
        f"completed in "
        f"{projection_time:.3f}s."
    )

    print(
        f"[OK] Raw projected coverage: "
        f"{projected_ratio:.6f}"
    )

    print(
        f"[OK] Atlas padding filled: "
        f"{filled_pixels:,} texels"
    )

    print(
        f"[OK] Final texture coverage: "
        f"{nonzero_ratio:.6f}"
    )

    # --------------------------------------------------------
    # 11. SAVE ALBEDO
    # --------------------------------------------------------

    print()

    print(
        "[11] Saving albedo texture"
    )

    print(
        "-" * 78
    )

    save_texture(
        texture
    )

    print(
        f"[OK] {OUTPUT_TEXTURE}"
    )

    # --------------------------------------------------------
    # 12. ATTACH MATERIAL
    # --------------------------------------------------------

    print()

    print(
        "[12] Attaching albedo material"
    )

    print(
        "-" * 78
    )

    attach_material(
        tensor_mesh,
        texture,
    )

    print(
        "[OK] defaultLit material "
        "+ albedo attached."
    )

    # --------------------------------------------------------
    # 13. EXPORT GLB
    # --------------------------------------------------------

    print()

    print(
        "[13] Exporting primary "
        "textured GLB"
    )

    print(
        "-" * 78
    )

    save_glb(
        tensor_mesh
    )

    print(
        f"[OK] {OUTPUT_GLTF}"
    )

    # --------------------------------------------------------
    # 14. EXPORT OBJ
    # --------------------------------------------------------

    print()

    print(
        "[14] Exporting secondary "
        "textured OBJ"
    )

    print(
        "-" * 78
    )

    save_obj(
        tensor_mesh
    )

    print(
        f"[OK] {OUTPUT_OBJ}"
    )

    # --------------------------------------------------------
    # 15. REPORT
    # --------------------------------------------------------

    print()

    print(
        "[15] Writing validation statistics"
    )

    print(
        "-" * 78
    )

    write_stats(
        start_time=start_time,
        video_path=video_path,
        video_info=video_info,
        mesh=mesh,
        uv_count=len(uvs),
        ba_source=ba["source"],
        ba_count=len(camera_params),
        mapping_name=mapping_name,
        mapping_results=mapping_results,
        calibration_source=calibration_source,
        K=K,
        selected_ba_ids=selected_ba_ids,
        selected_video_ids=selected_video_ids,
        sharpness=sharpness,
        texture=texture,
        projected_ratio=projected_ratio,
        filled_pixels=filled_pixels,
        nonzero_ratio=nonzero_ratio,
        projection_method=projection_method,
        bake_stats=bake_stats,
    )

    print(
        f"[OK] {OUTPUT_STATS}"
    )

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    elapsed = (
        time.time()
        - start_time
    )

    print()

    print_header(
        "ARIA-S3D | PHASE 5.9 PASS"
    )

    print()

    print(
        f"Mesh vertices        : "
        f"{len(vertices)}"
    )

    print(
        f"Mesh triangles       : "
        f"{len(triangles)}"
    )

    print(
        f"BA cameras           : "
        f"{len(camera_params)}"
    )

    print(
        f"Texture images       : "
        f"{len(images)}"
    )

    print(
        f"Texture resolution   : "
        f"{TEXTURE_SIZE} x "
        f"{TEXTURE_SIZE}"
    )

    print(
        f"Projection coverage  : "
        f"{nonzero_ratio:.6f}"
    )

    print(
        f"Processing time      : "
        f"{elapsed:.3f}s"
    )

    print()

    print(
        "Generated outputs:"
    )

    print(
        f"  {OUTPUT_TEXTURE}"
    )

    print(
        f"  {OUTPUT_GLTF}"
    )

    print(
        f"  {OUTPUT_OBJ}"
    )

    print(
        f"  {OUTPUT_STATS}"
    )

    print()

    print(
        "-" * 78
    )

    print(
        "NEXT STEP:"
    )

    print(
        "Phase 5.10 - Final textured-model validation"
    )

    print()

    print(
        "No synthetic texture fallback was used."
    )

    print(
        "Camera poses came from Bundle Adjustment."
    )

    print(
        "ARIA-S3D remains monocular; "
        "absolute metric scale is arbitrary."
    )

    print(
        "-" * 78
    )


if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print(
            "\n[ABORTED] Interrupted by user."
        )

        sys.exit(130)

    except Exception as exc:

        print()

        print_header(
            "ARIA-S3D | PHASE 5.9 FAILED"
        )

        print()

        print(
            f"[ERROR] {exc}"
        )

        print()

        print(
            "No PASS status was written."
        )

        print(
            "No synthetic fallback was generated."
        )

        sys.exit(1)
