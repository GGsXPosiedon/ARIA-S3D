import cv2
import numpy as np
from pathlib import Path


# ============================================================
# ARIA-S3D | PHASE 3
# BUNDLE ADJUSTMENT DATA PREPARATION
#
# Produces:
# - Consistent world-to-camera poses
# - World-space triangulated 3D points
# - 2D observations
# - Camera/point observation indices
# ============================================================

FRAME_DIR = Path("data/quality_frames")
OUTPUT_DIR = Path("data/output")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "ba_data.npz"


# ============================================================
# CAMERA INTRINSICS
# ============================================================

IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720

fx = 1280.0
fy = 1280.0

cx = IMAGE_WIDTH / 2.0
cy = IMAGE_HEIGHT / 2.0

K = np.array([
    [fx, 0.0, cx],
    [0.0, fy, cy],
    [0.0, 0.0, 1.0]
], dtype=np.float64)


# ============================================================
# LOAD FRAMES
# ============================================================

frames = sorted(
    FRAME_DIR.glob("*.jpg")
)

if len(frames) < 2:
    raise RuntimeError(
        "Need at least two quality frames."
    )


# ============================================================
# SIFT + FLANN
# ============================================================

sift = cv2.SIFT_create(
    nfeatures=3000
)

index_params = dict(
    algorithm=1,
    trees=5
)

search_params = dict(
    checks=50
)

flann = cv2.FlannBasedMatcher(
    index_params,
    search_params
)


# ============================================================
# STORAGE
# ============================================================

# World -> Camera extrinsics:
#
# X_camera = R * X_world + t

camera_R = [
    np.eye(3, dtype=np.float64)
]

camera_t = [
    np.zeros((3, 1), dtype=np.float64)
]

frame_ids = [0]

points_3d = []

camera_indices = []
point_indices = []
points_2d = []


# ============================================================
# FIRST CAMERA
# ============================================================

previous_frame_id = 0

previous_image = cv2.imread(
    str(frames[0]),
    cv2.IMREAD_GRAYSCALE
)

if previous_image is None:
    raise RuntimeError(
        "Could not load first frame."
    )

kp_previous, des_previous = (
    sift.detectAndCompute(
        previous_image,
        None
    )
)


print("=" * 72)
print("ARIA-S3D | BUNDLE ADJUSTMENT DATA PREPARATION")
print("=" * 72)

print(f"Frames available: {len(frames)}")
print(f"Initial features: {len(kp_previous)}")


# ============================================================
# PROCESS VIDEO
# ============================================================

for frame_id in range(1, len(frames)):

    current_image = cv2.imread(
        str(frames[frame_id]),
        cv2.IMREAD_GRAYSCALE
    )

    if current_image is None:
        continue

    kp_current, des_current = (
        sift.detectAndCompute(
            current_image,
            None
        )
    )

    if (
        des_previous is None
        or des_current is None
    ):
        continue


    # --------------------------------------------------------
    # FEATURE MATCHING
    # --------------------------------------------------------

    raw_matches = flann.knnMatch(
        des_previous,
        des_current,
        k=2
    )

    good_matches = []

    for pair in raw_matches:

        if len(pair) != 2:
            continue

        m, n = pair

        if m.distance < 0.7 * n.distance:
            good_matches.append(m)


    if len(good_matches) < 20:

        print(
            f"Frame {previous_frame_id:05d}"
            f" -> {frame_id:05d}"
            f" | insufficient matches"
        )

        continue


    pts_previous = np.float64([
        kp_previous[m.queryIdx].pt
        for m in good_matches
    ])

    pts_current = np.float64([
        kp_current[m.trainIdx].pt
        for m in good_matches
    ])


    # --------------------------------------------------------
    # ESSENTIAL MATRIX
    # --------------------------------------------------------

    E, essential_mask = (
        cv2.findEssentialMat(
            pts_previous,
            pts_current,
            K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.5
        )
    )

    if E is None:
        continue

    # OpenCV can occasionally return multiple candidates.
    if E.shape[0] > 3:
        E = E[:3, :]


    essential_mask = (
        essential_mask.ravel()
        .astype(bool)
    )

    pts_previous = (
        pts_previous[
            essential_mask
        ]
    )

    pts_current = (
        pts_current[
            essential_mask
        ]
    )

    if len(pts_previous) < 20:
        continue


    # --------------------------------------------------------
    # RELATIVE CAMERA POSE
    # --------------------------------------------------------

    _, R_relative, t_relative, pose_mask = (
        cv2.recoverPose(
            E,
            pts_previous,
            pts_current,
            K
        )
    )

    pose_mask = (
        pose_mask.ravel()
        .astype(bool)
    )

    pts_previous = (
        pts_previous[
            pose_mask
        ]
    )

    pts_current = (
        pts_current[
            pose_mask
        ]
    )

    if len(pts_previous) < 20:

        print(
            f"Frame {previous_frame_id:05d}"
            f" -> {frame_id:05d}"
            f" | weak pose"
        )

        continue


    # ========================================================
    # GLOBAL CAMERA POSE
    # ========================================================

    R_previous = camera_R[-1]
    t_previous = camera_t[-1]

    # recoverPose:
    #
    # X_current =
    # R_relative * X_previous + t_relative
    #
    # Therefore:
    #
    # R_current = R_rel * R_previous
    # t_current = R_rel * t_previous + t_rel

    R_current = (
        R_relative
        @
        R_previous
    )

    t_current = (
        R_relative
        @
        t_previous
        +
        t_relative
    )


    # ========================================================
    # GLOBAL PROJECTION MATRICES
    # ========================================================

    P_previous = (
        K
        @
        np.hstack(
            (
                R_previous,
                t_previous
            )
        )
    )

    P_current = (
        K
        @
        np.hstack(
            (
                R_current,
                t_current
            )
        )
    )


    # ========================================================
    # GLOBAL TRIANGULATION
    # ========================================================

    points_h = cv2.triangulatePoints(
        P_previous,
        P_current,
        pts_previous.T,
        pts_current.T
    )

    X_world = (
        points_h[:3]
        /
        points_h[3]
    ).T


    # --------------------------------------------------------
    # FINITE FILTER
    # --------------------------------------------------------

    finite = np.isfinite(
        X_world
    ).all(axis=1)

    X_world = X_world[finite]

    obs_previous = pts_previous[finite]
    obs_current = pts_current[finite]


    if len(X_world) == 0:
        continue


    # --------------------------------------------------------
    # CHEIRALITY CHECK
    # --------------------------------------------------------

    X_prev_camera = (
        R_previous
        @
        X_world.T
        +
        t_previous
    ).T

    X_curr_camera = (
        R_current
        @
        X_world.T
        +
        t_current
    ).T


    positive_depth = (
        (X_prev_camera[:, 2] > 0)
        &
        (X_curr_camera[:, 2] > 0)
    )


    X_world = (
        X_world[
            positive_depth
        ]
    )

    obs_previous = (
        obs_previous[
            positive_depth
        ]
    )

    obs_current = (
        obs_current[
            positive_depth
        ]
    )


    if len(X_world) == 0:
        continue


    # ========================================================
    # REPROJECTION FILTER
    # ========================================================

    def project_points(
        world_points,
        R,
        t
    ):

        camera_points = (
            R
            @
            world_points.T
            +
            t
        ).T

        projected = (
            K
            @
            camera_points.T
        ).T

        projected = (
            projected[:, :2]
            /
            projected[:, 2:3]
        )

        return projected


    proj_previous = project_points(
        X_world,
        R_previous,
        t_previous
    )

    proj_current = project_points(
        X_world,
        R_current,
        t_current
    )


    error_previous = np.linalg.norm(
        proj_previous
        -
        obs_previous,
        axis=1
    )

    error_current = np.linalg.norm(
        proj_current
        -
        obs_current,
        axis=1
    )


    reprojection_ok = (
        (error_previous < 3.0)
        &
        (error_current < 3.0)
    )


    X_world = (
        X_world[
            reprojection_ok
        ]
    )

    obs_previous = (
        obs_previous[
            reprojection_ok
        ]
    )

    obs_current = (
        obs_current[
            reprojection_ok
        ]
    )


    if len(X_world) < 10:

        print(
            f"Frame {previous_frame_id:05d}"
            f" -> {frame_id:05d}"
            f" | too few triangulated points"
        )

        continue


    # ========================================================
    # REGISTER NEW CAMERA
    # ========================================================

    previous_camera_index = (
        len(camera_R) - 1
    )

    current_camera_index = (
        len(camera_R)
    )

    camera_R.append(
        R_current
    )

    camera_t.append(
        t_current
    )

    frame_ids.append(
        frame_id
    )


    # ========================================================
    # STORE 3D POINTS + OBSERVATIONS
    # ========================================================

    for j in range(
        len(X_world)
    ):

        point_id = len(
            points_3d
        )

        points_3d.append(
            X_world[j]
        )


        # Previous-frame observation
        camera_indices.append(
            previous_camera_index
        )

        point_indices.append(
            point_id
        )

        points_2d.append(
            obs_previous[j]
        )


        # Current-frame observation
        camera_indices.append(
            current_camera_index
        )

        point_indices.append(
            point_id
        )

        points_2d.append(
            obs_current[j]
        )


    print(
        f"Frame {previous_frame_id:05d}"
        f" -> {frame_id:05d}"
        f" | matches {len(good_matches):4d}"
        f" | BA points {len(X_world):4d}"
    )


    # ========================================================
    # NEXT SUCCESSFUL CAMERA
    # ========================================================

    previous_frame_id = (
        frame_id
    )

    previous_image = (
        current_image
    )

    kp_previous = (
        kp_current
    )

    des_previous = (
        des_current
    )


# ============================================================
# CONVERT ARRAYS
# ============================================================

points_3d = np.asarray(
    points_3d,
    dtype=np.float64
)

camera_indices = np.asarray(
    camera_indices,
    dtype=np.int32
)

point_indices = np.asarray(
    point_indices,
    dtype=np.int32
)

points_2d = np.asarray(
    points_2d,
    dtype=np.float64
)

camera_R = np.asarray(
    camera_R,
    dtype=np.float64
)

camera_t = np.asarray(
    camera_t,
    dtype=np.float64
).reshape(-1, 3)

frame_ids = np.asarray(
    frame_ids,
    dtype=np.int32
)


# ============================================================
# CONVERT ROTATIONS → RODRIGUES
# ============================================================

camera_params = []

for R, t in zip(
    camera_R,
    camera_t
):

    rvec, _ = cv2.Rodrigues(
        R
    )

    camera_params.append(
        np.hstack(
            (
                rvec.ravel(),
                t.ravel()
            )
        )
    )


camera_params = np.asarray(
    camera_params,
    dtype=np.float64
)


# ============================================================
# SAVE BA DATA
# ============================================================

np.savez_compressed(
    OUTPUT_FILE,

    K=K,

    camera_params=(
        camera_params
    ),

    points_3d=(
        points_3d
    ),

    camera_indices=(
        camera_indices
    ),

    point_indices=(
        point_indices
    ),

    points_2d=(
        points_2d
    ),

    frame_ids=(
        frame_ids
    )
)


print()
print("=" * 72)
print("BA DATA PREPARATION COMPLETE")
print("=" * 72)

print(
    f"Cameras:       "
    f"{len(camera_params)}"
)

print(
    f"3D points:     "
    f"{len(points_3d)}"
)

print(
    f"Observations:  "
    f"{len(points_2d)}"
)

print(
    f"Saved to:      "
    f"{OUTPUT_FILE}"
)

print()
print(
    "Each 3D point currently has two image observations."
)

print(
    "Bundle Adjustment can now optimize "
    "camera poses and 3D geometry."
)

print("=" * 72)