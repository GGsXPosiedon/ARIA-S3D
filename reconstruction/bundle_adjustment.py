import cv2
import numpy as np
from pathlib import Path

from scipy.optimize import least_squares
from scipy.sparse import lil_matrix


# ============================================================
# ARIA-S3D | PHASE 3
# TRUE SPARSE BUNDLE ADJUSTMENT
# ============================================================

INPUT_FILE = Path(
    "data/output/ba_data.npz"
)

OUTPUT_DIR = Path(
    "data/output"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

if not INPUT_FILE.exists():

    raise FileNotFoundError(
        "\nBA data not found.\n"
        "Run:\n"
        "python reconstruction/"
        "prepare_ba_data.py"
    )


data = np.load(
    INPUT_FILE
)

K = data["K"]

camera_params = (
    data["camera_params"]
    .astype(np.float64)
)

points_3d = (
    data["points_3d"]
    .astype(np.float64)
)

camera_indices = (
    data["camera_indices"]
    .astype(np.int32)
)

point_indices = (
    data["point_indices"]
    .astype(np.int32)
)

points_2d = (
    data["points_2d"]
    .astype(np.float64)
)

frame_ids = (
    data["frame_ids"]
    .astype(np.int32)
)


n_cameras = len(
    camera_params
)

n_points = len(
    points_3d
)

n_observations = len(
    points_2d
)


print("=" * 72)
print("ARIA-S3D | SPARSE BUNDLE ADJUSTMENT")
print("=" * 72)

print(
    f"Cameras:       {n_cameras}"
)

print(
    f"3D points:     {n_points}"
)

print(
    f"Observations:  {n_observations}"
)


if n_cameras < 3:

    raise RuntimeError(
        "Need at least 3 cameras "
        "for this BA configuration."
    )


# ============================================================
# FIX GAUGE FREEDOM
#
# Camera 0 fixes world origin/orientation.
# Camera 1 also stays fixed to establish monocular scale.
#
# Cameras 2...N are optimized.
# ============================================================

FIXED_CAMERAS = 2

variable_camera_count = (
    n_cameras
    -
    FIXED_CAMERAS
)


# ============================================================
# VECTORIZED RODRIGUES ROTATION
# ============================================================

def rotate_points(
    points,
    rot_vecs
):

    theta = np.linalg.norm(
        rot_vecs,
        axis=1
    )[:, np.newaxis]


    with np.errstate(
        invalid="ignore",
        divide="ignore"
    ):

        v = np.divide(
            rot_vecs,
            theta,
            out=np.zeros_like(
                rot_vecs
            ),
            where=theta != 0
        )


    dot = np.sum(
        points * v,
        axis=1
    )[:, np.newaxis]


    cos_theta = np.cos(
        theta
    )

    sin_theta = np.sin(
        theta
    )


    rotated = (

        cos_theta * points

        +

        sin_theta
        *
        np.cross(
            v,
            points
        )

        +

        dot
        *
        (1.0 - cos_theta)
        *
        v
    )


    zero_rotation = (
        theta[:, 0]
        <
        1e-12
    )

    rotated[
        zero_rotation
    ] = points[
        zero_rotation
    ]


    return rotated


# ============================================================
# PROJECT 3D → IMAGE
# ============================================================

def project(
    points,
    cameras
):

    rotated = rotate_points(
        points,
        cameras[:, :3]
    )

    camera_points = (
        rotated
        +
        cameras[:, 3:6]
    )


    z = camera_points[:, 2]

    # Prevent division by zero
    z_safe = np.where(
        np.abs(z) < 1e-8,
        1e-8,
        z
    )


    u = (
        K[0, 0]
        *
        camera_points[:, 0]
        /
        z_safe
        +
        K[0, 2]
    )


    v = (
        K[1, 1]
        *
        camera_points[:, 1]
        /
        z_safe
        +
        K[1, 2]
    )


    return np.column_stack(
        (
            u,
            v
        )
    )


# ============================================================
# PACK / UNPACK OPTIMIZATION VARIABLES
# ============================================================

fixed_camera_params = (
    camera_params[
        :FIXED_CAMERAS
    ].copy()
)


def unpack_parameters(x):

    camera_variable_size = (
        variable_camera_count
        *
        6
    )


    variable_cameras = (
        x[
            :camera_variable_size
        ]
        .reshape(
            variable_camera_count,
            6
        )
    )


    optimized_points = (
        x[
            camera_variable_size:
        ]
        .reshape(
            n_points,
            3
        )
    )


    full_cameras = np.vstack(
        (
            fixed_camera_params,
            variable_cameras
        )
    )


    return (
        full_cameras,
        optimized_points
    )


# ============================================================
# REPROJECTION RESIDUALS
# ============================================================

def residual_function(x):

    cameras, points = (
        unpack_parameters(x)
    )


    projected = project(

        points[
            point_indices
        ],

        cameras[
            camera_indices
        ]
    )


    residuals = (
        projected
        -
        points_2d
    )


    return residuals.ravel()


# ============================================================
# SPARSE JACOBIAN STRUCTURE
# ============================================================

def build_sparsity():

    m = (
        n_observations
        *
        2
    )

    n = (

        variable_camera_count
        *
        6

        +

        n_points
        *
        3
    )


    A = lil_matrix(
        (m, n),
        dtype=int
    )


    observation_ids = np.arange(
        n_observations
    )


    # --------------------------------------------------------
    # CAMERA DEPENDENCIES
    # --------------------------------------------------------

    variable_mask = (
        camera_indices
        >=
        FIXED_CAMERAS
    )


    variable_obs = (
        observation_ids[
            variable_mask
        ]
    )


    variable_camera_indices = (

        camera_indices[
            variable_mask
        ]

        -

        FIXED_CAMERAS
    )


    for parameter in range(6):

        A[
            2 * variable_obs,
            variable_camera_indices * 6
            + parameter
        ] = 1

        A[
            2 * variable_obs + 1,
            variable_camera_indices * 6
            + parameter
        ] = 1


    # --------------------------------------------------------
    # POINT DEPENDENCIES
    # --------------------------------------------------------

    point_offset = (
        variable_camera_count
        *
        6
    )


    for parameter in range(3):

        A[
            2 * observation_ids,
            point_offset
            +
            point_indices * 3
            +
            parameter
        ] = 1

        A[
            2 * observation_ids + 1,
            point_offset
            +
            point_indices * 3
            +
            parameter
        ] = 1


    return A


# ============================================================
# INITIAL PARAMETER VECTOR
# ============================================================

x0 = np.hstack(
    (

        camera_params[
            FIXED_CAMERAS:
        ].ravel(),

        points_3d.ravel()
    )
)


# ============================================================
# INITIAL REPROJECTION ERROR
# ============================================================

initial_residuals = (
    residual_function(
        x0
    )
)


initial_rmse = np.sqrt(
    np.mean(
        initial_residuals ** 2
    )
)


print()
print(
    f"Initial reprojection RMSE: "
    f"{initial_rmse:.4f} px"
)


# ============================================================
# BUILD SPARSITY MATRIX
# ============================================================

print(
    "\nBuilding sparse "
    "Bundle Adjustment problem..."
)

jac_sparsity = (
    build_sparsity()
)


print(
    "Optimization variables:",
    len(x0)
)

print(
    "Residual values:",
    len(initial_residuals)
)


# ============================================================
# RUN BUNDLE ADJUSTMENT
# ============================================================

print()
print("=" * 72)
print("RUNNING BUNDLE ADJUSTMENT")
print("=" * 72)

result = least_squares(

    residual_function,

    x0,

    jac_sparsity=(
        jac_sparsity
    ),

    method="trf",

    x_scale="jac",

    loss="soft_l1",

    f_scale=1.0,

    ftol=1e-5,

    xtol=1e-5,

    gtol=1e-5,

    max_nfev=100,

    verbose=2
)


# ============================================================
# RESULTS
# ============================================================

optimized_cameras, optimized_points = (
    unpack_parameters(
        result.x
    )
)


final_residuals = (
    residual_function(
        result.x
    )
)


final_rmse = np.sqrt(
    np.mean(
        final_residuals ** 2
    )
)


improvement = (
    (
        initial_rmse
        -
        final_rmse
    )
    /
    max(
        initial_rmse,
        1e-12
    )
    *
    100.0
)


print()
print("=" * 72)
print("BUNDLE ADJUSTMENT COMPLETE")
print("=" * 72)

print(
    f"Initial RMSE: "
    f"{initial_rmse:.4f} px"
)

print(
    f"Final RMSE:   "
    f"{final_rmse:.4f} px"
)

print(
    f"Improvement:  "
    f"{improvement:.2f}%"
)

print(
    f"Iterations:   "
    f"{result.nfev}"
)

print(
    f"Success:      "
    f"{result.success}"
)

print(
    f"Message:      "
    f"{result.message}"
)


# ============================================================
# SAVE OPTIMIZED POINT CLOUD
# ============================================================

point_cloud_path = (
    OUTPUT_DIR
    /
    "bundle_adjusted_point_cloud.xyz"
)


np.savetxt(
    point_cloud_path,
    optimized_points,
    fmt="%.8f"
)


# ============================================================
# SAVE CAMERA DATA
# ============================================================

camera_npz_path = (
    OUTPUT_DIR
    /
    "bundle_adjusted_cameras.npz"
)


np.savez_compressed(

    camera_npz_path,

    camera_params=(
        optimized_cameras
    ),

    frame_ids=(
        frame_ids
    ),

    K=K
)


# ============================================================
# SAVE CAMERA CENTERS
# ============================================================

trajectory_path = (
    OUTPUT_DIR
    /
    "bundle_adjusted_trajectory.txt"
)


with open(
    trajectory_path,
    "w"
) as f:


    f.write(
        "# frame_id x y z\n"
    )


    for camera_index, params in enumerate(
        optimized_cameras
    ):


        rvec = params[:3]

        t = (
            params[3:6]
            .reshape(3, 1)
        )


        R, _ = cv2.Rodrigues(
            rvec
        )


        # Camera center in world coordinates:
        #
        # C = -R^T * t

        center = (
            -R.T
            @
            t
        ).ravel()


        f.write(

            f"{frame_ids[camera_index]} "

            f"{center[0]:.8f} "

            f"{center[1]:.8f} "

            f"{center[2]:.8f}\n"
        )


# ============================================================
# SAVE BA STATISTICS
# ============================================================

stats_path = (
    OUTPUT_DIR
    /
    "bundle_adjustment_stats.txt"
)


with open(
    stats_path,
    "w"
) as f:

    f.write(
        "ARIA-S3D Bundle Adjustment\n"
    )

    f.write(
        "==========================\n"
    )

    f.write(
        f"Cameras: {n_cameras}\n"
    )

    f.write(
        f"3D points: {n_points}\n"
    )

    f.write(
        f"Observations: "
        f"{n_observations}\n"
    )

    f.write(
        f"Initial RMSE: "
        f"{initial_rmse:.6f}\n"
    )

    f.write(
        f"Final RMSE: "
        f"{final_rmse:.6f}\n"
    )

    f.write(
        f"Improvement: "
        f"{improvement:.4f}%\n"
    )

    f.write(
        f"Optimization success: "
        f"{result.success}\n"
    )


print()
print("Saved outputs:")

print(
    point_cloud_path
)

print(
    camera_npz_path
)

print(
    trajectory_path
)

print(
    stats_path
)

print()
print(
    "Monocular scale remains arbitrary."
)

print("=" * 72)