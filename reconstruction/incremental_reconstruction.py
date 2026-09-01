import cv2
import numpy as np
import os
import glob


# ============================================================
# ARIA-S3D | INCREMENTAL 3D RECONSTRUCTION
# ============================================================

FRAMES_DIR = "data/quality_frames"
OUTPUT_DIR = "data/output"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------------------------------------------
# CAMERA INTRINSICS
# ------------------------------------------------------------

IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720

fx = 1280.0
fy = 1280.0
cx = IMAGE_WIDTH / 2
cy = IMAGE_HEIGHT / 2

K = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1]
], dtype=np.float64)


# ------------------------------------------------------------
# LOAD FRAMES
# ------------------------------------------------------------

frame_paths = sorted(
    glob.glob(os.path.join(FRAMES_DIR, "*.jpg"))
)

if len(frame_paths) < 2:
    print("ERROR: Need at least 2 frames.")
    exit()

print("=" * 60)
print("ARIA-S3D | INCREMENTAL RECONSTRUCTION")
print("=" * 60)

print(f"Frames found: {len(frame_paths)}")
print(f"Image size: {IMAGE_WIDTH} x {IMAGE_HEIGHT}")


# ------------------------------------------------------------
# SIFT
# ------------------------------------------------------------

sift = cv2.SIFT_create(
    nfeatures=3000
)


# ------------------------------------------------------------
# FLANN MATCHER
# ------------------------------------------------------------

FLANN_INDEX_KDTREE = 1

index_params = dict(
    algorithm=FLANN_INDEX_KDTREE,
    trees=5
)

search_params = dict(
    checks=50
)

flann = cv2.FlannBasedMatcher(
    index_params,
    search_params
)


# ------------------------------------------------------------
# CAMERA POSE
#
# World coordinate system:
# Frame 0 = origin
# ------------------------------------------------------------

R_global = np.eye(3)
t_global = np.zeros((3, 1))

camera_poses = []

camera_poses.append(
    np.hstack((R_global, t_global))
)


# ------------------------------------------------------------
# GLOBAL POINT CLOUD
# ------------------------------------------------------------

global_points = []


# ------------------------------------------------------------
# FIRST FRAME
# ------------------------------------------------------------

previous_image = cv2.imread(
    frame_paths[0],
    cv2.IMREAD_GRAYSCALE
)

if previous_image is None:
    print("ERROR: Could not read first frame.")
    exit()

kp_previous, des_previous = sift.detectAndCompute(
    previous_image,
    None
)

print()
print(f"Initial frame: {os.path.basename(frame_paths[0])}")
print(f"Initial features: {len(kp_previous)}")


# ------------------------------------------------------------
# PROCESS EACH FRAME
# ------------------------------------------------------------

for i in range(1, len(frame_paths)):

    current_path = frame_paths[i]

    current_image = cv2.imread(
        current_path,
        cv2.IMREAD_GRAYSCALE
    )

    if current_image is None:
        print(f"WARNING: Could not read {current_path}")
        continue

    kp_current, des_current = sift.detectAndCompute(
        current_image,
        None
    )

    if des_current is None or des_previous is None:
        print(f"Frame {i}: No descriptors.")
        continue


    # --------------------------------------------------------
    # FEATURE MATCHING
    # --------------------------------------------------------

    matches = flann.knnMatch(
        des_previous,
        des_current,
        k=2
    )

    good_matches = []

    for pair in matches:

        if len(pair) != 2:
            continue

        m, n = pair

        if m.distance < 0.7 * n.distance:
            good_matches.append(m)


    print(
        f"\nFrame {i-1:05d} -> Frame {i:05d} | "
        f"Matches: {len(good_matches)}"
    )


    if len(good_matches) < 8:
        print("Not enough matches. Skipping.")
        continue


    # --------------------------------------------------------
    # MATCHED POINTS
    # --------------------------------------------------------

    pts_previous = np.float32([
        kp_previous[m.queryIdx].pt
        for m in good_matches
    ])

    pts_current = np.float32([
        kp_current[m.trainIdx].pt
        for m in good_matches
    ])


    # --------------------------------------------------------
    # ESSENTIAL MATRIX
    # --------------------------------------------------------

    E, mask = cv2.findEssentialMat(
        pts_previous,
        pts_current,
        K,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=1.0
    )

    if E is None:
        print("Essential matrix estimation failed.")
        continue


    # --------------------------------------------------------
    # KEEP ONLY INLIERS
    # --------------------------------------------------------

    mask = mask.ravel().astype(bool)

    pts_previous_inliers = pts_previous[mask]
    pts_current_inliers = pts_current[mask]

    if len(pts_previous_inliers) < 8:
        print("Not enough geometric inliers.")
        continue


    # --------------------------------------------------------
    # RECOVER RELATIVE CAMERA POSE
    # --------------------------------------------------------

    _, R_relative, t_relative, pose_mask = cv2.recoverPose(
        E,
        pts_previous_inliers,
        pts_current_inliers,
        K
    )


    pose_mask = pose_mask.ravel().astype(bool)

    pts_previous_pose = pts_previous_inliers[pose_mask]
    pts_current_pose = pts_current_inliers[pose_mask]


    if len(pts_previous_pose) < 8:
        print("Not enough pose inliers.")
        continue


    # --------------------------------------------------------
    # ACCUMULATE CAMERA POSE
    #
    # NOTE:
    # Monocular translation has arbitrary scale.
    # --------------------------------------------------------

    t_global = t_global + R_global @ t_relative

    R_global = R_relative @ R_global

    camera_poses.append(
        np.hstack((R_global.copy(), t_global.copy()))
    )


    # --------------------------------------------------------
    # TRIANGULATE CURRENT FRAME
    # --------------------------------------------------------

    P_previous = K @ np.hstack((
        np.eye(3),
        np.zeros((3, 1))
    ))

    P_current = K @ np.hstack((
        R_relative,
        t_relative
    ))


    points_4d = cv2.triangulatePoints(
        P_previous,
        P_current,
        pts_previous_pose.T,
        pts_current_pose.T
    )


    # --------------------------------------------------------
    # CONVERT HOMOGENEOUS → 3D
    # --------------------------------------------------------

    points_3d = (
        points_4d[:3] /
        points_4d[3]
    ).T


    # --------------------------------------------------------
    # KEEP VALID 3D POINTS
    # --------------------------------------------------------

    valid = np.isfinite(points_3d).all(axis=1)

    points_3d = points_3d[valid]

    positive_depth = points_3d[:, 2] > 0

    points_3d = points_3d[positive_depth]


    # --------------------------------------------------------
    # ADD TO GLOBAL CLOUD
    # --------------------------------------------------------

    if len(points_3d) > 0:

        global_points.extend(
            points_3d.tolist()
        )


    print(
        f"Pose inliers: {len(pts_previous_pose)} | "
        f"3D points added: {len(points_3d)}"
    )

    print(
        "Camera position:",
        t_global.ravel()
    )


    # --------------------------------------------------------
    # PREPARE NEXT FRAME
    # --------------------------------------------------------

    previous_image = current_image

    kp_previous = kp_current
    des_previous = des_current


# ============================================================
# SAVE RESULTS
# ============================================================

print()
print("=" * 60)
print("INCREMENTAL RECONSTRUCTION COMPLETE")
print("=" * 60)

print(f"Camera poses estimated: {len(camera_poses)}")
print(f"Total 3D points: {len(global_points)}")


# ------------------------------------------------------------
# SAVE POINT CLOUD
# ------------------------------------------------------------

if len(global_points) > 0:

    global_points = np.array(
        global_points,
        dtype=np.float32
    )

    point_cloud_path = os.path.join(
        OUTPUT_DIR,
        "incremental_point_cloud.xyz"
    )

    np.savetxt(
        point_cloud_path,
        global_points,
        fmt="%.6f"
    )

    print(
        f"Point cloud saved to:\n"
        f"{point_cloud_path}"
    )


# ------------------------------------------------------------
# SAVE CAMERA TRAJECTORY
# ------------------------------------------------------------

trajectory_path = os.path.join(
    OUTPUT_DIR,
    "camera_trajectory.txt"
)

with open(trajectory_path, "w") as f:

    for i, pose in enumerate(camera_poses):

        R = pose[:, :3]
        t = pose[:, 3]

        f.write(
            f"{i} "
            f"{t[0]:.6f} "
            f"{t[1]:.6f} "
            f"{t[2]:.6f}\n"
        )


print(
    f"Camera trajectory saved to:\n"
    f"{trajectory_path}"
)

print()
print("IMPORTANT:")
print("This is a monocular reconstruction.")
print("Translation scale is arbitrary.")
print("The next stage will improve global consistency.")