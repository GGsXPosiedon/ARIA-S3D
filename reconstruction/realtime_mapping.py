import cv2
import numpy as np
import os
import time


# ============================================================
# ARIA-S3D | PHASE 4.4
# INCREMENTAL LIVE MAPPING
# ============================================================

print("=" * 70)
print("ARIA-S3D | PHASE 4.4")
print("INCREMENTAL LIVE MAPPING")
print("=" * 70)


# ============================================================
# CONFIGURATION
# ============================================================

CAMERA_ID = 0

IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720

FX = 1280.0
FY = 1280.0
CX = IMAGE_WIDTH / 2.0
CY = IMAGE_HEIGHT / 2.0

MIN_MATCHES = 25
MIN_TRIANGULATION_POINTS = 8

SIFT_FEATURES = 2500

RANSAC_THRESHOLD = 1.0

OUTPUT_DIR = "data/output"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# CAMERA INTRINSICS
# ============================================================

K = np.array(
    [
        [FX, 0, CX],
        [0, FY, CY],
        [0, 0, 1]
    ],
    dtype=np.float64
)


# ============================================================
# CAMERA DISTORTION
# ============================================================

DIST_COEFFS = np.zeros(
    (5, 1),
    dtype=np.float64
)


# ============================================================
# SIFT
# ============================================================

sift = cv2.SIFT_create(
    nfeatures=SIFT_FEATURES
)


# ============================================================
# FLANN MATCHER
# ============================================================

FLANN_INDEX_KDTREE = 1

index_params = {
    "algorithm": FLANN_INDEX_KDTREE,
    "trees": 5
}

search_params = {
    "checks": 50
}

flann = cv2.FlannBasedMatcher(
    index_params,
    search_params
)


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(CAMERA_ID)

cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    IMAGE_WIDTH
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    IMAGE_HEIGHT
)

if not cap.isOpened():

    print()
    print("ERROR: Could not open camera.")
    print("Check CAMERA_ID and camera permissions.")
    exit()


# ============================================================
# CAMERA INFORMATION
# ============================================================

actual_width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

actual_height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

print()
print(f"Camera ID        : {CAMERA_ID}")
print(
    f"Resolution       : "
    f"{actual_width} x {actual_height}"
)
print(f"SIFT features    : {SIFT_FEATURES}")
print(f"Minimum matches  : {MIN_MATCHES}")

print()
print("Controls:")
print("Q / ESC : Stop")
print("S       : Save current map")
print("R       : Reset map")

print()
print("=" * 70)
print("LIVE MAPPING STARTED")
print("=" * 70)


# ============================================================
# STATE
# ============================================================

previous_gray = None

previous_keypoints = None
previous_descriptors = None


# ------------------------------------------------------------
# GLOBAL CAMERA POSE
# ------------------------------------------------------------

R_global = np.eye(
    3,
    dtype=np.float64
)

t_global = np.zeros(
    (3, 1),
    dtype=np.float64
)


# ------------------------------------------------------------
# CAMERA TRAJECTORY
# ------------------------------------------------------------

camera_trajectory = []

camera_trajectory.append(
    t_global.flatten().copy()
)


# ------------------------------------------------------------
# GLOBAL POINT CLOUD
# ------------------------------------------------------------

global_points = []


# ============================================================
# STATISTICS
# ============================================================

frames_processed = 0
successful_frames = 0
failed_frames = 0

total_matches = 0
total_inliers = 0
total_points_added = 0

fps_values = []

mapping_start_time = time.time()


# ============================================================
# SAVE MAP FUNCTION
# ============================================================

def save_map():

    if len(global_points) > 0:

        points = np.asarray(
            global_points,
            dtype=np.float32
        )

        point_cloud_path = os.path.join(
            OUTPUT_DIR,
            "realtime_mapped_point_cloud.xyz"
        )

        np.savetxt(
            point_cloud_path,
            points,
            fmt="%.6f"
        )

    else:

        point_cloud_path = None


    trajectory_path = os.path.join(
        OUTPUT_DIR,
        "realtime_mapped_trajectory.txt"
    )

    with open(
        trajectory_path,
        "w"
    ) as f:

        for i, position in enumerate(
            camera_trajectory
        ):

            f.write(
                f"{i} "
                f"{position[0]:.6f} "
                f"{position[1]:.6f} "
                f"{position[2]:.6f}\n"
            )


    print()
    print("-" * 70)
    print("MAP SAVED")
    print("-" * 70)

    if point_cloud_path:

        print(
            f"Point cloud : "
            f"{point_cloud_path}"
        )

    print(
        f"Trajectory  : "
        f"{trajectory_path}"
    )

    print(
        f"3D points   : "
        f"{len(global_points)}"
    )

    print("-" * 70)


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    frame_start = time.time()

    ret, frame = cap.read()

    if not ret:

        print(
            "WARNING: Failed to capture frame."
        )

        failed_frames += 1
        continue


    frames_processed += 1


    # ========================================================
    # RESIZE IF NECESSARY
    # ========================================================

    if (
        frame.shape[1] != IMAGE_WIDTH
        or frame.shape[0] != IMAGE_HEIGHT
    ):

        frame = cv2.resize(
            frame,
            (
                IMAGE_WIDTH,
                IMAGE_HEIGHT
            )
        )


    # ========================================================
    # GRAYSCALE
    # ========================================================

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )


    # ========================================================
    # FEATURE DETECTION
    # ========================================================

    keypoints, descriptors = (
        sift.detectAndCompute(
            gray,
            None
        )
    )


    if descriptors is None:

        failed_frames += 1

        cv2.putText(
            frame,
            "NO FEATURES",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2
        )

        cv2.imshow(
            "ARIA-S3D | Live Mapping",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if (
            key == ord("q")
            or key == 27
        ):
            break

        continue


    # ========================================================
    # FIRST FRAME
    # ========================================================

    if (
        previous_gray is None
        or previous_descriptors is None
    ):

        previous_gray = gray.copy()

        previous_keypoints = keypoints

        previous_descriptors = descriptors

        cv2.putText(
            frame,
            "INITIALIZING MAP",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2
        )

        cv2.imshow(
            "ARIA-S3D | Live Mapping",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if (
            key == ord("q")
            or key == 27
        ):
            break

        continue


    # ========================================================
    # FEATURE MATCHING
    # ========================================================

    try:

        matches = flann.knnMatch(
            previous_descriptors,
            descriptors,
            k=2
        )

    except cv2.error:

        failed_frames += 1

        previous_gray = gray.copy()
        previous_keypoints = keypoints
        previous_descriptors = descriptors

        continue


    good_matches = []

    for pair in matches:

        if len(pair) != 2:
            continue

        m, n = pair

        if m.distance < 0.70 * n.distance:

            good_matches.append(m)


    match_count = len(
        good_matches
    )

    total_matches += match_count


    # ========================================================
    # NOT ENOUGH MATCHES
    # ========================================================

    if match_count < MIN_MATCHES:

        failed_frames += 1

        cv2.putText(
            frame,
            f"TRACKING WEAK | Matches: {match_count}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

        cv2.imshow(
            "ARIA-S3D | Live Mapping",
            frame
        )

        previous_gray = gray.copy()

        previous_keypoints = keypoints

        previous_descriptors = descriptors

        key = cv2.waitKey(1) & 0xFF

        if (
            key == ord("q")
            or key == 27
        ):
            break

        continue


    # ========================================================
    # MATCHED IMAGE POINTS
    # ========================================================

    pts_previous = np.float32(
        [
            previous_keypoints[
                m.queryIdx
            ].pt
            for m in good_matches
        ]
    )

    pts_current = np.float32(
        [
            keypoints[
                m.trainIdx
            ].pt
            for m in good_matches
        ]
    )


    # ========================================================
    # ESSENTIAL MATRIX
    # ========================================================

    E, mask = cv2.findEssentialMat(
        pts_previous,
        pts_current,
        K,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=RANSAC_THRESHOLD
    )


    if E is None or mask is None:

        failed_frames += 1

        previous_gray = gray.copy()
        previous_keypoints = keypoints
        previous_descriptors = descriptors

        continue


    # ========================================================
    # ESSENTIAL MATRIX INLIERS
    # ========================================================

    mask = mask.ravel().astype(bool)

    if len(mask) != len(pts_previous):

        failed_frames += 1

        previous_gray = gray.copy()
        previous_keypoints = keypoints
        previous_descriptors = descriptors

        continue


    pts_previous_inliers = (
        pts_previous[mask]
    )

    pts_current_inliers = (
        pts_current[mask]
    )


    inlier_count = len(
        pts_previous_inliers
    )

    total_inliers += inlier_count


    if inlier_count < 8:

        failed_frames += 1

        previous_gray = gray.copy()
        previous_keypoints = keypoints
        previous_descriptors = descriptors

        continue


    # ========================================================
    # RECOVER RELATIVE CAMERA POSE
    # ========================================================

    try:

        pose_count, R_relative, t_relative, pose_mask = (
            cv2.recoverPose(
                E,
                pts_previous_inliers,
                pts_current_inliers,
                K
            )
        )

    except cv2.error:

        failed_frames += 1

        previous_gray = gray.copy()
        previous_keypoints = keypoints
        previous_descriptors = descriptors

        continue


    pose_mask = pose_mask.ravel().astype(bool)


    if len(pose_mask) != len(
        pts_previous_inliers
    ):

        failed_frames += 1

        previous_gray = gray.copy()
        previous_keypoints = keypoints
        previous_descriptors = descriptors

        continue


    pts_previous_pose = (
        pts_previous_inliers[
            pose_mask
        ]
    )

    pts_current_pose = (
        pts_current_inliers[
            pose_mask
        ]
    )


    pose_inliers = len(
        pts_previous_pose
    )


    if pose_inliers < 8:

        failed_frames += 1

        previous_gray = gray.copy()
        previous_keypoints = keypoints
        previous_descriptors = descriptors

        continue


    # ========================================================
    # UPDATE GLOBAL CAMERA POSE
    #
    # IMPORTANT:
    # Monocular translation has
    # arbitrary scale.
    # ========================================================

    t_global = (
        t_global
        + R_global @ t_relative
    )

    R_global = (
        R_relative @ R_global
    )


    camera_trajectory.append(
        t_global.flatten().copy()
    )


    # ========================================================
    # TRIANGULATION
    # ========================================================

    P_previous = K @ np.hstack(
        (
            np.eye(3),
            np.zeros(
                (3, 1)
            )
        )
    )


    P_current = K @ np.hstack(
        (
            R_relative,
            t_relative
        )
    )


    try:

        points_4d = cv2.triangulatePoints(
            P_previous,
            P_current,
            pts_previous_pose.T,
            pts_current_pose.T
        )

    except cv2.error:

        failed_frames += 1

        previous_gray = gray.copy()
        previous_keypoints = keypoints
        previous_descriptors = descriptors

        continue


    # ========================================================
    # HOMOGENEOUS -> 3D
    # ========================================================

    w = points_4d[3]

    valid_w = np.abs(w) > 1e-8


    if not np.any(valid_w):

        failed_frames += 1

        previous_gray = gray.copy()
        previous_keypoints = keypoints
        previous_descriptors = descriptors

        continue


    points_3d = (
        points_4d[:3]
        / np.where(
            valid_w,
            w,
            1.0
        )
    ).T


    # ========================================================
    # VALIDATE 3D POINTS
    # ========================================================

    valid = np.isfinite(
        points_3d
    ).all(axis=1)


    points_3d = points_3d[
        valid
    ]


    if len(points_3d) == 0:

        failed_frames += 1

        previous_gray = gray.copy()
        previous_keypoints = keypoints
        previous_descriptors = descriptors

        continue


    # --------------------------------------------------------
    # REMOVE EXTREME DEPTH VALUES
    # --------------------------------------------------------

    depth_valid = (
        points_3d[:, 2] > 0
    ) & (
        points_3d[:, 2] < 500.0
    )


    points_3d = points_3d[
        depth_valid
    ]


    # ========================================================
    # ADD POINTS TO GLOBAL MAP
    # ========================================================

    points_added = len(
        points_3d
    )


    if points_added > 0:

        global_points.extend(
            points_3d.tolist()
        )

        total_points_added += (
            points_added
        )


    successful_frames += 1


    # ========================================================
    # FPS
    # ========================================================

    frame_time = (
        time.time()
        - frame_start
    )

    if frame_time > 0:

        fps = 1.0 / frame_time

        fps_values.append(
            fps
        )

    else:

        fps = 0.0


    # ========================================================
    # DRAW TRACKING INFORMATION
    # ========================================================

    cv2.putText(
        frame,
        f"MAP POINTS: {len(global_points)}",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"MATCHES: {match_count}",
        (30, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"INLIERS: {pose_inliers}",
        (30, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"FPS: {fps:.2f}",
        (30, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "LIVE MAPPING",
        (30, 165),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )


    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.imshow(
        "ARIA-S3D | Live Mapping",
        frame
    )


    # ========================================================
    # UPDATE PREVIOUS FRAME
    # ========================================================

    previous_gray = gray.copy()

    previous_keypoints = keypoints

    previous_descriptors = descriptors


    # ========================================================
    # KEYBOARD
    # ========================================================

    key = cv2.waitKey(1) & 0xFF


    if key == ord("s"):

        save_map()


    elif key == ord("r"):

        print()
        print("RESETTING LIVE MAP...")

        global_points = []

        camera_trajectory = []

        R_global = np.eye(
            3,
            dtype=np.float64
        )

        t_global = np.zeros(
            (3, 1),
            dtype=np.float64
        )

        camera_trajectory.append(
            t_global.flatten().copy()
        )

        print("Map reset.")


    elif (
        key == ord("q")
        or key == 27
    ):

        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()


# ============================================================
# SAVE FINAL MAP
# ============================================================

save_map()


# ============================================================
# FINAL STATISTICS
# ============================================================

elapsed_time = (
    time.time()
    - mapping_start_time
)


average_fps = (
    np.mean(fps_values)
    if fps_values
    else 0.0
)


average_matches = (
    total_matches / frames_processed
    if frames_processed > 0
    else 0.0
)


average_inliers = (
    total_inliers / successful_frames
    if successful_frames > 0
    else 0.0
)


mapping_success_rate = (
    successful_frames
    / frames_processed
    * 100.0
    if frames_processed > 0
    else 0.0
)


# ============================================================
# SAVE STATISTICS
# ============================================================

stats_path = os.path.join(
    OUTPUT_DIR,
    "realtime_mapping_stats.txt"
)


with open(
    stats_path,
    "w"
) as f:

    f.write(
        "ARIA-S3D | PHASE 4.4\n"
    )

    f.write(
        "INCREMENTAL LIVE MAPPING\n"
    )

    f.write(
        "========================================\n"
    )

    f.write(
        f"Frames processed: "
        f"{frames_processed}\n"
    )

    f.write(
        f"Successful mapping frames: "
        f"{successful_frames}\n"
    )

    f.write(
        f"Failed frames: "
        f"{failed_frames}\n"
    )

    f.write(
        f"Mapping success rate: "
        f"{mapping_success_rate:.2f}%\n"
    )

    f.write(
        f"Average matches: "
        f"{average_matches:.2f}\n"
    )

    f.write(
        f"Average pose inliers: "
        f"{average_inliers:.2f}\n"
    )

    f.write(
        f"Total mapped 3D points: "
        f"{len(global_points)}\n"
    )

    f.write(
        f"Average FPS: "
        f"{average_fps:.2f}\n"
    )

    f.write(
        f"Runtime seconds: "
        f"{elapsed_time:.2f}\n"
    )

    f.write(
        "\n"
    )

    f.write(
        "IMPORTANT:\n"
    )

    f.write(
        "This is a monocular reconstruction.\n"
    )

    f.write(
        "Absolute metric scale remains arbitrary.\n"
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("=" * 70)
print("ARIA-S3D | PHASE 4.4 COMPLETE")
print("=" * 70)

print(
    f"Frames processed        : "
    f"{frames_processed}"
)

print(
    f"Successful mapping      : "
    f"{successful_frames}"
)

print(
    f"Failed frames           : "
    f"{failed_frames}"
)

print(
    f"Mapping success rate    : "
    f"{mapping_success_rate:.2f}%"
)

print(
    f"Average matches         : "
    f"{average_matches:.2f}"
)

print(
    f"Average pose inliers    : "
    f"{average_inliers:.2f}"
)

print(
    f"Total mapped 3D points  : "
    f"{len(global_points)}"
)

print(
    f"Average FPS             : "
    f"{average_fps:.2f}"
)

print()
print("Generated outputs:")

print(
    "  data/output/"
    "realtime_mapped_point_cloud.xyz"
)

print(
    "  data/output/"
    "realtime_mapped_trajectory.txt"
)

print(
    "  data/output/"
    "realtime_mapping_stats.txt"
)

print()
print("IMPORTANT:")
print(
    "This remains a monocular reconstruction."
)

print(
    "Absolute metric scale remains arbitrary."
)

print()
print(
    "NEXT STEP:"
)

print(
    "Phase 4.5 - Live 3D visualization"
)

print("=" * 70)