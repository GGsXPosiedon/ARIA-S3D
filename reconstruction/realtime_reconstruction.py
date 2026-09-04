import cv2
import numpy as np
import os
import time
from pathlib import Path
import matplotlib.pyplot as plt


# ============================================================
# ARIA-S3D | PHASE 4
# REAL-TIME MONOCULAR 3D RECONSTRUCTION
# ============================================================

OUTPUT_DIR = Path("data/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

POINT_CLOUD_PATH = OUTPUT_DIR / "realtime_point_cloud.xyz"
TRAJECTORY_PATH = OUTPUT_DIR / "realtime_camera_trajectory.txt"


# ============================================================
# CAMERA CONFIGURATION
# ============================================================

CAMERA_ID = 0

IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720

# Approximate camera intrinsics.
# These should eventually be replaced by real calibration.
fx = 1280.0
fy = 1280.0
cx = IMAGE_WIDTH / 2.0
cy = IMAGE_HEIGHT / 2.0

K = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1]
], dtype=np.float64)


# ============================================================
# REAL-TIME SETTINGS
# ============================================================

SIFT_FEATURES = 2500

RATIO_TEST = 0.70

MIN_MATCHES = 30

RANSAC_THRESHOLD = 1.0

# Add points every N frames
TRIANGULATION_INTERVAL = 2

# Maximum points retained in live map
MAX_MAP_POINTS = 50000


# ============================================================
# ARIA-S3D HEADER
# ============================================================

print("=" * 70)
print("ARIA-S3D | PHASE 4")
print("REAL-TIME MONOCULAR 3D RECONSTRUCTION")
print("=" * 70)

print()
print("Initializing camera...")
print(f"Resolution: {IMAGE_WIDTH} x {IMAGE_HEIGHT}")
print(f"SIFT features: {SIFT_FEATURES}")
print()


# ============================================================
# CAMERA INITIALIZATION
# ============================================================

camera = cv2.VideoCapture(CAMERA_ID)

camera.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    IMAGE_WIDTH
)

camera.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    IMAGE_HEIGHT
)

if not camera.isOpened():
    raise RuntimeError(
        "ERROR: Could not open camera."
    )


# ============================================================
# SIFT FEATURE DETECTOR
# ============================================================

sift = cv2.SIFT_create(
    nfeatures=SIFT_FEATURES
)


# ============================================================
# FLANN FEATURE MATCHER
# ============================================================

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


# ============================================================
# CAMERA STATE
#
# World coordinate system:
# First frame = origin
# ============================================================

R_world = np.eye(3, dtype=np.float64)

t_world = np.zeros(
    (3, 1),
    dtype=np.float64
)


# ============================================================
# GLOBAL MAP
# ============================================================

global_points = []

camera_centers = [
    np.zeros(3)
]


# ============================================================
# FIRST FRAME
# ============================================================

ret, frame = camera.read()

if not ret:
    camera.release()

    raise RuntimeError(
        "ERROR: Could not read first camera frame."
    )


# Resize if camera does not return requested resolution
frame = cv2.resize(
    frame,
    (IMAGE_WIDTH, IMAGE_HEIGHT)
)

previous_gray = cv2.cvtColor(
    frame,
    cv2.COLOR_BGR2GRAY
)

kp_previous, des_previous = sift.detectAndCompute(
    previous_gray,
    None
)

if des_previous is None:
    camera.release()

    raise RuntimeError(
        "ERROR: Could not detect features in first frame."
    )


print(
    f"Initial features detected: "
    f"{len(kp_previous)}"
)

print()
print("Real-time reconstruction started.")
print()
print("Controls:")
print("  Q / ESC : Stop reconstruction")
print("  S       : Save current point cloud")
print()


# ============================================================
# LIVE 3D VIEW
# ============================================================

plt.ion()

fig = plt.figure(
    figsize=(10, 7)
)

ax = fig.add_subplot(
    111,
    projection="3d"
)

scatter = None
trajectory_line = None


# ============================================================
# FPS TRACKING
# ============================================================

frame_count = 0

start_time = time.time()

last_visualization_time = 0

visualization_interval = 0.5


# ============================================================
# SAVE FUNCTION
# ============================================================

def save_reconstruction():

    print()
    print("=" * 60)
    print("SAVING REAL-TIME RECONSTRUCTION")
    print("=" * 60)

    # --------------------------------------------------------
    # Save point cloud
    # --------------------------------------------------------

    if len(global_points) > 0:

        points = np.asarray(
            global_points,
            dtype=np.float32
        )

        np.savetxt(
            POINT_CLOUD_PATH,
            points,
            fmt="%.6f"
        )

        print(
            f"Point cloud saved:\n"
            f"{POINT_CLOUD_PATH}"
        )

    else:

        print(
            "No 3D points available."
        )


    # --------------------------------------------------------
    # Save camera trajectory
    # --------------------------------------------------------

    with open(
        TRAJECTORY_PATH,
        "w"
    ) as f:

        for i, center in enumerate(
            camera_centers
        ):

            f.write(
                f"{i} "
                f"{center[0]:.6f} "
                f"{center[1]:.6f} "
                f"{center[2]:.6f}\n"
            )

    print(
        f"Camera trajectory saved:\n"
        f"{TRAJECTORY_PATH}"
    )

    print(
        f"Total map points: "
        f"{len(global_points)}"
    )

    print("=" * 60)


# ============================================================
# REAL-TIME LOOP
# ============================================================

try:

    while True:

        # ----------------------------------------------------
        # Capture frame
        # ----------------------------------------------------

        ret, frame = camera.read()

        if not ret:

            print(
                "WARNING: Camera frame unavailable."
            )

            continue


        frame = cv2.resize(
            frame,
            (IMAGE_WIDTH, IMAGE_HEIGHT)
        )


        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )


        frame_count += 1


        # ----------------------------------------------------
        # Feature detection
        # ----------------------------------------------------

        kp_current, des_current = sift.detectAndCompute(
            gray,
            None
        )


        if des_current is None:

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
                "ARIA-S3D | Real-Time Tracking",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:
                break

            continue


        # ----------------------------------------------------
        # Feature matching
        # ----------------------------------------------------

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

            if m.distance < RATIO_TEST * n.distance:

                good_matches.append(m)


        match_count = len(
            good_matches
        )


        # ----------------------------------------------------
        # Display tracking information
        # ----------------------------------------------------

        cv2.putText(
            frame,
            f"Frame: {frame_count}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Features: {len(kp_current)}",
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Good matches: {match_count}",
            (20, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )


        # ----------------------------------------------------
        # Check minimum matches
        # ----------------------------------------------------

        if match_count < MIN_MATCHES:

            cv2.putText(
                frame,
                "TRACKING WEAK",
                (20, 135),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

            cv2.imshow(
                "ARIA-S3D | Real-Time Tracking",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:
                break

            previous_gray = gray
            kp_previous = kp_current
            des_previous = des_current

            continue


        # ----------------------------------------------------
        # Matched image coordinates
        # ----------------------------------------------------

        pts_previous = np.float32([
            kp_previous[m.queryIdx].pt
            for m in good_matches
        ])

        pts_current = np.float32([
            kp_current[m.trainIdx].pt
            for m in good_matches
        ])


        # ----------------------------------------------------
        # Essential matrix
        # ----------------------------------------------------

        E, mask = cv2.findEssentialMat(
            pts_previous,
            pts_current,
            K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=RANSAC_THRESHOLD
        )


        if E is None:

            print(
                f"Frame {frame_count}: "
                "Essential matrix failed."
            )

            previous_gray = gray
            kp_previous = kp_current
            des_previous = des_current

            continue


        # ----------------------------------------------------
        # Recover relative pose
        # ----------------------------------------------------

        try:

            inlier_count, R_relative, t_relative, pose_mask = (
                cv2.recoverPose(
                    E,
                    pts_previous,
                    pts_current,
                    K
                )
            )

        except cv2.error:

            print(
                f"Frame {frame_count}: "
                "Pose recovery failed."
            )

            previous_gray = gray
            kp_previous = kp_current
            des_previous = des_current

            continue


        if inlier_count < MIN_MATCHES:

            cv2.putText(
                frame,
                "POSE UNSTABLE",
                (20, 135),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

            cv2.imshow(
                "ARIA-S3D | Real-Time Tracking",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:
                break

            previous_gray = gray
            kp_previous = kp_current
            des_previous = des_current

            continue


        # ----------------------------------------------------
        # Keep pose inliers
        # ----------------------------------------------------

        pose_mask = (
            pose_mask.ravel().astype(bool)
        )

        pts_previous_pose = (
            pts_previous[pose_mask]
        )

        pts_current_pose = (
            pts_current[pose_mask]
        )


        # ----------------------------------------------------
        # Update world camera pose
        #
        # Current camera:
        #
        # R_current = R_relative @ R_world
        #
        # t_current =
        # R_relative @ t_world + t_relative
        # ----------------------------------------------------

        R_new = (
            R_relative @ R_world
        )

        t_new = (
            R_relative @ t_world
            + t_relative
        )


        R_world = R_new
        t_world = t_new


        # ----------------------------------------------------
        # Camera center in world coordinates
        #
        # C = -R^T t
        # ----------------------------------------------------

        camera_center = (
            -R_world.T @ t_world
        ).ravel()


        camera_centers.append(
            camera_center.copy()
        )


        # ----------------------------------------------------
        # Triangulation
        # ----------------------------------------------------

        if frame_count % TRIANGULATION_INTERVAL == 0:

            P_previous = (
                K @ np.hstack((
                    np.eye(3),
                    np.zeros((3, 1))
                ))
            )


            P_current = (
                K @ np.hstack((
                    R_relative,
                    t_relative
                ))
            )


            try:

                points_4d = (
                    cv2.triangulatePoints(
                        P_previous,
                        P_current,
                        pts_previous_pose.T,
                        pts_current_pose.T
                    )
                )

            except cv2.error:

                points_4d = None


            if points_4d is not None:

                # ------------------------------------------------
                # Homogeneous → Euclidean
                # ------------------------------------------------

                w = points_4d[3]

                valid_w = (
                    np.abs(w) > 1e-8
                )


                points_3d = np.zeros(
                    (
                        points_4d.shape[1],
                        3
                    ),
                    dtype=np.float64
                )


                points_3d[valid_w] = (
                    points_4d[:3, valid_w]
                    / w[valid_w]
                ).T


                # ------------------------------------------------
                # Valid finite points
                # ------------------------------------------------

                finite = np.isfinite(
                    points_3d
                ).all(axis=1)


                points_3d = (
                    points_3d[finite]
                )


                # ------------------------------------------------
                # Positive depth
                # ------------------------------------------------

                if len(points_3d) > 0:

                    positive_depth = (
                        points_3d[:, 2] > 0
                    )

                    points_3d = (
                        points_3d[
                            positive_depth
                        ]
                    )


                # ------------------------------------------------
                # Transform points into world coordinates
                # ------------------------------------------------

                if len(points_3d) > 0:

                    points_world = (
                        R_world.T
                        @ (
                            points_3d.T
                            - t_world
                        )
                    ).T


                    # ------------------------------------------------
                    # Add to global map
                    # ------------------------------------------------

                    global_points.extend(
                        points_world.tolist()
                    )


                    # ------------------------------------------------
                    # Limit map size
                    # ------------------------------------------------

                    if len(global_points) > MAX_MAP_POINTS:

                        global_points = (
                            global_points[
                                -MAX_MAP_POINTS:
                            ]
                        )


        # ----------------------------------------------------
        # FPS calculation
        # ----------------------------------------------------

        elapsed = (
            time.time()
            - start_time
        )

        fps = (
            frame_count / elapsed
            if elapsed > 0
            else 0
        )


        # ----------------------------------------------------
        # Display pose/map information
        # ----------------------------------------------------

        cv2.putText(
            frame,
            f"Pose inliers: {inlier_count}",
            (20, 165),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Map points: {len(global_points)}",
            (20, 195),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (20, 225),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 0),
            2
        )


        # ----------------------------------------------------
        # Draw tracked feature points
        # ----------------------------------------------------

        for point in pts_current_pose[:100]:

            px = int(point[0])
            py = int(point[1])

            cv2.circle(
                frame,
                (px, py),
                2,
                (0, 255, 0),
                -1
            )


        # ----------------------------------------------------
        # Show camera image
        # ----------------------------------------------------

        cv2.imshow(
            "ARIA-S3D | Real-Time Tracking",
            frame
        )


        # ----------------------------------------------------
        # Update 3D visualization
        # ----------------------------------------------------

        current_time = time.time()

        if (
            current_time
            - last_visualization_time
            > visualization_interval
        ):

            last_visualization_time = (
                current_time
            )


            if len(global_points) > 0:

                points_array = np.asarray(
                    global_points,
                    dtype=np.float32
                )


                # Keep visualization lightweight
                display_points = (
                    points_array[
                        ::max(
                            1,
                            len(points_array) // 5000
                        )
                    ]
                )


                ax.clear()


                ax.scatter(
                    display_points[:, 0],
                    display_points[:, 1],
                    display_points[:, 2],
                    s=2,
                    alpha=0.6
                )


                # Camera trajectory
                trajectory = np.asarray(
                    camera_centers
                )


                if len(trajectory) > 1:

                    ax.plot(
                        trajectory[:, 0],
                        trajectory[:, 1],
                        trajectory[:, 2],
                        linewidth=2
                    )


                # Current camera
                ax.scatter(
                    [camera_center[0]],
                    [camera_center[1]],
                    [camera_center[2]],
                    s=50
                )


                ax.set_title(
                    "ARIA-S3D — Real-Time 3D Map"
                )

                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                ax.set_zlabel("Z")


                plt.draw()
                plt.pause(0.001)


        # ----------------------------------------------------
        # Keyboard controls
        # ----------------------------------------------------

        key = cv2.waitKey(1) & 0xFF


        if key == ord("q") or key == 27:

            break


        if key == ord("s"):

            save_reconstruction()


        # ----------------------------------------------------
        # Prepare next frame
        # ----------------------------------------------------

        previous_gray = gray

        kp_previous = kp_current

        des_previous = des_current


finally:

    # ========================================================
    # CLEANUP
    # ========================================================

    print()
    print("=" * 70)
    print("ARIA-S3D | STOPPING REAL-TIME RECONSTRUCTION")
    print("=" * 70)

    save_reconstruction()

    camera.release()

    cv2.destroyAllWindows()

    plt.ioff()

    plt.close("all")


    print()
    print("REAL-TIME RECONSTRUCTION STOPPED")
    print()
    print("Generated outputs:")

    print(
        f"  {POINT_CLOUD_PATH}"
    )

    print(
        f"  {TRAJECTORY_PATH}"
    )

    print()
    print("IMPORTANT:")
    print(
        "This remains a monocular reconstruction."
    )

    print(
        "Absolute metric scale is arbitrary."
    )

    print("=" * 70)