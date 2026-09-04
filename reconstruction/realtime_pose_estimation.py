import cv2
import numpy as np
import os
import time


# ============================================================
# ARIA-S3D | PHASE 4.3
# REAL-TIME POSE ESTIMATION
# ============================================================

OUTPUT_DIR = "data/output"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# CAMERA PARAMETERS
# ============================================================

CAMERA_ID = 0

IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720

fx = 1280.0
fy = 1280.0
cx = IMAGE_WIDTH / 2.0
cy = IMAGE_HEIGHT / 2.0

K = np.array(
    [
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ],
    dtype=np.float64
)


# ============================================================
# PARAMETERS
# ============================================================

MIN_MATCHES = 25
RANSAC_THRESHOLD = 1.0
RANSAC_PROBABILITY = 0.999

SIFT_FEATURES = 2500

DISPLAY_SCALE = 1.0


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("ARIA-S3D | PHASE 4.3")
print("REAL-TIME POSE ESTIMATION")
print("=" * 70)

print()
print(f"Camera ID          : {CAMERA_ID}")
print(f"Resolution         : {IMAGE_WIDTH} x {IMAGE_HEIGHT}")
print(f"SIFT features      : {SIFT_FEATURES}")
print(f"Minimum matches    : {MIN_MATCHES}")
print(f"RANSAC threshold   : {RANSAC_THRESHOLD}")
print()
print("Controls:")
print("Q / ESC : Stop")
print()


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(CAMERA_ID)

if not cap.isOpened():

    print("ERROR: Could not open camera.")

    raise SystemExit


cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    IMAGE_WIDTH
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    IMAGE_HEIGHT
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
# FIRST FRAME
# ============================================================

ret, first_frame = cap.read()

if not ret or first_frame is None:

    print("ERROR: Could not read first camera frame.")

    cap.release()

    raise SystemExit


previous_gray = cv2.cvtColor(
    first_frame,
    cv2.COLOR_BGR2GRAY
)


kp_previous, des_previous = sift.detectAndCompute(
    previous_gray,
    None
)


if des_previous is None:

    print("ERROR: No features detected in first frame.")

    cap.release()

    raise SystemExit


print(
    f"Initial features detected: {len(kp_previous)}"
)


# ============================================================
# GLOBAL CAMERA POSE
#
# Frame 0 = world origin
#
# R_global : camera rotation
# t_global : camera translation
#
# NOTE:
# Monocular translation has arbitrary scale.
# ============================================================

R_global = np.eye(
    3,
    dtype=np.float64
)

t_global = np.zeros(
    (3, 1),
    dtype=np.float64
)


# ============================================================
# POSE HISTORY
# ============================================================

pose_history = []

pose_history.append(
    {
        "frame": 0,
        "x": 0.0,
        "y": 0.0,
        "z": 0.0,
        "tracking": True
    }
)


# ============================================================
# STATISTICS
# ============================================================

frame_count = 0

successful_pose_frames = 0

failed_pose_frames = 0

total_matches = 0

total_inliers = 0

fps_history = []

previous_time = time.time()


# ============================================================
# REAL-TIME LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret or frame is None:

        print("WARNING: Camera frame unavailable.")

        failed_pose_frames += 1

        continue


    frame_count += 1


    # --------------------------------------------------------
    # CURRENT FRAME
    # --------------------------------------------------------

    current_gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )


    # --------------------------------------------------------
    # FEATURE DETECTION
    # --------------------------------------------------------

    kp_current, des_current = sift.detectAndCompute(
        current_gray,
        None
    )


    pose_success = False

    good_matches = []

    inlier_count = 0


    # --------------------------------------------------------
    # DESCRIPTOR VALIDATION
    # --------------------------------------------------------

    if (
        des_current is not None
        and des_previous is not None
    ):

        # ----------------------------------------------------
        # FEATURE MATCHING
        # ----------------------------------------------------

        try:

            matches = flann.knnMatch(
                des_previous,
                des_current,
                k=2
            )

        except cv2.error:

            matches = []


        # ----------------------------------------------------
        # RATIO TEST
        # ----------------------------------------------------

        for pair in matches:

            if len(pair) != 2:

                continue

            m, n = pair

            if m.distance < 0.7 * n.distance:

                good_matches.append(m)


        total_matches += len(good_matches)


        # ----------------------------------------------------
        # POSE ESTIMATION
        # ----------------------------------------------------

        if len(good_matches) >= MIN_MATCHES:

            pts_previous = np.float32(
                [
                    kp_previous[m.queryIdx].pt
                    for m in good_matches
                ]
            )

            pts_current = np.float32(
                [
                    kp_current[m.trainIdx].pt
                    for m in good_matches
                ]
            )


            # ------------------------------------------------
            # ESSENTIAL MATRIX
            # ------------------------------------------------

            try:

                E, essential_mask = cv2.findEssentialMat(
                    pts_previous,
                    pts_current,
                    K,
                    method=cv2.RANSAC,
                    prob=RANSAC_PROBABILITY,
                    threshold=RANSAC_THRESHOLD
                )

            except cv2.error:

                E = None
                essential_mask = None


            if (
                E is not None
                and essential_mask is not None
            ):

                essential_mask = (
                    essential_mask.ravel().astype(bool)
                )


                pts_previous_inliers = (
                    pts_previous[essential_mask]
                )

                pts_current_inliers = (
                    pts_current[essential_mask]
                )


                # --------------------------------------------
                # RECOVER RELATIVE POSE
                # --------------------------------------------

                if len(pts_previous_inliers) >= 8:

                    try:

                        _, R_relative, t_relative, pose_mask = (
                            cv2.recoverPose(
                                E,
                                pts_previous_inliers,
                                pts_current_inliers,
                                K
                            )
                        )

                    except cv2.error:

                        R_relative = None
                        t_relative = None
                        pose_mask = None


                    if (
                        R_relative is not None
                        and t_relative is not None
                        and pose_mask is not None
                    ):

                        pose_mask = (
                            pose_mask.ravel().astype(bool)
                        )

                        inlier_count = int(
                            np.count_nonzero(pose_mask)
                        )

                        total_inliers += inlier_count


                        # ------------------------------------
                        # VALID POSE
                        # ------------------------------------

                        if inlier_count >= 8:

                            # --------------------------------
                            # UPDATE GLOBAL POSE
                            #
                            # t_relative has arbitrary
                            # monocular scale.
                            # --------------------------------

                            t_global = (
                                t_global
                                + R_global @ t_relative
                            )


                            R_global = (
                                R_relative @ R_global
                            )


                            pose_success = True


    # ========================================================
    # TRACKING STATUS
    # ========================================================

    if pose_success:

        successful_pose_frames += 1

        tracking_status = "TRACKING"

    else:

        failed_pose_frames += 1

        tracking_status = "POSE LOST"


    # ========================================================
    # CAMERA POSITION
    # ========================================================

    camera_position = t_global.ravel()

    camera_x = float(camera_position[0])

    camera_y = float(camera_position[1])

    camera_z = float(camera_position[2])


    # ========================================================
    # SAVE POSE HISTORY
    # ========================================================

    pose_history.append(
        {
            "frame": frame_count,
            "x": camera_x,
            "y": camera_y,
            "z": camera_z,
            "tracking": pose_success
        }
    )


    # ========================================================
    # FPS
    # ========================================================

    current_time = time.time()

    delta_time = (
        current_time - previous_time
    )

    previous_time = current_time


    if delta_time > 0:

        current_fps = 1.0 / delta_time

    else:

        current_fps = 0.0


    fps_history.append(
        current_fps
    )


    # ========================================================
    # VISUALIZATION
    # ========================================================

    display = frame.copy()


    # --------------------------------------------------------
    # STATUS
    # --------------------------------------------------------

    cv2.putText(
        display,
        f"ARIA-S3D | PHASE 4.3",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )


    cv2.putText(
        display,
        f"Pose: {tracking_status}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (
            (0, 255, 0)
            if pose_success
            else (0, 0, 255)
        ),
        2
    )


    # --------------------------------------------------------
    # MATCH INFORMATION
    # --------------------------------------------------------

    cv2.putText(
        display,
        f"Matches: {len(good_matches)}",
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    cv2.putText(
        display,
        f"Pose inliers: {inlier_count}",
        (20, 135),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    # --------------------------------------------------------
    # CAMERA POSITION
    # --------------------------------------------------------

    cv2.putText(
        display,
        f"X: {camera_x:.3f}",
        (20, 175),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        display,
        f"Y: {camera_y:.3f}",
        (20, 205),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        display,
        f"Z: {camera_z:.3f}",
        (20, 235),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    # --------------------------------------------------------
    # FPS
    # --------------------------------------------------------

    cv2.putText(
        display,
        f"FPS: {current_fps:.2f}",
        (20, 275),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    # --------------------------------------------------------
    # FRAME NUMBER
    # --------------------------------------------------------

    cv2.putText(
        display,
        f"Frame: {frame_count}",
        (20, 310),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    # ========================================================
    # SHOW FRAME
    # ========================================================

    if DISPLAY_SCALE != 1.0:

        display = cv2.resize(
            display,
            None,
            fx=DISPLAY_SCALE,
            fy=DISPLAY_SCALE
        )


    cv2.imshow(
        "ARIA-S3D | Real-Time Pose Estimation",
        display
    )


    # ========================================================
    # PREPARE NEXT FRAME
    # ========================================================

    previous_gray = current_gray

    kp_previous = kp_current

    des_previous = des_current


    # ========================================================
    # EXIT
    # ========================================================

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q") or key == 27:

        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()


# ============================================================
# FINAL STATISTICS
# ============================================================

print()
print("=" * 70)
print("ARIA-S3D | PHASE 4.3 COMPLETE")
print("=" * 70)

print()

print(
    f"Frames processed       : {frame_count}"
)

print(
    f"Successful pose frames : {successful_pose_frames}"
)

print(
    f"Failed pose frames     : {failed_pose_frames}"
)


# ------------------------------------------------------------
# SUCCESS RATE
# ------------------------------------------------------------

if frame_count > 0:

    pose_success_rate = (
        successful_pose_frames
        / frame_count
        * 100.0
    )

else:

    pose_success_rate = 0.0


print(
    f"Pose success rate      : {pose_success_rate:.2f}%"
)


# ------------------------------------------------------------
# MATCH STATISTICS
# ------------------------------------------------------------

if frame_count > 0:

    average_matches = (
        total_matches
        / frame_count
    )

else:

    average_matches = 0.0


if successful_pose_frames > 0:

    average_inliers = (
        total_inliers
        / successful_pose_frames
    )

else:

    average_inliers = 0.0


print(
    f"Average good matches   : {average_matches:.2f}"
)

print(
    f"Average pose inliers   : {average_inliers:.2f}"
)


# ------------------------------------------------------------
# FPS STATISTICS
# ------------------------------------------------------------

if len(fps_history) > 0:

    valid_fps = [
        fps
        for fps in fps_history
        if np.isfinite(fps) and fps > 0
    ]

else:

    valid_fps = []


if len(valid_fps) > 0:

    average_fps = np.mean(valid_fps)

    minimum_fps = np.min(valid_fps)

    maximum_fps = np.max(valid_fps)

else:

    average_fps = 0.0

    minimum_fps = 0.0

    maximum_fps = 0.0


print(
    f"Average FPS           : {average_fps:.2f}"
)

print(
    f"Minimum FPS           : {minimum_fps:.2f}"
)

print(
    f"Maximum FPS           : {maximum_fps:.2f}"
)


# ============================================================
# SAVE POSE TRAJECTORY
# ============================================================

pose_path = os.path.join(
    OUTPUT_DIR,
    "realtime_pose_trajectory.txt"
)


with open(
    pose_path,
    "w"
) as f:

    f.write(
        "# frame x y z tracking\n"
    )

    for pose in pose_history:

        tracking_value = (
            1
            if pose["tracking"]
            else 0
        )

        f.write(
            f"{pose['frame']} "
            f"{pose['x']:.6f} "
            f"{pose['y']:.6f} "
            f"{pose['z']:.6f} "
            f"{tracking_value}\n"
        )


print()
print(
    f"Pose trajectory saved to:"
)

print(
    f"{pose_path}"
)


# ============================================================
# SAVE ROTATION + TRANSLATION
# ============================================================

final_pose_path = os.path.join(
    OUTPUT_DIR,
    "realtime_final_pose.txt"
)


with open(
    final_pose_path,
    "w"
) as f:

    f.write(
        "ARIA-S3D FINAL CAMERA POSE\n"
    )

    f.write(
        "Rotation matrix:\n"
    )

    for row in R_global:

        f.write(
            " ".join(
                f"{value:.9f}"
                for value in row
            )
            + "\n"
        )


    f.write(
        "\nTranslation vector:\n"
    )

    for value in t_global.ravel():

        f.write(
            f"{value:.9f}\n"
        )


print(
    f"Final camera pose saved to:"
)

print(
    f"{final_pose_path}"
)


# ============================================================
# SAVE STATISTICS
# ============================================================

stats_path = os.path.join(
    OUTPUT_DIR,
    "realtime_pose_stats.txt"
)


with open(
    stats_path,
    "w"
) as f:

    f.write(
        "ARIA-S3D | PHASE 4.3 REAL-TIME POSE ESTIMATION\n"
    )

    f.write(
        "=" * 60
        + "\n"
    )

    f.write(
        f"Frames processed: {frame_count}\n"
    )

    f.write(
        f"Successful pose frames: "
        f"{successful_pose_frames}\n"
    )

    f.write(
        f"Failed pose frames: "
        f"{failed_pose_frames}\n"
    )

    f.write(
        f"Pose success rate: "
        f"{pose_success_rate:.2f}%\n"
    )

    f.write(
        f"Average good matches: "
        f"{average_matches:.2f}\n"
    )

    f.write(
        f"Average pose inliers: "
        f"{average_inliers:.2f}\n"
    )

    f.write(
        f"Average FPS: "
        f"{average_fps:.2f}\n"
    )

    f.write(
        f"Minimum FPS: "
        f"{minimum_fps:.2f}\n"
    )

    f.write(
        f"Maximum FPS: "
        f"{maximum_fps:.2f}\n"
    )

    f.write(
        "\nFinal camera position:\n"
    )

    f.write(
        f"X = {camera_x:.6f}\n"
    )

    f.write(
        f"Y = {camera_y:.6f}\n"
    )

    f.write(
        f"Z = {camera_z:.6f}\n"
    )

    f.write(
        "\nIMPORTANT:\n"
    )

    f.write(
        "This is a monocular reconstruction.\n"
    )

    f.write(
        "Absolute metric scale remains arbitrary.\n"
    )


print(
    f"Pose statistics saved to:"
)

print(
    f"{stats_path}"
)


# ============================================================
# FINAL MESSAGE
# ============================================================

print()
print("=" * 70)

print(
    "REAL-TIME POSE ESTIMATION STOPPED"
)

print("=" * 70)

print()
print("Generated outputs:")

print(
    f"  {pose_path}"
)

print(
    f"  {final_pose_path}"
)

print(
    f"  {stats_path}"
)

print()
print("NEXT STEP:")
print("Phase 4.4 - Incremental live mapping")

print("=" * 70)

print()
print("IMPORTANT:")
print(
    "This remains a monocular pose estimation system."
)

print(
    "Translation scale is arbitrary."
)

print("=" * 70)