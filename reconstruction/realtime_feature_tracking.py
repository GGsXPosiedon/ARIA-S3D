import cv2
import numpy as np
import time
from pathlib import Path


# ============================================================
# ARIA-S3D | PHASE 4.2
# REAL-TIME FEATURE TRACKING
# ============================================================

OUTPUT_DIR = Path("data/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRACK_STATS_PATH = OUTPUT_DIR / "realtime_tracking_stats.txt"


# ============================================================
# CAMERA CONFIGURATION
# ============================================================

CAMERA_ID = 0

IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720


# ============================================================
# TRACKING CONFIGURATION
# ============================================================

SIFT_FEATURES = 2500

LOWE_RATIO = 0.70

MIN_GOOD_MATCHES = 25

RANSAC_REPROJECTION_THRESHOLD = 1.5

MAX_DRAWN_TRACKS = 250


# ============================================================
# INITIALIZE CAMERA
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
# FEATURE DETECTOR
# ============================================================

sift = cv2.SIFT_create(
    nfeatures=SIFT_FEATURES
)


# ============================================================
# FEATURE MATCHER
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
# HEADER
# ============================================================

print("=" * 70)
print("ARIA-S3D | PHASE 4.2")
print("REAL-TIME FEATURE TRACKING")
print("=" * 70)

print()
print(f"Camera ID: {CAMERA_ID}")
print(f"Resolution: {IMAGE_WIDTH} x {IMAGE_HEIGHT}")
print(f"SIFT features: {SIFT_FEATURES}")
print(f"Minimum good matches: {MIN_GOOD_MATCHES}")
print()

print("Controls:")
print("  Q / ESC : Stop")
print()


# ============================================================
# READ INITIAL FRAME
# ============================================================

ret, frame_previous = camera.read()

if not ret:

    camera.release()

    raise RuntimeError(
        "ERROR: Could not read first frame."
    )


frame_previous = cv2.resize(
    frame_previous,
    (IMAGE_WIDTH, IMAGE_HEIGHT)
)


gray_previous = cv2.cvtColor(
    frame_previous,
    cv2.COLOR_BGR2GRAY
)


kp_previous, des_previous = (
    sift.detectAndCompute(
        gray_previous,
        None
    )
)


if des_previous is None:

    camera.release()

    raise RuntimeError(
        "ERROR: No descriptors in first frame."
    )


# ============================================================
# TRACKING STATISTICS
# ============================================================

frame_count = 0

successful_tracking_frames = 0

weak_tracking_frames = 0

lost_tracking_frames = 0

total_good_matches = 0

total_inliers = 0

start_time = time.time()


# ============================================================
# MAIN LOOP
# ============================================================

try:

    while True:

        ret, frame_current = camera.read()

        if not ret:

            print(
                "WARNING: Could not read camera frame."
            )

            continue


        frame_current = cv2.resize(
            frame_current,
            (IMAGE_WIDTH, IMAGE_HEIGHT)
        )


        gray_current = cv2.cvtColor(
            frame_current,
            cv2.COLOR_BGR2GRAY
        )


        frame_count += 1


        # ----------------------------------------------------
        # DETECT FEATURES
        # ----------------------------------------------------

        kp_current, des_current = (
            sift.detectAndCompute(
                gray_current,
                None
            )
        )


        if des_current is None:

            lost_tracking_frames += 1

            cv2.putText(
                frame_current,
                "TRACKING LOST - NO FEATURES",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

            cv2.imshow(
                "ARIA-S3D | Real-Time Feature Tracking",
                frame_current
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:
                break

            gray_previous = gray_current
            kp_previous = kp_current
            des_previous = des_current

            continue


        # ----------------------------------------------------
        # FEATURE MATCHING
        # ----------------------------------------------------

        try:

            raw_matches = flann.knnMatch(
                des_previous,
                des_current,
                k=2
            )

        except cv2.error:

            raw_matches = []


        good_matches = []


        for pair in raw_matches:

            if len(pair) != 2:
                continue

            m, n = pair

            if m.distance < LOWE_RATIO * n.distance:

                good_matches.append(m)


        good_match_count = len(
            good_matches
        )


        total_good_matches += (
            good_match_count
        )


        # ----------------------------------------------------
        # WEAK TRACKING
        # ----------------------------------------------------

        if good_match_count < MIN_GOOD_MATCHES:

            weak_tracking_frames += 1

            cv2.putText(
                frame_current,
                "TRACKING WEAK",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 165, 255),
                2
            )

            cv2.putText(
                frame_current,
                f"Good matches: {good_match_count}",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 165, 255),
                2
            )

            cv2.imshow(
                "ARIA-S3D | Real-Time Feature Tracking",
                frame_current
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:
                break

            gray_previous = gray_current
            kp_previous = kp_current
            des_previous = des_current

            continue


        # ----------------------------------------------------
        # MATCHED 2D POINTS
        # ----------------------------------------------------

        pts_previous = np.float32([
            kp_previous[
                m.queryIdx
            ].pt
            for m in good_matches
        ])


        pts_current = np.float32([
            kp_current[
                m.trainIdx
            ].pt
            for m in good_matches
        ])


        # ----------------------------------------------------
        # GEOMETRIC FILTERING
        #
        # Fundamental matrix used here only to verify
        # geometric consistency of tracked features.
        # ----------------------------------------------------

        F, mask = cv2.findFundamentalMat(
            pts_previous,
            pts_current,
            cv2.FM_RANSAC,
            RANSAC_REPROJECTION_THRESHOLD,
            0.99
        )


        if F is None or mask is None:

            lost_tracking_frames += 1

            cv2.putText(
                frame_current,
                "GEOMETRIC TRACKING FAILED",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

            cv2.imshow(
                "ARIA-S3D | Real-Time Feature Tracking",
                frame_current
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:
                break

            gray_previous = gray_current
            kp_previous = kp_current
            des_previous = des_current

            continue


        # ----------------------------------------------------
        # KEEP GEOMETRIC INLIERS
        # ----------------------------------------------------

        mask = mask.ravel().astype(bool)

        inlier_previous = (
            pts_previous[mask]
        )

        inlier_current = (
            pts_current[mask]
        )


        inlier_count = len(
            inlier_current
        )


        total_inliers += inlier_count


        # ----------------------------------------------------
        # TRACKING QUALITY
        # ----------------------------------------------------

        if inlier_count >= MIN_GOOD_MATCHES:

            successful_tracking_frames += 1

            status_text = "TRACKING GOOD"

            status_color = (
                0,
                255,
                0
            )

        else:

            weak_tracking_frames += 1

            status_text = "TRACKING UNSTABLE"

            status_color = (
                0,
                165,
                255
            )


        # ----------------------------------------------------
        # DRAW FEATURE TRACKS
        # ----------------------------------------------------

        draw_count = min(
            len(inlier_current),
            MAX_DRAWN_TRACKS
        )


        for i in range(draw_count):

            p1 = tuple(
                np.int32(
                    inlier_previous[i]
                )
            )

            p2 = tuple(
                np.int32(
                    inlier_current[i]
                )
            )


            cv2.line(
                frame_current,
                p1,
                p2,
                (255, 0, 0),
                1
            )


            cv2.circle(
                frame_current,
                p2,
                3,
                (0, 255, 0),
                -1
            )


        # ----------------------------------------------------
        # FPS
        # ----------------------------------------------------

        elapsed = (
            time.time()
            -
            start_time
        )


        fps = (
            frame_count / elapsed
            if elapsed > 0
            else 0
        )


        # ----------------------------------------------------
        # TRACKING SUCCESS RATE
        # ----------------------------------------------------

        tracking_success_rate = (

            successful_tracking_frames
            /
            max(
                frame_count,
                1
            )

            *
            100.0
        )


        # ----------------------------------------------------
        # HUD
        # ----------------------------------------------------

        cv2.putText(
            frame_current,
            status_text,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            status_color,
            2
        )


        cv2.putText(
            frame_current,
            f"Frame: {frame_count}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )


        cv2.putText(
            frame_current,
            f"Features: {len(kp_current)}",
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )


        cv2.putText(
            frame_current,
            f"Good matches: {good_match_count}",
            (20, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )


        cv2.putText(
            frame_current,
            f"RANSAC inliers: {inlier_count}",
            (20, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )


        cv2.putText(
            frame_current,
            f"Success rate: {tracking_success_rate:.1f}%",
            (20, 190),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )


        cv2.putText(
            frame_current,
            f"FPS: {fps:.1f}",
            (20, 220),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )


        # ----------------------------------------------------
        # DISPLAY
        # ----------------------------------------------------

        cv2.imshow(
            "ARIA-S3D | Real-Time Feature Tracking",
            frame_current
        )


        key = cv2.waitKey(1) & 0xFF


        if key == ord("q") or key == 27:
            break


        # ----------------------------------------------------
        # PREPARE NEXT FRAME
        # ----------------------------------------------------

        gray_previous = gray_current

        kp_previous = kp_current

        des_previous = des_current


finally:

    # ========================================================
    # CLEANUP
    # ========================================================

    camera.release()

    cv2.destroyAllWindows()


# ============================================================
# FINAL STATISTICS
# ============================================================

elapsed = (
    time.time()
    -
    start_time
)


average_fps = (
    frame_count / elapsed
    if elapsed > 0
    else 0.0
)


average_good_matches = (
    total_good_matches / frame_count
    if frame_count > 0
    else 0.0
)


average_inliers = (
    total_inliers / frame_count
    if frame_count > 0
    else 0.0
)


success_rate = (

    successful_tracking_frames
    /
    frame_count
    *
    100.0

    if frame_count > 0
    else 0.0
)


print()
print("=" * 70)
print("ARIA-S3D | PHASE 4.2 COMPLETE")
print("=" * 70)

print(
    f"Frames processed: "
    f"{frame_count}"
)

print(
    f"Successful tracking frames: "
    f"{successful_tracking_frames}"
)

print(
    f"Weak tracking frames: "
    f"{weak_tracking_frames}"
)

print(
    f"Lost tracking frames: "
    f"{lost_tracking_frames}"
)

print(
    f"Tracking success rate: "
    f"{success_rate:.2f}%"
)

print(
    f"Average good matches: "
    f"{average_good_matches:.2f}"
)

print(
    f"Average geometric inliers: "
    f"{average_inliers:.2f}"
)

print(
    f"Average FPS: "
    f"{average_fps:.2f}"
)


# ============================================================
# SAVE TRACKING STATS
# ============================================================

with open(
    TRACK_STATS_PATH,
    "w"
) as f:

    f.write(
        "ARIA-S3D Phase 4.2 "
        "Real-Time Feature Tracking\n"
    )

    f.write(
        "=" * 50 + "\n"
    )

    f.write(
        f"Frames processed: "
        f"{frame_count}\n"
    )

    f.write(
        f"Successful tracking frames: "
        f"{successful_tracking_frames}\n"
    )

    f.write(
        f"Weak tracking frames: "
        f"{weak_tracking_frames}\n"
    )

    f.write(
        f"Lost tracking frames: "
        f"{lost_tracking_frames}\n"
    )

    f.write(
        f"Tracking success rate: "
        f"{success_rate:.6f}%\n"
    )

    f.write(
        f"Average good matches: "
        f"{average_good_matches:.6f}\n"
    )

    f.write(
        f"Average geometric inliers: "
        f"{average_inliers:.6f}\n"
    )

    f.write(
        f"Average FPS: "
        f"{average_fps:.6f}\n"
    )


print()
print(
    f"Tracking statistics saved to:\n"
    f"{TRACK_STATS_PATH}"
)

print()
print(
    "NEXT STEP:"
)

print(
    "Phase 4.3 — Real-time pose estimation"
)

print("=" * 70)