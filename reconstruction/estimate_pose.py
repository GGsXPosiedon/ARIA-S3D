import cv2
import numpy as np
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

FRAME_DIR = Path("data/quality_frames")

frames = sorted(FRAME_DIR.glob("*.jpg"))

if len(frames) < 16:
    raise RuntimeError("Need at least 16 frames.")


# Test different temporal gaps
FRAME_GAPS = [1, 3, 5, 10, 15]


# ============================================================
# LOAD FIRST FRAME
# ============================================================

img1 = cv2.imread(
    str(frames[0]),
    cv2.IMREAD_GRAYSCALE
)

if img1 is None:
    raise RuntimeError("Could not read first frame.")


height, width = img1.shape


print("=" * 70)
print("ARIA-S3D | ADAPTIVE CAMERA POSE TEST")
print("=" * 70)

print(f"Image size: {width} x {height}")
print(f"Total frames: {len(frames)}")
print()


# ============================================================
# CAMERA MATRIX
# ============================================================

# Prototype approximation.
# We will replace this with real calibration later.

focal_length = max(width, height)

cx = width / 2
cy = height / 2

K = np.array([
    [focal_length, 0, cx],
    [0, focal_length, cy],
    [0, 0, 1]
], dtype=np.float64)


print("Camera matrix:")
print(K)
print()


# ============================================================
# SIFT
# ============================================================

sift = cv2.SIFT_create(
    nfeatures=3000
)


kp1, des1 = sift.detectAndCompute(
    img1,
    None
)

print(f"Features in frame 0: {len(kp1)}")
print()


# ============================================================
# TEST DIFFERENT FRAME GAPS
# ============================================================

results = []


for gap in FRAME_GAPS:

    if gap >= len(frames):
        continue

    frame2_path = frames[gap]

    img2 = cv2.imread(
        str(frame2_path),
        cv2.IMREAD_GRAYSCALE
    )

    if img2 is None:
        continue


    kp2, des2 = sift.detectAndCompute(
        img2,
        None
    )


    if des2 is None:
        continue


    # --------------------------------------------------------
    # Feature matching
    # --------------------------------------------------------

    matcher = cv2.BFMatcher()

    raw_matches = matcher.knnMatch(
        des1,
        des2,
        k=2
    )


    good_matches = []

    for m, n in raw_matches:

        if m.distance < 0.7 * n.distance:
            good_matches.append(m)


    if len(good_matches) < 20:

        print(
            f"Gap {gap:2d} | "
            f"Matches: {len(good_matches):4d} | "
            f"NOT ENOUGH MATCHES"
        )

        continue


    # --------------------------------------------------------
    # Matched coordinates
    # --------------------------------------------------------

    pts1 = np.float32([
        kp1[m.queryIdx].pt
        for m in good_matches
    ])

    pts2 = np.float32([
        kp2[m.trainIdx].pt
        for m in good_matches
    ])


    # --------------------------------------------------------
    # Essential matrix
    # --------------------------------------------------------

    E, mask = cv2.findEssentialMat(
        pts1,
        pts2,
        K,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=1.5
    )


    if E is None:

        print(
            f"Gap {gap:2d} | "
            f"Matches: {len(good_matches):4d} | "
            f"Essential matrix FAILED"
        )

        continue


    # --------------------------------------------------------
    # Recover pose
    # --------------------------------------------------------

    try:

        inliers, R, t, pose_mask = cv2.recoverPose(
            E,
            pts1,
            pts2,
            K,
            mask=mask
        )

    except cv2.error:

        print(
            f"Gap {gap:2d} | "
            f"Pose recovery FAILED"
        )

        continue


    inlier_ratio = (
        inliers / len(good_matches)
    ) * 100


    results.append({
        "gap": gap,
        "matches": len(good_matches),
        "inliers": inliers,
        "ratio": inlier_ratio,
        "R": R,
        "t": t
    })


    print(
        f"Gap {gap:2d} | "
        f"Matches: {len(good_matches):4d} | "
        f"Inliers: {inliers:4d} | "
        f"Inlier ratio: {inlier_ratio:6.2f}%"
    )


# ============================================================
# FIND BEST PAIR
# ============================================================

print()
print("=" * 70)
print("POSE TEST SUMMARY")
print("=" * 70)


if not results:

    print("No usable pose found.")
    raise SystemExit


best = max(
    results,
    key=lambda x: x["inliers"]
)


print()
print("BEST FRAME PAIR")
print("-" * 70)

print(
    f"Frame 0 → Frame {best['gap']}"
)

print(
    f"Good matches:       {best['matches']}"
)

print(
    f"Geometric inliers:   {best['inliers']}"
)

print(
    f"Inlier ratio:        {best['ratio']:.2f}%"
)


print()
print("Rotation matrix R:")
print(best["R"])


print()
print("Translation direction:")
print(best["t"].flatten())


print()
print("=" * 70)
print("INTERPRETATION")
print("=" * 70)

if best["inliers"] >= 100:

    print("🟢 Strong geometric consistency.")

elif best["inliers"] >= 50:

    print("🟡 Usable, but should be improved.")

else:

    print("🔴 Weak geometry. Do NOT triangulate yet.")


print()
print(
    "Translation scale is still unknown in monocular vision."
)

print("=" * 70)