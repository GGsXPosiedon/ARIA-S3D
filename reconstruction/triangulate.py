import cv2
import numpy as np
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

FRAME_DIR = Path("data/quality_frames")

FRAME_1 = FRAME_DIR / "frame_00000.jpg"
FRAME_2 = FRAME_DIR / "frame_00010.jpg"

OUTPUT_DIR = Path("data/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD IMAGES
# ============================================================

img1 = cv2.imread(str(FRAME_1), cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(str(FRAME_2), cv2.IMREAD_GRAYSCALE)

if img1 is None or img2 is None:
    raise RuntimeError("Could not load input frames.")


height, width = img1.shape


print("=" * 70)
print("ARIA-S3D | FIRST 3D TRIANGULATION")
print("=" * 70)

print(f"Image size: {width} x {height}")
print(f"Frame 1: {FRAME_1.name}")
print(f"Frame 2: {FRAME_2.name}")


# ============================================================
# CAMERA INTRINSICS
# ============================================================

focal_length = max(width, height)

cx = width / 2
cy = height / 2

K = np.array([
    [focal_length, 0, cx],
    [0, focal_length, cy],
    [0, 0, 1]
], dtype=np.float64)


print()
print("Camera matrix:")
print(K)


# ============================================================
# SIFT FEATURES
# ============================================================

sift = cv2.SIFT_create(
    nfeatures=3000
)

kp1, des1 = sift.detectAndCompute(
    img1,
    None
)

kp2, des2 = sift.detectAndCompute(
    img2,
    None
)

print()
print(f"Features frame 1: {len(kp1)}")
print(f"Features frame 2: {len(kp2)}")


# ============================================================
# FEATURE MATCHING
# ============================================================

bf = cv2.BFMatcher()

raw_matches = bf.knnMatch(
    des1,
    des2,
    k=2
)

good_matches = []

for m, n in raw_matches:

    if m.distance < 0.7 * n.distance:
        good_matches.append(m)


print(f"Good matches: {len(good_matches)}")


# ============================================================
# MATCHED POINTS
# ============================================================

pts1 = np.float32([
    kp1[m.queryIdx].pt
    for m in good_matches
])

pts2 = np.float32([
    kp2[m.trainIdx].pt
    for m in good_matches
])


# ============================================================
# ESSENTIAL MATRIX
# ============================================================

E, mask = cv2.findEssentialMat(
    pts1,
    pts2,
    K,
    method=cv2.RANSAC,
    prob=0.999,
    threshold=1.5
)

if E is None:
    raise RuntimeError(
        "Essential matrix estimation failed."
    )


# ============================================================
# KEEP ONLY GEOMETRIC INLIERS
# ============================================================

mask = mask.ravel().astype(bool)

pts1_inliers = pts1[mask]
pts2_inliers = pts2[mask]

print(
    f"Geometric inliers: {len(pts1_inliers)}"
)


# ============================================================
# RECOVER CAMERA POSE
# ============================================================

_, R, t, pose_mask = cv2.recoverPose(
    E,
    pts1_inliers,
    pts2_inliers,
    K
)


print()
print("Rotation matrix:")
print(R)

print()
print("Translation direction:")
print(t.flatten())


# ============================================================
# CAMERA PROJECTION MATRICES
# ============================================================

# Camera 1
P1 = K @ np.hstack(
    (
        np.eye(3),
        np.zeros((3, 1))
    )
)


# Camera 2
P2 = K @ np.hstack(
    (
        R,
        t
    )
)


# ============================================================
# TRIANGULATION
# ============================================================

points_4d = cv2.triangulatePoints(
    P1,
    P2,
    pts1_inliers.T,
    pts2_inliers.T
)


# Convert homogeneous coordinates → 3D
points_3d = (
    points_4d[:3]
    /
    points_4d[3]
).T


# ============================================================
# REMOVE INVALID POINTS
# ============================================================

valid = np.isfinite(points_3d).all(axis=1)

points_3d = points_3d[valid]


print()
print("=" * 70)
print("TRIANGULATION RESULT")
print("=" * 70)

print(
    f"3D points generated: {len(points_3d)}"
)


# ============================================================
# DEPTH FILTER
# ============================================================

# Camera 1 coordinates
z_values = points_3d[:, 2]

positive_depth = z_values > 0

points_3d = points_3d[positive_depth]


print(
    f"Points with positive depth: {len(points_3d)}"
)


# ============================================================
# SAVE POINT CLOUD
# ============================================================

output_file = OUTPUT_DIR / "point_cloud.xyz"

np.savetxt(
    output_file,
    points_3d,
    fmt="%.6f"
)


print()
print(f"Point cloud saved to:")
print(output_file)


# ============================================================
# STATISTICS
# ============================================================

if len(points_3d) > 0:

    mins = points_3d.min(axis=0)
    maxs = points_3d.max(axis=0)
    means = points_3d.mean(axis=0)

    print()
    print("3D CLOUD STATISTICS")
    print("-" * 70)

    print("Minimum XYZ:")
    print(mins)

    print()
    print("Maximum XYZ:")
    print(maxs)

    print()
    print("Mean XYZ:")
    print(means)


print()
print("=" * 70)
print("IMPORTANT")
print("=" * 70)

print(
    "This is a RELATIVE 3D reconstruction."
)

print(
    "The scale is arbitrary because monocular camera "
    "translation has unknown scale."
)

print("=" * 70)