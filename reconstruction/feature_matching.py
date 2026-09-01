import cv2
from pathlib import Path


FRAME_DIR = Path("data/quality_frames")

frames = sorted(FRAME_DIR.glob("*.jpg"))

if len(frames) < 2:
    raise RuntimeError("Need at least 2 frames.")


# --------------------------------------------------
# SIFT detector
# --------------------------------------------------

sift = cv2.SIFT_create(
    nfeatures=3000
)


# --------------------------------------------------
# FLANN matcher
# --------------------------------------------------

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


print("=" * 60)
print("ARIA-S3D | FEATURE MATCHING")
print("=" * 60)

print(f"Frames available: {len(frames)}")
print()


# --------------------------------------------------
# Compare consecutive frames
# --------------------------------------------------

total_matches = 0
successful_pairs = 0

for i in range(len(frames) - 1):

    frame1_path = frames[i]
    frame2_path = frames[i + 1]

    img1 = cv2.imread(str(frame1_path), cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(str(frame2_path), cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        continue

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        continue

    matches = flann.knnMatch(
        des1,
        des2,
        k=2
    )

    # Lowe's ratio test
    good_matches = []

    for m, n in matches:

        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    match_count = len(good_matches)

    total_matches += match_count

    if match_count >= 50:
        successful_pairs += 1

    print(
        f"{frame1_path.name} → "
        f"{frame2_path.name} | "
        f"Good matches: {match_count}"
    )


# --------------------------------------------------
# Summary
# --------------------------------------------------

average_matches = (
    total_matches / (len(frames) - 1)
)

print()
print("=" * 60)
print("FEATURE MATCHING COMPLETE")
print("=" * 60)

print(f"Frame pairs tested:       {len(frames) - 1}")
print(f"Average good matches:     {average_matches:.2f}")
print(f"Successful frame pairs:   {successful_pairs}")

print("=" * 60)