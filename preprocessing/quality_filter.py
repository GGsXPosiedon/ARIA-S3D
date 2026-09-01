import cv2
import numpy as np
from pathlib import Path

INPUT_DIR = Path("data/frames")
OUTPUT_DIR = Path("data/quality_frames")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def sharpness_score(image):
    """
    Higher value = sharper image.
    Uses variance of Laplacian.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def brightness_score(image):
    """
    Returns average brightness from 0 to 255.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))


def feature_score(image):
    """
    Counts SIFT keypoints.
    More useful visual features generally
    means better reconstruction potential.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    keypoints, _ = sift.detectAndCompute(gray, None)

    return len(keypoints)


frames = sorted(INPUT_DIR.glob("*.jpg"))

if not frames:
    raise RuntimeError("No frames found in data/frames")

print("=" * 60)
print("ARIA-S3D | FRAME QUALITY ANALYSIS")
print("=" * 60)
print(f"Frames found: {len(frames)}")
print()

results = []

for index, frame_path in enumerate(frames):

    image = cv2.imread(str(frame_path))

    if image is None:
        print(f"Could not read: {frame_path.name}")
        continue

    sharpness = sharpness_score(image)
    brightness = brightness_score(image)
    features = feature_score(image)

    results.append({
        "path": frame_path,
        "sharpness": sharpness,
        "brightness": brightness,
        "features": features
    })

    print(
        f"{frame_path.name} | "
        f"Sharpness: {sharpness:8.2f} | "
        f"Brightness: {brightness:6.2f} | "
        f"Features: {features}"
    )


# --------------------------------------------------
# Select good frames
# --------------------------------------------------

selected = []

for result in results:

    sharpness_ok = result["sharpness"] > 100
    brightness_ok = 30 < result["brightness"] < 230
    features_ok = result["features"] > 100

    if sharpness_ok and brightness_ok and features_ok:
        selected.append(result)


# --------------------------------------------------
# Copy selected frames
# --------------------------------------------------

for result in selected:

    image = cv2.imread(str(result["path"]))

    output_path = OUTPUT_DIR / result["path"].name

    cv2.imwrite(str(output_path), image)


print()
print("=" * 60)
print("QUALITY FILTER COMPLETE")
print("=" * 60)
print(f"Input frames:    {len(results)}")
print(f"Selected frames: {len(selected)}")
print(f"Rejected frames: {len(results) - len(selected)}")
print(f"Output:          {OUTPUT_DIR}")
print("=" * 60)