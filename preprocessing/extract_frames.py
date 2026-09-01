import cv2
from pathlib import Path

VIDEO_PATH = Path("data/raw_video/video.mp4")
OUTPUT_DIR = Path("data/frames")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = total_frames / fps if fps > 0 else 0

print("=" * 50)
print("ARIA-S3D | VIDEO ANALYSIS")
print("=" * 50)
print(f"FPS:             {fps:.2f}")
print(f"Total frames:    {total_frames}")
print(f"Duration:        {duration:.2f} seconds")
print("=" * 50)

# Target: approximately 5 frames per second
interval = max(round(fps / 5), 1)

frame_number = 0
saved_frames = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    if frame_number % interval == 0:
        filename = OUTPUT_DIR / f"frame_{saved_frames:05d}.jpg"

        success = cv2.imwrite(str(filename), frame)

        if success:
            saved_frames += 1

    frame_number += 1

cap.release()

print(f"Frames extracted: {saved_frames}")
print(f"Output directory: {OUTPUT_DIR}")
print("Extraction complete.")