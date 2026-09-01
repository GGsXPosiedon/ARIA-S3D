# ARIA-S3D

### AI-powered Monocular 3D Reconstruction System

ARIA-S3D is an experimental computer vision system designed to reconstruct 3D environments from monocular video.

The project currently implements the fundamental stages of a Structure-from-Motion pipeline.

## Current Pipeline

Video
↓
Frame Extraction
↓
Frame Quality Filtering
↓
SIFT Feature Detection
↓
Feature Matching
↓
Camera Pose Estimation
↓
Adaptive Frame Pair Selection
↓
3D Triangulation
↓
3D Point Cloud Visualization

## Current Capabilities

- Extract frames from video
- Evaluate frame quality
- Detect SIFT features
- Match features between frames
- Estimate relative camera pose
- Select geometrically consistent frame pairs
- Triangulate 3D points
- Visualize reconstructed point clouds

## Current Results

First prototype successfully reconstructed:

- 191 video frames processed
- 1,420 valid 3D points generated
- 1,420 points with positive depth
- Successful feature matching across tested frame pairs
- 3D point cloud visualization implemented

## Important Limitation

The current system uses monocular vision.

Therefore, the reconstructed 3D scene has an arbitrary scale. The system currently estimates relative camera motion rather than absolute real-world distance.

## Technology Stack

- Python
- OpenCV
- NumPy
- Matplotlib
- SIFT
- Epipolar Geometry
- Triangulation
- Structure from Motion

## Project Structure

ARIA-S3D/
│
├── preprocessing/
│   ├── extract_frames.py
│   └── quality_filter.py
│
├── reconstruction/
│   ├── feature_matching.py
│   ├── estimate_pose.py
│   └── triangulate.py
│
├── viewer/
│   └── point_cloud_viewer.py
│
├── data/
│   ├── frames/
│   ├── quality_frames/
│   ├── raw_video/
│   └── output/
│
├── .gitignore
└── README.md

## Roadmap

### Phase 1 — Prototype
- [x] Frame extraction
- [x] Frame quality filtering
- [x] Feature matching
- [x] Camera pose estimation
- [x] Triangulation
- [x] Point cloud visualization

### Phase 2 — Incremental Reconstruction
- [ ] Multi-frame reconstruction
- [ ] Camera trajectory estimation
- [ ] Global point cloud
- [ ] Point cloud merging
- [ ] Outlier rejection

### Phase 3 — Advanced Reconstruction
- [ ] Bundle adjustment
- [ ] Better camera calibration
- [ ] Dense reconstruction
- [ ] Improved scale estimation

### Phase 4 — Real-Time System
- [ ] Live camera input
- [ ] Real-time tracking
- [ ] Real-time mapping
- [ ] 3D environment visualization

## Status

🚧 Active Development

ARIA-S3D is currently an experimental research/prototype project.