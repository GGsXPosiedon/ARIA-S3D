# ARIA-S3D

### AI-powered Monocular 3D Reconstruction System

ARIA-S3D is an experimental computer vision system for reconstructing 3D environments from monocular video.

The project now includes an offline pipeline from source video through bundle-adjusted geometry, UV generation, photographic texture projection, and final asset validation.

## Current Pipeline

Video -> frame extraction -> quality filtering -> SIFT features -> feature matching -> camera pose estimation -> incremental reconstruction -> bundle adjustment -> mesh cleanup and optimization -> XAtlas UV generation -> photographic texture projection -> GLB/OBJ export and validation.

## Current Capabilities

- Extract and quality-filter frames from video
- Detect and match SIFT features
- Estimate camera poses and triangulate 3D points
- Run incremental reconstruction and bundle adjustment
- Reconstruct, clean, validate, and optimize triangle meshes
- Generate seam-aware XAtlas UVs
- Project photographic albedo from calibrated video frames
- Export textured GLB and OBJ assets
- Validate UVs, geometry, materials, embedded textures, and sidecars
- Support real-time tracking, mapping, and visualization components

## v0.5 Results

- 115,638 UV-atlas vertices and 129,470 triangles
- 78.9977% XAtlas atlas utilization
- 2,048 x 2,048 photographic albedo texture
- Textured GLB with an embedded PNG image and linked base-color material
- Textured OBJ with an MTL sidecar
- Phase 5.10 validation report with geometry and asset-integrity checks
- 29.48% non-black atlas coverage after visibility-aware photographic baking and bounded padding

The primary generated assets are written to `data/output/` and are intentionally excluded from Git because they are large and reproducible.

## Important Limitations

The system uses monocular vision, so reconstructed scale is arbitrary. Texture quality depends on calibration, camera coverage, mesh visibility, source image quality, exposure, and white balance. The current v0.5 texture atlas remains experimental and should be visually reviewed before production use.

## Technology Stack

- Python
- OpenCV
- Open3D
- NumPy
- Pillow
- XAtlas
- SIFT
- Bundle adjustment and epipolar geometry

## Roadmap

### Phase 1 — Prototype

- [x] Frame extraction
- [x] Frame quality filtering
- [x] Feature matching
- [x] Camera pose estimation
- [x] Triangulation
- [x] Point cloud visualization

### Phase 2 — Incremental Reconstruction

- [x] Multi-frame reconstruction
- [x] Camera trajectory estimation
- [x] Global point cloud
- [x] Point cloud merging
- [x] Outlier rejection

### Phase 3 — Advanced Reconstruction

- [x] Bundle adjustment
- [x] Better camera calibration integration
- [x] Surface reconstruction
- [ ] Improved absolute scale estimation

### Phase 4 — Real-Time System

- [x] Live camera input components
- [x] Real-time tracking
- [x] Real-time mapping
- [x] 3D environment visualization

### Phase 5 — Textured Reconstruction

- [x] Point-cloud preprocessing and downsampling
- [x] Surface reconstruction and mesh cleanup
- [x] Adaptive mesh optimization
- [x] Bundle-adjusted camera integration
- [x] XAtlas UV atlas generation
- [x] Calibrated photographic texture projection
- [x] Textured GLB and OBJ export
- [x] Final textured-model validation

### Post-v0.5 — Quality Improvements

- [ ] Improve visible texture coverage beyond the current 29.48%
- [ ] Add a cross-platform textured-model preview renderer
- [ ] Improve camera calibration and dense multi-view texture blending
- [ ] Add automated release artifact packaging

## Validation

From the project root, run:

```powershell
.venv\Scripts\python.exe reconstruction\validate_textured_model.py
```

## Status

v0.5 release candidate complete.

ARIA-S3D remains an experimental research/prototype project.
