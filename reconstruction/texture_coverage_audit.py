"""
ARIA-S3D | PHASE 6.2
TEXTURE COVERAGE AND SOURCE-VIDEO AUDIT

This diagnostic does not modify the textured model. It measures:
  - source-video quality for the selected projection frames
  - calibrated camera/image overlap
  - front-facing and depth-consistent mesh visibility
  - UV regions that are still untextured

Outputs are written to data/output/:
  texture_coverage_audit.txt
  texture_coverage_audit.npz
  texture_visibility_heatmap.png
"""

from pathlib import Path
import sys

import cv2
import numpy as np
import open3d as o3d

import texture_projection as projection


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "output"
TEXTURE = OUTPUT / "aria_albedo_texture.png"
REPORT = OUTPUT / "texture_coverage_audit.txt"
ARRAYS = OUTPUT / "texture_coverage_audit.npz"
HEATMAP = OUTPUT / "texture_visibility_heatmap.png"

DEPTH_SCALE = 4
DEPTH_TOLERANCE = 0.12
MIN_FACING = 0.05
HEATMAP_SIZE = 1024


def compute_visibility(vertices, triangles, camera_params, K, width, height):
    points = vertices[triangles]
    centers = points.mean(axis=1)
    edges_a = points[:, 1] - points[:, 0]
    edges_b = points[:, 2] - points[:, 0]
    normals = np.cross(edges_a, edges_b)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)

    faces = len(triangles)
    views_per_face = np.zeros(faces, dtype=np.int32)
    geometric_views = np.zeros(faces, dtype=np.int32)
    depth_rejects = 0
    camera_rows = []

    depth_width = max(1, int(np.ceil(width / DEPTH_SCALE)))
    depth_height = max(1, int(np.ceil(height / DEPTH_SCALE)))

    for params in camera_params:
        R = projection.rodrigues_to_matrix(params[:3])
        t = params[3:6]

        camera_vertices = (R @ vertices.T).T + t
        vertex_depth = camera_vertices[:, 2]
        vertex_projection = (K @ camera_vertices.T).T
        vertex_valid = vertex_depth > 1e-8
        vertex_uv = np.zeros((len(vertices), 2), dtype=np.float64)
        vertex_uv[vertex_valid] = (
            vertex_projection[vertex_valid, :2]
            / vertex_projection[vertex_valid, 2:3]
        )

        in_image_vertex = (
            vertex_valid
            & (vertex_uv[:, 0] >= 0)
            & (vertex_uv[:, 0] < width)
            & (vertex_uv[:, 1] >= 0)
            & (vertex_uv[:, 1] < height)
        )
        depth = np.full((depth_height, depth_width), np.inf, dtype=np.float32)
        depth_x = np.clip(
            (vertex_uv[in_image_vertex, 0] / DEPTH_SCALE).astype(np.int32),
            0,
            depth_width - 1,
        )
        depth_y = np.clip(
            (vertex_uv[in_image_vertex, 1] / DEPTH_SCALE).astype(np.int32),
            0,
            depth_height - 1,
        )
        np.minimum.at(
            depth,
            (depth_y, depth_x),
            vertex_depth[in_image_vertex].astype(np.float32),
        )
        finite = np.isfinite(depth)
        if np.any(finite):
            depth_work = depth.copy()
            depth_work[~finite] = float(np.max(depth[finite]))
            depth = cv2.erode(depth_work, np.ones((3, 3), np.uint8))

        camera_center = -R.T @ t
        to_camera = camera_center[None, :] - centers
        to_camera /= np.maximum(np.linalg.norm(to_camera, axis=1, keepdims=True), 1e-12)
        facing = np.sum(normals * to_camera, axis=1)

        camera_centers = (R @ centers.T).T + t
        center_depth = camera_centers[:, 2]
        center_projection = (K @ camera_centers.T).T
        center_uv = np.zeros((faces, 2), dtype=np.float64)
        center_valid = center_depth > 1e-8
        center_uv[center_valid] = (
            center_projection[center_valid, :2]
            / center_projection[center_valid, 2:3]
        )
        inside = (
            center_valid
            & (center_uv[:, 0] >= projection.IMAGE_BORDER_MARGIN)
            & (center_uv[:, 0] < width - projection.IMAGE_BORDER_MARGIN)
            & (center_uv[:, 1] >= projection.IMAGE_BORDER_MARGIN)
            & (center_uv[:, 1] < height - projection.IMAGE_BORDER_MARGIN)
        )
        geometric = inside & (facing >= MIN_FACING)

        depth_x = np.clip((center_uv[:, 0] / DEPTH_SCALE).astype(np.int32), 0, depth_width - 1)
        depth_y = np.clip((center_uv[:, 1] / DEPTH_SCALE).astype(np.int32), 0, depth_height - 1)
        depth_reference = depth[depth_y, depth_x]
        depth_consistent = (
            np.isfinite(depth_reference)
            & (center_depth <= depth_reference * (1.0 + DEPTH_TOLERANCE) + 1e-3)
        )
        visible = geometric & depth_consistent

        views_per_face += visible.astype(np.int32)
        geometric_views += geometric.astype(np.int32)
        depth_rejects += int(np.count_nonzero(geometric & ~depth_consistent))
        camera_rows.append(
            {
                "geometric_faces": int(np.count_nonzero(geometric)),
                "visible_faces": int(np.count_nonzero(visible)),
            }
        )

    return views_per_face, geometric_views, depth_rejects, camera_rows


def uv_area(uv_triangles):
    a = uv_triangles[:, 0]
    b = uv_triangles[:, 1]
    c = uv_triangles[:, 2]
    return 0.5 * np.abs(
        (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
        - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    )


def make_heatmap(uv_triangles, views_per_face):
    heatmap = np.zeros((HEATMAP_SIZE, HEATMAP_SIZE, 3), dtype=np.uint8)
    maximum = max(1, int(views_per_face.max()))

    for uv, views in zip(uv_triangles, views_per_face):
        if views <= 0:
            continue
        points = np.round(uv * (HEATMAP_SIZE - 1)).astype(np.int32)
        points[:, 1] = (HEATMAP_SIZE - 1) - points[:, 1]
        value = int(round(255.0 * views / maximum))
        color = cv2.applyColorMap(
            np.asarray([[value]], dtype=np.uint8),
            cv2.COLORMAP_TURBO,
        )[0, 0].tolist()
        cv2.fillConvexPoly(heatmap, points, color)

    cv2.imwrite(str(HEATMAP), heatmap)


def main():
    mesh_path = projection.MESH_FILE
    uv_path = projection.UV_FILE
    mesh = o3d.io.read_triangle_mesh(str(mesh_path), enable_post_processing=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.triangles, dtype=np.int64)
    uv_triangles = np.load(uv_path).reshape(-1, 3, 2).astype(np.float64)

    video_path = projection.load_video_path()
    video_info = projection.load_video_info(video_path)
    metadata = projection.load_video_metadata()
    ba = projection.load_ba_cameras()
    K, calibration_source = projection.build_intrinsics(
        video_info["width"],
        video_info["height"],
        metadata,
        ba_data={"K": ba["K"]} if ba["K"] is not None else ba["data"],
    )

    mapped_ids, mapping_name, mapping_results = projection.select_best_frame_mapping(
        video_path,
        ba["frame_ids"],
        video_info["total_frames"],
        metadata,
    )
    observations = projection.select_texture_observations(
        video_path,
        mapped_ids,
        video_info["width"],
        video_info["height"],
    )

    sharpness = np.asarray([obs["sharpness"] for obs in observations], dtype=np.float64)
    luminance = []
    dark_fraction = []
    for obs in observations:
        gray = cv2.cvtColor(obs["frame"], cv2.COLOR_BGR2GRAY)
        luminance.append(float(gray.mean()))
        dark_fraction.append(float(np.mean(gray < 10)))

    views, geometric_views, depth_rejects, camera_rows = compute_visibility(
        vertices,
        triangles,
        ba["camera_params"],
        K,
        video_info["width"],
        video_info["height"],
    )

    areas = uv_area(uv_triangles)
    total_area = max(float(areas.sum()), 1e-12)
    visible_area = float(areas[views > 0].sum() / total_area)
    geometric_area = float(areas[geometric_views > 0].sum() / total_area)

    texture = cv2.imread(str(TEXTURE), cv2.IMREAD_COLOR)
    if texture is None:
        raise RuntimeError(f"Could not read projected texture: {TEXTURE}")
    texture_mask = np.any(texture > 2, axis=2)
    center_uv = uv_triangles.mean(axis=1)
    center_x = np.clip(np.rint(center_uv[:, 0] * (texture.shape[1] - 1)).astype(np.int32), 0, texture.shape[1] - 1)
    center_y = np.clip(np.rint((1.0 - center_uv[:, 1]) * (texture.shape[0] - 1)).astype(np.int32), 0, texture.shape[0] - 1)
    textured_faces = texture_mask[center_y, center_x]

    make_heatmap(uv_triangles, views)
    np.savez_compressed(
        ARRAYS,
        views_per_face=views,
        geometric_views=geometric_views,
        uv_area=areas,
        textured_face_centers=textured_faces,
    )

    mapping_score = max(mapping_results, key=lambda x: x["score"])
    lines = [
        "ARIA-S3D | PHASE 6.2 TEXTURE COVERAGE AUDIT",
        "=" * 72,
        "",
        "SOURCE VIDEO",
        f"  Path                    : {video_path}",
        f"  Resolution              : {video_info['width']} x {video_info['height']}",
        f"  FPS / frames            : {video_info['fps']:.4f} / {video_info['total_frames']}",
        f"  Duration                : {video_info['duration']:.3f}s",
        f"  Selected frames         : {len(observations)}",
        f"  Mean sharpness          : {float(sharpness.mean()):.3f}",
        f"  Sharpness p10 / median  : {float(np.percentile(sharpness, 10)):.3f} / {float(np.median(sharpness)):.3f}",
        f"  Mean luminance          : {float(np.mean(luminance)):.3f}",
        f"  Mean dark-pixel fraction: {float(np.mean(dark_fraction)):.4f}",
        "",
        "CALIBRATION AND MAPPING",
        f"  Calibration source      : {calibration_source}",
        f"  Selected mapping        : {mapping_name}",
        f"  Best mapping score      : {mapping_score['score']:.4f}",
        f"  Best frame correspondence: {mapping_score['mean_correspondence']:.6f}",
        "",
        "MESH VISIBILITY",
        f"  Mesh triangles           : {len(triangles)}",
        f"  Faces with geometric view: {int(np.count_nonzero(geometric_views > 0))}",
        f"  Faces with visible view  : {int(np.count_nonzero(views > 0))}",
        f"  Faces with 3+ views      : {int(np.count_nonzero(views >= 3))}",
        f"  UV area geometrically viewable: {geometric_area:.6f} ({100*geometric_area:.2f}%)",
        f"  UV area depth-consistent : {visible_area:.6f} ({100*visible_area:.2f}%)",
        f"  Depth-consistency rejects: {depth_rejects}",
        f"  Mean views / visible face: {float(views[views > 0].mean()) if np.any(views > 0) else 0.0:.3f}",
        "",
        "TEXTURE RESULT",
        f"  Atlas file               : {TEXTURE}",
        f"  Atlas non-black coverage: {float(texture_mask.mean()):.6f} ({100*float(texture_mask.mean()):.2f}%)",
        f"  Textured face centers   : {int(np.count_nonzero(textured_faces))} / {len(textured_faces)}",
        f"  Visibility heatmap      : {HEATMAP}",
        f"  Audit arrays            : {ARRAYS}",
        "",
        "INTERPRETATION",
    ]

    if float(np.mean(sharpness)) < 250:
        lines.append("  Source video quality is a significant risk: selected frames are soft.")
    else:
        lines.append("  Source video sharpness is adequate for this diagnostic.")
    if visible_area < 0.50:
        lines.append("  Limited calibrated visibility is a primary cause of empty texture regions.")
    elif texture_mask.mean() < visible_area * 0.50:
        lines.append("  Baking efficiency is the primary cause: visible UV area is not becoming textured.")
    else:
        lines.append("  Texture coverage is broadly consistent with calibrated mesh visibility.")
    lines.append("  This audit does not change geometry, cameras, or the texture output.")

    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print("\n[OK] Coverage audit complete.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)
