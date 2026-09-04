import os
import time
import numpy as np

try:
    import open3d as o3d
except ImportError:
    print("=" * 70)
    print("ARIA-S3D | PHASE 4.5")
    print("LIVE 3D VISUALIZATION")
    print("=" * 70)
    print()
    print("[ERROR] Open3D is not installed.")
    print()
    print("Install it with:")
    print("pip install open3d")
    print()
    print("Then run this script again.")
    exit()


# ============================================================
# ARIA-S3D | PHASE 4.5
# LIVE 3D VISUALIZATION
# ============================================================

OUTPUT_DIR = "data/output"

POINT_CLOUD_FILE = os.path.join(
    OUTPUT_DIR,
    "realtime_mapped_point_cloud.xyz"
)

TRAJECTORY_FILE = os.path.join(
    OUTPUT_DIR,
    "realtime_mapped_trajectory.txt"
)

WINDOW_NAME = "ARIA-S3D | Live 3D Visualization"


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("ARIA-S3D | PHASE 4.5")
print("LIVE 3D VISUALIZATION")
print("=" * 70)
print()


# ============================================================
# CHECK INPUT FILES
# ============================================================

print("[1] Checking Phase 4.4 outputs")
print("-" * 70)

if not os.path.exists(POINT_CLOUD_FILE):
    print("[ERROR] Mapped point cloud not found:")
    print(f"        {POINT_CLOUD_FILE}")
    exit()

print("[OK] Mapped point cloud")
print(f"     {POINT_CLOUD_FILE}")


if not os.path.exists(TRAJECTORY_FILE):
    print("[ERROR] Mapped trajectory not found:")
    print(f"        {TRAJECTORY_FILE}")
    exit()

print("[OK] Mapped trajectory")
print(f"     {TRAJECTORY_FILE}")


# ============================================================
# LOAD POINT CLOUD
# ============================================================

print()
print("[2] Loading mapped point cloud")
print("-" * 70)

try:
    points = np.loadtxt(
        POINT_CLOUD_FILE,
        dtype=np.float64
    )
except Exception as e:
    print("[ERROR] Could not load point cloud.")
    print(f"        {e}")
    exit()


# ------------------------------------------------------------
# Normalize shape
# ------------------------------------------------------------

if points.ndim == 1:

    if points.size != 3:
        print("[ERROR] Invalid point cloud format.")
        exit()

    points = points.reshape(1, 3)


if points.shape[1] < 3:
    print("[ERROR] Point cloud must contain X Y Z coordinates.")
    exit()


points = points[:, :3]


# ------------------------------------------------------------
# Remove invalid points
# ------------------------------------------------------------

valid_mask = np.isfinite(points).all(axis=1)

points = points[valid_mask]


if len(points) == 0:
    print("[ERROR] No valid 3D points found.")
    exit()


print(f"3D points loaded : {len(points)}")

print(
    f"X range          : "
    f"{points[:, 0].min():.6f} -> {points[:, 0].max():.6f}"
)

print(
    f"Y range          : "
    f"{points[:, 1].min():.6f} -> {points[:, 1].max():.6f}"
)

print(
    f"Z range          : "
    f"{points[:, 2].min():.6f} -> {points[:, 2].max():.6f}"
)


# ============================================================
# LOAD CAMERA TRAJECTORY
# ============================================================

print()
print("[3] Loading camera trajectory")
print("-" * 70)

try:

    trajectory = np.loadtxt(
        TRAJECTORY_FILE,
        dtype=np.float64
    )

except Exception as e:

    print("[ERROR] Could not load trajectory.")
    print(f"        {e}")
    exit()


# ------------------------------------------------------------
# Normalize trajectory shape
# ------------------------------------------------------------

if trajectory.ndim == 1:

    if trajectory.size < 4:
        print("[ERROR] Invalid trajectory format.")
        exit()

    trajectory = trajectory.reshape(1, -1)


if trajectory.shape[1] >= 4:

    camera_positions = trajectory[:, 1:4]

else:

    if trajectory.shape[1] == 3:
        camera_positions = trajectory[:, :3]

    else:
        print("[ERROR] Invalid trajectory format.")
        exit()


# ------------------------------------------------------------
# Remove invalid camera positions
# ------------------------------------------------------------

valid_camera_mask = np.isfinite(
    camera_positions
).all(axis=1)

camera_positions = camera_positions[
    valid_camera_mask
]


print(
    f"Camera positions : "
    f"{len(camera_positions)}"
)


# ============================================================
# CREATE OPEN3D POINT CLOUD
# ============================================================

print()
print("[4] Creating 3D point cloud")
print("-" * 70)

pcd = o3d.geometry.PointCloud()

pcd.points = o3d.utility.Vector3dVector(
    points
)


# ------------------------------------------------------------
# Downsample if extremely large
# ------------------------------------------------------------

MAX_DISPLAY_POINTS = 100000

if len(points) > MAX_DISPLAY_POINTS:

    print(
        f"[INFO] Point cloud contains {len(points)} points."
    )

    print(
        f"[INFO] Downsampling for visualization to "
        f"{MAX_DISPLAY_POINTS} points."
    )

    step = max(
        1,
        len(points) // MAX_DISPLAY_POINTS
    )

    display_points = points[::step]

    pcd.points = o3d.utility.Vector3dVector(
        display_points
    )

    print(
        f"[OK] Display points : "
        f"{len(display_points)}"
    )


# ============================================================
# CREATE CAMERA TRAJECTORY
# ============================================================

print()
print("[5] Creating camera trajectory")
print("-" * 70)

trajectory_lines = []

if len(camera_positions) >= 2:

    for i in range(
        len(camera_positions) - 1
    ):

        trajectory_lines.append(
            [i, i + 1]
        )


trajectory_line_set = o3d.geometry.LineSet()

if len(camera_positions) > 0:

    trajectory_line_set.points = (
        o3d.utility.Vector3dVector(
            camera_positions
        )
    )

if len(trajectory_lines) > 0:

    trajectory_line_set.lines = (
        o3d.utility.Vector2iVector(
            trajectory_lines
        )
    )


print(
    f"Trajectory segments : "
    f"{len(trajectory_lines)}"
)


# ============================================================
# CAMERA POSITION MARKER
# ============================================================

camera_marker = o3d.geometry.TriangleMesh.create_sphere(
    radius=0.35
)

if len(camera_positions) > 0:

    camera_marker.translate(
        camera_positions[-1]
    )


# ============================================================
# COORDINATE FRAME
# ============================================================

coordinate_frame = (
    o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=2.0,
        origin=[0, 0, 0]
    )
)


# ============================================================
# VISUALIZER
# ============================================================

print()
print("[6] Starting Open3D visualization")
print("-" * 70)

print()
print("Controls:")
print("  Left mouse   : Rotate")
print("  Right mouse  : Pan")
print("  Mouse wheel  : Zoom")
print("  R            : Reset view")
print("  Q / ESC      : Close")
print()


vis = o3d.visualization.Visualizer()

vis.create_window(
    window_name=WINDOW_NAME,
    width=1280,
    height=720
)


# ============================================================
# ADD GEOMETRY
# ============================================================

vis.add_geometry(pcd)

if len(trajectory_lines) > 0:

    vis.add_geometry(
        trajectory_line_set
    )

vis.add_geometry(
    camera_marker
)

vis.add_geometry(
    coordinate_frame
)


# ============================================================
# RENDER SETTINGS
# ============================================================

render_option = vis.get_render_option()

render_option.point_size = 2.0

render_option.background_color = np.array(
    [0.02, 0.02, 0.02]
)


# ============================================================
# INITIAL CAMERA VIEW
# ============================================================

view_control = vis.get_view_control()

view_control.set_zoom(0.8)


# ============================================================
# VISUALIZATION LOOP
# ============================================================

print("=" * 70)
print("LIVE 3D VISUALIZATION STARTED")
print("=" * 70)

start_time = time.time()

while True:

    # --------------------------------------------------------
    # Update visualization
    # --------------------------------------------------------

    if not vis.poll_events():

        break

    vis.update_renderer()

    time.sleep(0.01)


# ============================================================
# SHUTDOWN
# ============================================================

vis.destroy_window()


# ============================================================
# FINAL REPORT
# ============================================================

elapsed_time = time.time() - start_time

print()
print("=" * 70)
print("ARIA-S3D | PHASE 4.5 COMPLETE")
print("=" * 70)

print()
print(
    f"Displayed 3D points : "
    f"{len(pcd.points)}"
)

print(
    f"Camera positions    : "
    f"{len(camera_positions)}"
)

print(
    f"Trajectory segments : "
    f"{len(trajectory_lines)}"
)

print(
    f"Visualization time  : "
    f"{elapsed_time:.2f} seconds"
)

print()
print("Generated / visualized:")
print(
    f"  Point cloud : "
    f"{POINT_CLOUD_FILE}"
)

print(
    f"  Trajectory  : "
    f"{TRAJECTORY_FILE}"
)

print()
print("IMPORTANT:")
print("This visualization represents a monocular reconstruction.")
print("Absolute metric scale remains arbitrary.")

print()
print("NEXT STEP:")
print("Phase 4.6 - Tracking-loss recovery")

print("=" * 70)