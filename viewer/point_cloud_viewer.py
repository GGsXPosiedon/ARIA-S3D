import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

POINT_CLOUD = Path("data/output/filtered_point_cloud.xyz")


# ============================================================
# LOAD POINT CLOUD
# ============================================================

if not POINT_CLOUD.exists():
    raise FileNotFoundError(
        f"Point cloud not found: {POINT_CLOUD}"
    )


points = np.loadtxt(POINT_CLOUD)

if points.ndim == 1:
    points = points.reshape(1, -1)


x = points[:, 0]
y = points[:, 1]
z = points[:, 2]


print("=" * 60)
print("ARIA-S3D | 3D POINT CLOUD VIEWER")
print("=" * 60)

print(f"Points loaded: {len(points)}")

print()
print("X range:", x.min(), "to", x.max())
print("Y range:", y.min(), "to", y.max())
print("Z range:", z.min(), "to", z.max())


# ============================================================
# 3D VISUALIZATION
# ============================================================

fig = plt.figure(figsize=(12, 8))

ax = fig.add_subplot(111, projection="3d")

ax.scatter(
    x,
    y,
    z,
    s=2,
    alpha=0.7
)


# ============================================================
# CAMERA / AXIS LABELS
# ============================================================

ax.set_title(
    "ARIA-S3D — Reconstructed 3D Point Cloud"
)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")


# ============================================================
# DISPLAY
# ============================================================

plt.tight_layout()

plt.show()