import numpy as np
from pathlib import Path


# ============================================================
# ARIA-S3D | OUTLIER REJECTION
# Removes isolated / unreliable points from the global cloud
# ============================================================

INPUT_FILE = Path("data/output/global_point_cloud.xyz")
OUTPUT_FILE = Path("data/output/filtered_point_cloud.xyz")


def load_point_cloud(path):
    """Load XYZ point cloud."""

    if not path.exists():
        raise FileNotFoundError(f"Point cloud not found: {path}")

    points = np.loadtxt(path)

    # Handle single-point files safely
    if points.ndim == 1:
        points = points.reshape(1, -1)

    # Keep only XYZ
    points = points[:, :3]

    return points


def statistical_outlier_removal(points, k=12, std_ratio=2.0):
    """
    Remove points that are significantly farther from their
    local neighbors than the rest of the cloud.

    k:
        Number of neighboring points considered.

    std_ratio:
        Controls how aggressive the filtering is.
    """

    print(f"Analyzing {len(points)} points...")
    print(f"Neighbor count: {k}")
    print(f"Standard deviation ratio: {std_ratio}")

    if len(points) <= k:
        print("Not enough points for outlier rejection.")
        return points

    # --------------------------------------------------------
    # Compute pairwise distances
    # --------------------------------------------------------

    print("\nComputing local point distances...")

    # Chunked processing prevents excessive RAM usage
    distances = np.zeros(len(points))

    chunk_size = 500

    for start in range(0, len(points), chunk_size):

        end = min(start + chunk_size, len(points))

        chunk = points[start:end]

        diff = chunk[:, None, :] - points[None, :, :]

        dist = np.sqrt(np.sum(diff ** 2, axis=2))

        # Ignore the point itself
        dist[dist == 0] = np.inf

        # k nearest neighbors
        nearest = np.partition(
            dist,
            k,
            axis=1
        )[:, :k]

        distances[start:end] = np.mean(nearest, axis=1)

    # --------------------------------------------------------
    # Calculate statistical threshold
    # --------------------------------------------------------

    mean_distance = np.mean(distances)
    std_distance = np.std(distances)

    threshold = mean_distance + std_ratio * std_distance

    print("\nDistance statistics:")
    print(f"Mean neighbor distance : {mean_distance:.6f}")
    print(f"Std deviation          : {std_distance:.6f}")
    print(f"Outlier threshold      : {threshold:.6f}")

    # --------------------------------------------------------
    # Keep valid points
    # --------------------------------------------------------

    mask = distances <= threshold

    filtered_points = points[mask]

    removed = len(points) - len(filtered_points)

    print("\nOutlier rejection:")
    print(f"Original points : {len(points)}")
    print(f"Removed points  : {removed}")
    print(f"Remaining points: {len(filtered_points)}")

    return filtered_points


def save_point_cloud(points, path):
    """Save XYZ point cloud."""

    path.parent.mkdir(parents=True, exist_ok=True)

    np.savetxt(
        path,
        points,
        fmt="%.6f"
    )

    print(f"\nFiltered point cloud saved to:")
    print(path)


def print_statistics(points, title):
    """Print point cloud statistics."""

    minimum = np.min(points, axis=0)
    maximum = np.max(points, axis=0)
    mean = np.mean(points, axis=0)

    print(f"\n{title}")
    print("-" * 55)

    print(
        f"X range: {minimum[0]:.4f} to {maximum[0]:.4f}"
    )

    print(
        f"Y range: {minimum[1]:.4f} to {maximum[1]:.4f}"
    )

    print(
        f"Z range: {minimum[2]:.4f} to {maximum[2]:.4f}"
    )

    print(
        f"Mean XYZ: "
        f"[{mean[0]:.4f}, "
        f"{mean[1]:.4f}, "
        f"{mean[2]:.4f}]"
    )


def main():

    print("=" * 65)
    print("ARIA-S3D | OUTLIER REJECTION")
    print("=" * 65)

    # --------------------------------------------------------
    # Load cloud
    # --------------------------------------------------------

    print("\nLoading global point cloud:")
    print(INPUT_FILE)

    points = load_point_cloud(INPUT_FILE)

    print(f"\nPoints loaded: {len(points)}")

    print_statistics(
        points,
        "ORIGINAL POINT CLOUD STATISTICS"
    )

    # --------------------------------------------------------
    # Remove outliers
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("PERFORMING STATISTICAL OUTLIER REJECTION")
    print("=" * 65)

    filtered_points = statistical_outlier_removal(
        points,
        k=12,
        std_ratio=2.0
    )

    # --------------------------------------------------------
    # Save result
    # --------------------------------------------------------

    save_point_cloud(
        filtered_points,
        OUTPUT_FILE
    )

    # --------------------------------------------------------
    # Final statistics
    # --------------------------------------------------------

    print_statistics(
        filtered_points,
        "FILTERED POINT CLOUD STATISTICS"
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    reduction = (
        (len(points) - len(filtered_points))
        / len(points)
        * 100
    )

    print("\n" + "=" * 65)
    print("OUTLIER REJECTION COMPLETE")
    print("=" * 65)

    print(f"Input points       : {len(points)}")
    print(f"Output points      : {len(filtered_points)}")
    print(f"Points removed     : {len(points) - len(filtered_points)}")
    print(f"Point reduction    : {reduction:.2f}%")

    print("\nIMPORTANT:")
    print("The filtered cloud removes statistically isolated points.")
    print("This improves the reliability of the reconstructed scene.")

    print("\nNext step:")
    print("Phase 2 will be complete after validating the filtered cloud.")


if __name__ == "__main__":
    main()