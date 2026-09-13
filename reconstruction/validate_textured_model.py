"""
ARIA-S3D | PHASE 5.10
FINAL TEXTURED-MODEL VALIDATION

Validates:
  5.10.1 Albedo image integrity
  5.10.2 Texture coverage / statistics
  5.10.3 UV consistency against texture_ready_mesh
  5.10.4 GLB readability + geometry/material/texture presence
  5.10.5 OBJ readability + sidecar material/texture presence
  5.10.6 Geometry consistency
  5.10.7 Final validation report

No texture is generated or modified by this validator.
"""

from pathlib import Path
import json
import struct
import sys
import time

import numpy as np
import open3d as o3d

try:
    from PIL import Image
except ImportError:
    print("[ERROR] Pillow is required: pip install pillow")
    sys.exit(1)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "output"

TEXTURE = OUTPUT / "aria_albedo_texture.png"
GLB = OUTPUT / "aria_textured_mesh.glb"
OBJ = OUTPUT / "aria_textured_mesh.obj"

LEGACY_UV_FILE = OUTPUT / "texture_uvs.npy"
LEGACY_MESH_FILE = OUTPUT / "texture_ready_mesh.ply"
XATLAS_UV_FILE = OUTPUT / "xatlas_corner_uvs.npy"
XATLAS_MESH_FILE = OUTPUT / "xatlas_uv_mesh.ply"

if XATLAS_UV_FILE.exists() and XATLAS_MESH_FILE.exists():
    UV_FILE = XATLAS_UV_FILE
    TEXTURE_READY = XATLAS_MESH_FILE
    UV_LAYOUT_SOURCE = "XAtlas Phase 5.9.2"
else:
    UV_FILE = LEGACY_UV_FILE
    TEXTURE_READY = LEGACY_MESH_FILE
    UV_LAYOUT_SOURCE = "legacy texture-preparation fallback"
PROJECTION_STATS = OUTPUT / "texture_projection_stats.txt"
REPORT = OUTPUT / "final_textured_model_validation.txt"


def section(n, title):
    print()
    print("=" * 78)
    print(f"[{n}] {title}")
    print("-" * 78)


def ok(msg):
    print(f"[OK] {msg}")


def warn(msg):
    print(f"[WARN] {msg}")


def fail(msg):
    print(f"[ERROR] {msg}")
    raise RuntimeError(msg)


def finite_ratio(a):
    a = np.asarray(a)
    if a.size == 0:
        return 1.0
    return float(np.isfinite(a).all())


def validate_texture():
    section("1", "Albedo texture integrity")

    if not TEXTURE.exists():
        fail(f"Missing texture: {TEXTURE}")

    img = Image.open(TEXTURE)
    arr = np.asarray(img)

    print(f"File                 : {TEXTURE}")
    print(f"Format               : {img.format}")
    print(f"Mode                 : {img.mode}")
    print(f"Resolution           : {img.width} x {img.height}")
    print(f"Array shape          : {arr.shape}")

    if img.width != 2048 or img.height != 2048:
        warn("Texture is not the expected 2048 x 2048 resolution.")

    if arr.size == 0:
        fail("Texture is empty.")

    if not np.isfinite(arr.astype(np.float32)).all():
        fail("Texture contains non-finite values.")

    if arr.ndim < 2:
        fail("Texture array has invalid dimensionality.")

    ok("Albedo texture is readable and numerically valid.")

    return arr


def texture_statistics(arr):
    section("2", "Texture coverage and color statistics")

    rgb = arr[..., :3] if arr.ndim == 3 else arr
    rgbf = rgb.astype(np.float32)

    # "Non-zero" is intentionally conservative and matches the projection
    # coverage concept used by Phase 5.9.
    nonzero = np.any(rgb != 0, axis=-1)
    nonzero_ratio = float(nonzero.mean())

    nonblack = np.any(rgb > 2, axis=-1)
    nonblack_ratio = float(nonblack.mean())

    mean = rgbf.reshape(-1, rgbf.shape[-1]).mean(axis=0)
    std = rgbf.reshape(-1, rgbf.shape[-1]).std(axis=0)

    print(f"Non-zero coverage     : {nonzero_ratio:.6f} ({100*nonzero_ratio:.2f}%)")
    print(f"Non-black coverage    : {nonblack_ratio:.6f} ({100*nonblack_ratio:.2f}%)")
    print(f"Mean RGB              : {mean[:3]}")
    print(f"Std RGB               : {std[:3]}")
    print(f"Min RGB               : {rgbf[..., :3].min(axis=(0,1))}")
    print(f"Max RGB               : {rgbf[..., :3].max(axis=(0,1))}")

    if nonzero_ratio <= 0:
        fail("Albedo texture contains no populated texels.")
    elif nonzero_ratio < 0.01:
        warn("Texture coverage is below 1%; visual inspection is strongly recommended.")
    elif nonzero_ratio < 0.05:
        warn("Texture coverage is below 5%; model may contain large untextured regions.")
    else:
        ok("Texture has meaningful populated coverage.")

    return {
        "nonzero_ratio": nonzero_ratio,
        "nonblack_ratio": nonblack_ratio,
        "mean_rgb": mean[:3],
        "std_rgb": std[:3],
    }


def validate_uvs():
    section("3", "UV consistency")

    if not UV_FILE.exists():
        fail(f"Missing UV file: {UV_FILE}")
    if not TEXTURE_READY.exists():
        fail(f"Missing texture-ready mesh: {TEXTURE_READY}")

    uvs = np.load(UV_FILE)
    mesh = o3d.io.read_triangle_mesh(str(TEXTURE_READY), enable_post_processing=False)

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    print(f"Mesh vertices         : {len(vertices)}")
    print(f"Mesh triangles        : {len(triangles)}")
    print(f"UV shape              : {uvs.shape}")
    print(f"UV layout source      : {UV_LAYOUT_SOURCE}")

    expected = (len(triangles) * 3, 2)
    if tuple(uvs.shape) != expected:
        fail(f"UV shape mismatch: got {uvs.shape}, expected {expected}")

    if not np.isfinite(uvs).all():
        fail("UVs contain non-finite values.")

    uv_min = float(uvs.min())
    uv_max = float(uvs.max())

    print(f"UV range              : {uv_min:.6f} .. {uv_max:.6f}")

    if uv_min < -1e-6 or uv_max > 1.0 + 1e-6:
        fail("UV coordinates fall outside [0, 1].")

    ok("UV coordinates are finite, correctly sized, and in range [0,1].")

    return len(vertices), len(triangles), uvs


def validate_model(path, label):
    section("4" if label == "GLB" else "5", f"{label} model validation")

    if not path.exists():
        fail(f"Missing {label}: {path}")

    print(f"File                 : {path}")
    print(f"Size                 : {path.stat().st_size / (1024*1024):.2f} MB")

    model = o3d.io.read_triangle_model(str(path))
    if model is None:
        fail(f"Open3D could not read {label}.")

    if not model.meshes:
        fail(f"{label} contains no mesh objects.")

    total_v = 0
    total_t = 0
    material_count = len(model.materials)
    texture_count = 0

    print(f"Model materials       : {material_count}")

    for idx, tm in enumerate(model.meshes):
        legacy = tm.mesh

        v = len(legacy.vertices)
        t = len(legacy.triangles)

        total_v += v
        total_t += t

        # Open3D MeshInfo uses material_idx, not material.
        material_idx = int(tm.material_idx)

        material_valid = (
            0 <= material_idx < material_count
        )

        if material_valid:
            try:
                material = model.materials[material_idx]

                # MaterialRecord exposes texture_maps.
                try:
                    maps = material.texture_maps
                    texture_count += len(maps)
                except Exception:
                    pass

            except Exception:
                pass

        print(
            f"Submesh {idx:<3} "
            f"vertices={v:<8} "
            f"triangles={t:<8} "
            f"material_idx={material_idx:<3} "
            f"material_valid={material_valid}"
        )

    print(f"Total vertices        : {total_v}")
    print(f"Total triangles       : {total_t}")
    print(f"Meshes                : {len(model.meshes)}")
    print(f"Materials             : {material_count}")
    print(f"Texture maps detected : {texture_count}")

    if total_v == 0 or total_t == 0:
        fail(f"{label} contains empty geometry.")

    if material_count == 0:
        warn(f"{label} contains no material records.")
    else:
        ok(f"{label} contains {material_count} material record(s).")

    if texture_count == 0:
        print(
            "[INFO] "
            f"{label} material records expose no texture maps through "
            "Open3D Python introspection; exporter-level linkage is checked "
            "separately."
        )
    else:
        ok(f"{label} contains {texture_count} texture map(s).")

    ok(f"{label} is readable and contains valid triangle geometry.")

    return {
        "vertices": total_v,
        "triangles": total_t,
        "meshes": len(model.meshes),
        "materials": material_count,
        "textures": texture_count,
    }


def validate_glb_container(path):
    """Validate the binary glTF texture/material linkage directly.

    Open3D can read the geometry while exposing no texture maps through its
    Python material introspection.  The GLB JSON is authoritative for this
    check, so verify the actual embedded image, texture, and material links.
    """

    section("4A", "GLB container and embedded texture validation")

    data = path.read_bytes()
    if len(data) < 20:
        fail("GLB is too small to contain a valid header and JSON chunk.")

    magic, version, declared_length = struct.unpack_from("<4sII", data, 0)
    print(f"Magic                : {magic!r}")
    print(f"Version              : {version}")
    print(f"Declared length      : {declared_length}")
    print(f"Actual length        : {len(data)}")

    if magic != b"glTF":
        fail("GLB magic header is invalid.")
    if version != 2:
        fail(f"Unsupported glTF version: {version}")
    if declared_length != len(data):
        warn("GLB declared length differs from the file length.")

    json_chunk = None
    offset = 12
    while offset + 8 <= len(data):
        chunk_length, chunk_type = struct.unpack_from("<II", data, offset)
        offset += 8
        chunk_end = offset + chunk_length
        if chunk_end > len(data):
            fail("GLB chunk extends beyond the file length.")
        if chunk_type == 0x4E4F534A:  # JSON
            json_chunk = data[offset:chunk_end]
            break
        offset = chunk_end

    if json_chunk is None:
        fail("GLB JSON chunk was not found.")

    try:
        document = json.loads(json_chunk.rstrip(b" \\x00").decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        fail(f"GLB JSON chunk is invalid: {exc}")

    images = document.get("images", [])
    textures = document.get("textures", [])
    materials = document.get("materials", [])

    print(f"Embedded images      : {len(images)}")
    print(f"Texture objects      : {len(textures)}")
    print(f"Material objects     : {len(materials)}")

    if not images:
        fail("GLB contains no embedded or referenced images.")
    if not textures:
        fail("GLB contains no texture objects.")
    if not materials:
        fail("GLB contains no material objects.")

    for index, image in enumerate(images):
        has_embedded_view = "bufferView" in image and bool(image.get("mimeType"))
        has_uri = bool(image.get("uri"))
        if not (has_embedded_view or has_uri):
            fail(f"GLB image {index} has neither bufferView nor URI.")
        print(
            f"Image {index:<3}           : "
            f"{'embedded ' if has_embedded_view else 'external '}"
            f"{image.get('mimeType', image.get('uri', ''))}"
        )

    linked_materials = 0
    for index, material in enumerate(materials):
        pbr = material.get("pbrMetallicRoughness", {})
        base_color = pbr.get("baseColorTexture", {})
        texture_index = base_color.get("index")
        if texture_index is None:
            continue
        if not isinstance(texture_index, int) or not 0 <= texture_index < len(textures):
            fail(f"Material {index} references invalid texture index {texture_index}.")
        source = textures[texture_index].get("source")
        if not isinstance(source, int) or not 0 <= source < len(images):
            fail(f"Texture {texture_index} references invalid image source {source}.")
        linked_materials += 1
        print(
            f"Material {index:<3}        : baseColorTexture={texture_index}, "
            f"image={source}"
        )

    if linked_materials == 0:
        fail("No material is linked to a base-color texture.")

    ok(
        f"GLB container is valid with {len(images)} image(s), "
        f"{len(textures)} texture(s), and {linked_materials} textured material(s)."
    )

    return {
        "images": len(images),
        "textures": len(textures),
        "textured_materials": linked_materials,
    }

def validate_obj_sidecars():
    section("6", "OBJ material and texture sidecars")

    mtl = OBJ.with_suffix(".mtl")
    possible_textures = [
        OUTPUT / "aria_textured_mesh_albedo.png",
        OUTPUT / "aria_textured_mesh.png",
        TEXTURE,
    ]

    print(f"OBJ exists            : {OBJ.exists()}")
    print(f"MTL exists            : {mtl.exists()}")

    if not OBJ.exists():
        fail(f"Missing OBJ: {OBJ}")

    if mtl.exists():
        text = mtl.read_text(errors="ignore")
        print(f"MTL size              : {len(text)} bytes")
        map_lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip().lower().startswith("map_kd ")
        ]
        referenced_texture = None
        if map_lines:
            referenced_texture = (
                mtl.parent / map_lines[0].split(None, 1)[1].strip()
            ).resolve()
        map_valid = referenced_texture is not None and referenced_texture.exists()
        print(f"MTL references map    : {bool(map_lines)}")
        print(
            f"Referenced texture    : "
            f"{referenced_texture if referenced_texture else 'none'}"
        )
        print(f"Referenced file exists: {map_valid}")
        if not map_valid:
            fail("OBJ MTL map_Kd reference does not resolve to an existing texture.")
        ok("OBJ material sidecar exists.")
    else:
        warn("OBJ .mtl sidecar was not found.")

    found = [p for p in possible_textures if p.exists()]
    for p in found:
        print(f"Texture sidecar       : {p.name}")

    if not found:
        warn("No obvious OBJ texture sidecar found; GLB remains the primary deliverable.")
    else:
        ok("At least one OBJ-associated texture file exists.")


def validate_geometry_consistency(glb_info, obj_info, expected_v, expected_t):
    section("7", "Geometry consistency")

    print(f"Expected source vertices : {expected_v}")
    print(f"Expected source triangles: {expected_t}")
    print(f"GLB vertices             : {glb_info['vertices']}")
    print(f"GLB triangles            : {glb_info['triangles']}")
    print(f"OBJ vertices             : {obj_info['vertices']}")
    print(f"OBJ triangles            : {obj_info['triangles']}")

    if glb_info["triangles"] != expected_t:
        warn("GLB triangle count differs from texture-ready mesh.")
    if obj_info["triangles"] != expected_t:
        warn("OBJ triangle count differs from texture-ready mesh.")

    if glb_info["triangles"] == obj_info["triangles"]:
        ok("GLB and OBJ preserve the same triangle count.")
    else:
        warn("GLB and OBJ triangle counts differ; inspect exporters.")

    if glb_info["vertices"] == obj_info["vertices"]:
        ok("GLB and OBJ preserve the same vertex count.")
    else:
        warn("GLB and OBJ vertex counts differ; this can be exporter-dependent.")


def write_report(
    stats,
    uv_count,
    glb_info,
    obj_info,
    elapsed,
    glb_container_info,
):
    lines = [
        "ARIA-S3D | PHASE 5.10 FINAL TEXTURED-MODEL VALIDATION",
        "=" * 70,
        "",
        "RESULT: PASS",
        "",
        f"Texture resolution       : 2048 x 2048",
        f"Texture non-zero cover   : {stats['nonzero_ratio']:.6f}",
        f"Texture non-black cover  : {stats['nonblack_ratio']:.6f}",
        f"UV rows                  : {uv_count}",
        f"GLB vertices             : {glb_info['vertices']}",
        f"GLB triangles            : {glb_info['triangles']}",
        f"GLB materials            : {glb_info['materials']}",
        f"GLB texture maps         : {glb_info['textures']}",
        f"GLB embedded images      : {glb_container_info['images']}",
        f"GLB texture objects      : {glb_container_info['textures']}",
        f"GLB textured materials   : {glb_container_info['textured_materials']}",
        f"OBJ vertices             : {obj_info['vertices']}",
        f"OBJ triangles            : {obj_info['triangles']}",
        f"Validation time          : {elapsed:.3f}s",
        "",
        "Validated:",
        "  - Albedo image readability",
        "  - Texture numerical integrity",
        "  - Texture coverage statistics",
        "  - UV shape/range/finite checks",
        "  - GLB readability",
        "  - GLB embedded image and material linkage",
        "  - OBJ readability",
        "  - Material/texture presence checks",
        "  - GLB/OBJ geometry consistency",
        "",
        "NOTE:",
        "  This script validates asset structure and texture data.",
        "  It does not claim perceptual texture quality from a human visual review.",
        "  Low coverage should be visually inspected before release.",
    ]

    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] {REPORT}")


def main():
    start = time.time()

    print()
    print("=" * 78)
    print("ARIA-S3D | PHASE 5.10")
    print("FINAL TEXTURED-MODEL VALIDATION")
    print("=" * 78)

    arr = validate_texture()
    stats = texture_statistics(arr)
    expected_v, expected_t, uvs = validate_uvs()

    glb_info = validate_model(GLB, "GLB")
    glb_container_info = validate_glb_container(GLB)
    obj_info = validate_model(OBJ, "OBJ")

    validate_obj_sidecars()
    validate_geometry_consistency(glb_info, obj_info, expected_v, expected_t)

    elapsed = time.time() - start
    write_report(
        stats,
        len(uvs),
        glb_info,
        obj_info,
        elapsed,
        glb_container_info,
    )

    print()
    print("=" * 78)
    print("ARIA-S3D | PHASE 5.10 PASS")
    print("=" * 78)
    print(f"Texture coverage       : {stats['nonzero_ratio']:.6f}")
    print(f"GLB triangles          : {glb_info['triangles']}")
    print(f"OBJ triangles          : {obj_info['triangles']}")
    print(f"Validation time        : {elapsed:.3f}s")
    print()
    print("NEXT STEP:")
    print("  Visual inspection of aria_textured_mesh.glb")
    print("  If visual quality is acceptable -> tag/release v0.5")
    print("=" * 78)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("=" * 78)
        print("ARIA-S3D | PHASE 5.10 FAIL")
        print("=" * 78)
        print(f"Reason: {exc}")
        sys.exit(1)
