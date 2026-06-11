import numpy as np
from pxr import UsdGeom, Usd, Vt
from PIL import Image
from isaacsim.core.utils import prims as prims_utils

def collect_points(prim: Usd.Prim):
    points_list = []

    for sub_prim in Usd.PrimRange(prim):
        if not sub_prim.IsA(UsdGeom.Mesh):
            continue

        mesh = UsdGeom.Mesh(sub_prim)
        points: Vt.Vec3fArray = mesh.GetPointsAttr().Get()
        if points is None:
            continue
        for p in points:
            points_list.append([p[0], p[1], p[2]])

    return np.asarray(points_list, dtype=np.float32)


def points_to_grid(points: np.ndarray, axes=("x", "z"), cell_size=0.005, padding=0.02):
    axis_id = {"x": 0, "y": 1, "z": 2}
    i0 = axis_id[axes[0]]
    i1 = axis_id[axes[1]]

    uv: np.ndarray = points[:, [i0, i1]]

    uv_min: np.ndarray = uv.min(axis=0) - padding
    uv_max = uv.max(axis=0) + padding

    size = np.ceil((uv_max - uv_min) / cell_size).astype(int)
    w, h = int(size[0]), int(size[1])

    grid = np.zeros((h, w), dtype=np.uint8)

    pix = ((uv - uv_min) / cell_size).astype(int)
    pix[:, 0] = np.clip(pix[:, 0], 0, w - 1)
    pix[:, 1] = np.clip(pix[:, 1], 0, h - 1)

    # image y 方向通常要 flip
    grid[h - 1 - pix[:, 1], pix[:, 0]] = 255

    return Image.fromarray(grid), uv_min, uv_max


def pixel_to_world_uv(row: int, col: int, meta: dict, center: bool = False):
    """
    Convert image pixel coordinate to projected real-world coordinate.

    Args:
        row, col:
            image coordinate, row downward, col rightward.
        meta:
            metadata returned by project_points_to_mask().
        center:
            If True, return the coordinate of pixel center.
            If False, return the coordinate of pixel corner/grid index.

    Returns:
        uv_world:
            np.ndarray, shape = (2,)
            Coordinate along meta["axes"], e.g. axes=("x", "z") -> [x, z].
    """
    cell_size = meta["cell_size"]
    uv_min = meta["uv_min"]
    height = meta["height"]

    offset = 0.5 if center else 0.0

    u = uv_min[0] + (col + offset) * cell_size

    # because image row is flipped
    v_index = height - 1 - row
    v = uv_min[1] + (v_index + offset) * cell_size

    return u, v

import math
import numpy as np

from pxr import Usd, UsdGeom
import cv2


def collect_world_triangles(prim_path: str, time=Usd.TimeCode.Default()):
    """
    Collect all mesh triangles under root_prim_path and transform them to world coordinates.

    Returns:
        triangles: np.ndarray, shape = (N, 3, 3)
                   N triangles, each triangle has 3 vertices, each vertex is xyz.
    """
    prim = prims_utils.get_prim_at_path(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Invalid prim path: {prim_path}")

    all_triangles = []

    for sub_prim in Usd.PrimRange(prim):
        if not sub_prim.IsA(UsdGeom.Mesh):
            continue

        mesh = UsdGeom.Mesh(sub_prim)

        points = mesh.GetPointsAttr().Get(time)
        face_vertex_counts = mesh.GetFaceVertexCountsAttr().Get(time)
        face_vertex_indices = mesh.GetFaceVertexIndicesAttr().Get(time)

        if points is None or face_vertex_counts is None or face_vertex_indices is None:
            continue

        local_to_world = UsdGeom.Xformable(sub_prim).ComputeLocalToWorldTransform(time)

        # Convert local mesh points to world points
        world_points = []
        for p in points:
            wp = local_to_world.Transform(p)
            world_points.append([float(wp[0]), float(wp[1]), float(wp[2])])

        world_points = np.asarray(world_points, dtype=np.float64)

        # Triangulate faces by fan triangulation:
        # polygon [v0, v1, v2, v3] -> triangles:
        # [v0, v1, v2], [v0, v2, v3]
        offset = 0
        for count in face_vertex_counts:
            count = int(count)
            indices = face_vertex_indices[offset: offset + count]
            offset += count

            if count < 3:
                continue

            v0 = int(indices[0])
            for i in range(1, count - 1):
                v1 = int(indices[i])
                v2 = int(indices[i + 1])

                tri = np.stack([world_points[v0], world_points[v1], world_points[v2]], axis=0)
                all_triangles.append(tri)

    if len(all_triangles) == 0:
        raise RuntimeError(f"No mesh triangles found under {prim_path}")

    return np.asarray(all_triangles, dtype=np.float64)


def triangle_areas(triangles: np.ndarray):
    """
    triangles: shape = (N, 3, 3)
    """
    v0 = triangles[:, 0, :]
    v1 = triangles[:, 1, :]
    v2 = triangles[:, 2, :]
    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    return areas


def sample_triangle_surface(triangles: np.ndarray, cell_size: float, sample_density: float = 2.0,
                            min_samples_per_triangle=3, max_samples_per_triangle=50000, seed=0):
    """
    Uniformly sample points on triangle surfaces.
    Args:
        triangles: shape = (N, 3, 3)

        cell_size: output grid resolution in world units, e.g. 0.005 meter.

        sample_density:
            roughly controls samples per cell area.
            samples_per_m2 = sample_density / (cell_size ** 2)

            For example:
                cell_size = 0.005
                sample_density = 2.0
                samples_per_m2 = 80000

        min_samples_per_triangle: ensure small triangles are still represented.
        max_samples_per_triangle: avoid extremely large faces causing too many samples.

    Returns:
        sampled_points: np.ndarray, shape = (M, 3)
    """
    rng = np.random.default_rng(seed)

    areas = triangle_areas(triangles)
    samples_per_m2 = sample_density / (cell_size * cell_size)

    sampled = []

    for tri, area in zip(triangles, areas):
        if area <= 1e-12:
            continue

        n = int(math.ceil(area * samples_per_m2))
        n = max(n, min_samples_per_triangle)
        n = min(n, max_samples_per_triangle)

        v0, v1, v2 = tri

        # Uniform sampling on triangle using barycentric coordinates
        r1 = rng.random(n)
        r2 = rng.random(n)

        sqrt_r1 = np.sqrt(r1)

        a = 1.0 - sqrt_r1
        b = sqrt_r1 * (1.0 - r2)
        c = sqrt_r1 * r2

        pts = (a[:, None] * v0[None, :] + b[:, None] * v1[None, :] + c[:, None] * v2[None, :])
        sampled.append(pts)

    if len(sampled) == 0:
        raise RuntimeError("No valid surface samples generated.")

    sampled_points = np.concatenate(sampled, axis=0)

    # Add all triangle vertices too, to preserve sharp corners
    vertices = triangles.reshape(-1, 3)
    sampled_points = np.concatenate([sampled_points, vertices], axis=0)

    return sampled_points


def project_points_to_mask(points: np.ndarray, axes=("x", "z"), cell_size=0.005, padding=0.02, dilate_px=1):
    """
    Project 3D sampled points to a 2D binary occupancy mask.
    Args:
        points: shape = (N, 3)
        axes:
            ("x", "y") -> XY top view
            ("x", "z") -> XZ front view
            ("y", "z") -> YZ side view
        cell_size: pixel size in world units.
        padding: world-unit margin around object.
        dilate_px: optional dilation to close tiny sampling holes.
    Returns:
        mask: np.uint8 image, occupied = 255, free = 0
        meta: dict with uv_min, uv_max, cell_size, axes
    """
    axis_id = {"x": 0, "y": 1, "z": 2}

    i0 = axis_id[axes[0]]
    i1 = axis_id[axes[1]]

    uv = points[:, [i0, i1]]

    uv_min: np.ndarray = uv.min(axis=0) - padding
    uv_max: np.ndarray = uv.max(axis=0) + padding

    size: np.ndarray = np.ceil((uv_max - uv_min) / cell_size).astype(np.int32) + 1
    width = int(size[0])
    height = int(size[1])

    mask: np.ndarray = np.zeros((height, width), dtype=np.uint8)

    pix: np.ndarray = np.floor((uv - uv_min) / cell_size).astype(np.int32)

    u = np.clip(pix[:, 0], 0, width - 1)
    v = np.clip(pix[:, 1], 0, height - 1)

    # image row goes downward, world axis usually goes upward
    row = height - 1 - v
    col = u

    mask[row, col] = 255

    if dilate_px > 0:
        try:
            import cv2

            kernel_size = 2 * dilate_px + 1
            kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)
        except ImportError:
            pass

    meta = {"axes": axes, "cell_size": cell_size,
            "uv_min": uv_min, "uv_max": uv_max, 
            "width": width, "height": height}
    
    # Real-world projected coordinate of image origin: pixel (row=0, col=0)
    meta["image_origin_uv"] = pixel_to_world_uv(row=0, col=0, meta=meta, center=True)

    return mask, meta


class OrthographicProject:
    def __init__(self, prim_path: str, cell_size: float = 0.005, sample_density: float = 2.0,
                 padding: float = 0.02, dilate_px: int = 1, seed: int = 0):
        self._prim_path = prim_path
        self._cell_size = cell_size
        self._sample_density = sample_density
        self._padding = padding
        self._dilate_px = dilate_px
        self._seed = seed

        self._triangles = None
        self._points = None

    def _update(self):
        self._triangles = collect_world_triangles(self._prim_path)
        self._points = sample_triangle_surface(self._triangles, cell_size=self._cell_size,
                                              sample_density=self._sample_density, seed=self._seed)

    def _get_mask(self, axes=("x", "z")):
        if self._points is None:
            self._update()

        return project_points_to_mask(self._points, axes=axes, cell_size=self._cell_size,
                                      padding=self._padding, dilate_px=self._dilate_px)

    @property
    def projection_on_xy(self):
        return self._get_mask(("x", "y"))

    @property
    def projection_on_xz(self):
        return self._get_mask(("x", "z"))

    @property
    def projection_on_yz(self):
        return self._get_mask(("y", "z"))