from isaacsim import SimulationApp

simulation_app = SimulationApp({
    "headless": False,
    "renderer": "RayTracedLighting",
})

import math
import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from scipy import ndimage
from skimage import measure

from pxr import UsdGeom, UsdShade, Sdf, Gf, Vt
import omni.usd


# =========================
# Config
# =========================

@dataclass
class BoxInfo:
    center: np.ndarray   # shape: (3,)
    size: np.ndarray     # shape: (3,)
    yaw: float           # rad


BOX_COUNT = 18

BOX_SIZE_MIN = np.array([0.35, 0.28, 0.25])
BOX_SIZE_MAX = np.array([0.75, 0.55, 0.50])

VOXEL_SIZE = 0.035        # smaller = better mesh, slower
WRAP_GAP = 0.04           # distance between wrap film and boxes
SMOOTH_ITER = 2

RANDOM_SEED = 7


# =========================
# USD helpers
# =========================

def get_stage():
    return omni.usd.get_context().get_stage()


def create_omnipbr_material(
    stage,
    mat_path: str,
    color=(1.0, 1.0, 1.0),
    opacity=0.26,
    roughness=0.18,
    metallic=0.0,
):
    """
    Create a simple translucent plastic-like material.
    For RGB SDG, this is usually enough.
    """
    mat = UsdShade.Material.Define(stage, mat_path)
    shader = UsdShade.Shader.Define(stage, mat_path + "/Shader")

    shader.CreateIdAttr("OmniPBR")

    shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("opacity_constant", Sdf.ValueTypeNames.Float).Set(float(opacity))
    shader.CreateInput("roughness_constant", Sdf.ValueTypeNames.Float).Set(float(roughness))
    shader.CreateInput("metallic_constant", Sdf.ValueTypeNames.Float).Set(float(metallic))

    # Make material transparent in RTX viewport/rendering.
    shader.CreateInput("enable_opacity", Sdf.ValueTypeNames.Bool).Set(True)

    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


def bind_material(prim, material):
    UsdShade.MaterialBindingAPI(prim).Bind(material)


def create_box_mesh(stage, path: str, box: BoxInfo, material=None):
    """
    Create a visual box mesh manually.
    Using mesh instead of physics cuboid because this script focuses on RGB SDG.
    """
    sx, sy, sz = box.size * 0.5

    local_vertices = np.array([
        [-sx, -sy, -sz],
        [ sx, -sy, -sz],
        [ sx,  sy, -sz],
        [-sx,  sy, -sz],
        [-sx, -sy,  sz],
        [ sx, -sy,  sz],
        [ sx,  sy,  sz],
        [-sx,  sy,  sz],
    ], dtype=np.float32)

    c = math.cos(box.yaw)
    s = math.sin(box.yaw)
    R = np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)

    vertices = local_vertices @ R.T + box.center

    faces = [
        [0, 1, 2, 3],  # bottom
        [4, 7, 6, 5],  # top
        [0, 4, 5, 1],
        [1, 5, 6, 2],
        [2, 6, 7, 3],
        [3, 7, 4, 0],
    ]

    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([Gf.Vec3f(*v) for v in vertices])
    mesh.CreateFaceVertexCountsAttr([4] * len(faces))
    mesh.CreateFaceVertexIndicesAttr([idx for face in faces for idx in face])
    mesh.CreateSubdivisionSchemeAttr("none")

    prim = mesh.GetPrim()
    if material is not None:
        bind_material(prim, material)

    return prim


def create_mesh_from_vertices_faces(stage, path: str, vertices: np.ndarray, faces: np.ndarray, material=None):
    mesh = UsdGeom.Mesh.Define(stage, path)

    mesh.CreatePointsAttr([Gf.Vec3f(float(v[0]), float(v[1]), float(v[2])) for v in vertices])
    mesh.CreateFaceVertexCountsAttr([3] * len(faces))
    mesh.CreateFaceVertexIndicesAttr([int(i) for face in faces for i in face])
    mesh.CreateSubdivisionSchemeAttr("none")

    prim = mesh.GetPrim()

    if material is not None:
        bind_material(prim, material)

    return prim


# =========================
# Box pile generation
# =========================

def generate_irregular_box_pile() -> List[BoxInfo]:
    """
    Generate an intentionally irregular pile.
    This is not physical stacking, only visual randomization.
    """
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    boxes: List[BoxInfo] = []

    # approximate pallet-like footprint
    grid_x = [-0.55, 0.0, 0.55]
    grid_y = [-0.40, 0.0, 0.40]

    current_height = {}

    for _ in range(BOX_COUNT):
        gx = random.choice(grid_x)
        gy = random.choice(grid_y)

        key = (gx, gy)
        base_z = current_height.get(key, 0.0)

        size = np.random.uniform(BOX_SIZE_MIN, BOX_SIZE_MAX)

        # make pile irregular
        jitter_xy = np.random.uniform([-0.08, -0.08], [0.08, 0.08])
        yaw = np.random.uniform(-0.16, 0.16)

        center = np.array([
            gx + jitter_xy[0],
            gy + jitter_xy[1],
            base_z + size[2] * 0.5,
        ], dtype=np.float32)

        boxes.append(BoxInfo(center=center, size=size, yaw=float(yaw)))

        current_height[key] = base_z + size[2] * np.random.uniform(0.82, 1.05)

    # remove some upper boxes visually by not filling all columns equally
    return boxes


# =========================
# Voxelization
# =========================

def compute_scene_bounds(boxes: List[BoxInfo], margin: float) -> Tuple[np.ndarray, np.ndarray]:
    all_pts = []

    for box in boxes:
        sx, sy, sz = box.size * 0.5

        corners = np.array([
            [-sx, -sy, -sz],
            [ sx, -sy, -sz],
            [ sx,  sy, -sz],
            [-sx,  sy, -sz],
            [-sx, -sy,  sz],
            [ sx, -sy,  sz],
            [ sx,  sy,  sz],
            [-sx,  sy,  sz],
        ], dtype=np.float32)

        c = math.cos(box.yaw)
        s = math.sin(box.yaw)
        R = np.array([
            [c, -s, 0.0],
            [s,  c, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)

        corners_w = corners @ R.T + box.center
        all_pts.append(corners_w)

    all_pts = np.concatenate(all_pts, axis=0)

    mn = all_pts.min(axis=0) - margin
    mx = all_pts.max(axis=0) + margin

    # make sure bottom has some space
    mn[2] = min(mn[2], -0.02)

    return mn, mx


def voxelize_oriented_boxes(
    boxes: List[BoxInfo],
    voxel_size: float,
    margin: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rasterize oriented boxes into a 3D occupancy grid.
    Return:
        occupancy: bool volume with shape [nx, ny, nz]
        origin: world coordinate of grid index [0,0,0]
    """
    origin, max_bound = compute_scene_bounds(boxes, margin)

    dims = np.ceil((max_bound - origin) / voxel_size).astype(int) + 1
    nx, ny, nz = dims.tolist()

    occ = np.zeros((nx, ny, nz), dtype=bool)

    for box in boxes:
        # restrict voxel check to local AABB of this box
        bmin, bmax = compute_scene_bounds([box], margin=voxel_size * 2.0)

        imin = np.maximum(np.floor((bmin - origin) / voxel_size).astype(int), 0)
        imax = np.minimum(np.ceil((bmax - origin) / voxel_size).astype(int), dims - 1)

        xs = np.arange(imin[0], imax[0] + 1)
        ys = np.arange(imin[1], imax[1] + 1)
        zs = np.arange(imin[2], imax[2] + 1)

        X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
        pts = np.stack([X, Y, Z], axis=-1).astype(np.float32)
        pts_world = origin + pts * voxel_size

        # world -> box local
        c = math.cos(-box.yaw)
        s = math.sin(-box.yaw)
        R_inv = np.array([
            [c, -s, 0.0],
            [s,  c, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)

        local = (pts_world - box.center) @ R_inv.T
        half = box.size * 0.5

        inside = (
            (np.abs(local[..., 0]) <= half[0]) &
            (np.abs(local[..., 1]) <= half[1]) &
            (np.abs(local[..., 2]) <= half[2])
        )

        occ[X[inside], Y[inside], Z[inside]] = True

    return occ, origin


# =========================
# Wrap mesh generation
# =========================

def smooth_vertices(vertices: np.ndarray, faces: np.ndarray, iterations: int = 2, alpha: float = 0.35):
    """
    Simple Laplacian smoothing.
    Good enough for visual wrap film.
    """
    if iterations <= 0:
        return vertices

    adjacency = [[] for _ in range(len(vertices))]

    for f in faces:
        a, b, c = int(f[0]), int(f[1]), int(f[2])
        adjacency[a].extend([b, c])
        adjacency[b].extend([a, c])
        adjacency[c].extend([a, b])

    adjacency = [list(set(nbs)) for nbs in adjacency]

    v = vertices.copy()

    for _ in range(iterations):
        new_v = v.copy()
        for i, nbs in enumerate(adjacency):
            if not nbs:
                continue
            avg = v[nbs].mean(axis=0)
            new_v[i] = (1.0 - alpha) * v[i] + alpha * avg
        v = new_v

    return v


def generate_wrap_mesh_from_boxes(
    boxes: List[BoxInfo],
    voxel_size: float,
    wrap_gap: float,
    smooth_iter: int,
):
    margin = wrap_gap + voxel_size * 4.0

    occ, origin = voxelize_oriented_boxes(
        boxes=boxes,
        voxel_size=voxel_size,
        margin=margin,
    )

    # close small gaps before dilation
    occ = ndimage.binary_closing(occ, structure=np.ones((3, 3, 3), dtype=bool), iterations=1)

    # outward dilation: this creates the film offset
    dilation_iter = max(1, int(round(wrap_gap / voxel_size)))
    wrap_occ = ndimage.binary_dilation(
        occ,
        structure=np.ones((3, 3, 3), dtype=bool),
        iterations=dilation_iter,
    )

    # Optionally remove internal volume, keeping only outer shell-like surface source.
    # marching_cubes only needs the scalar field boundary.
    volume = wrap_occ.astype(np.float32)

    verts_idx, faces, normals, values = measure.marching_cubes(
        volume,
        level=0.5,
        spacing=(voxel_size, voxel_size, voxel_size),
    )

    vertices_world = verts_idx + origin

    vertices_world = smooth_vertices(
        vertices=vertices_world,
        faces=faces,
        iterations=smooth_iter,
        alpha=0.25,
    )

    # add small wrinkle-like geometric noise
    noise_strength = voxel_size * 0.18
    noise = np.random.normal(0.0, noise_strength, size=vertices_world.shape)
    noise[:, 2] *= 0.45
    vertices_world += noise

    return vertices_world.astype(np.float32), faces.astype(np.int32)


# =========================
# Scene
# =========================

def setup_camera_and_light(stage):
    # Camera
    cam_path = "/World/Camera"
    cam = UsdGeom.Camera.Define(stage, cam_path)
    cam_prim = cam.GetPrim()

    xform = UsdGeom.Xformable(cam_prim)
    xform.AddTranslateOp().Set(Gf.Vec3d(3.2, -4.0, 2.2))
    xform.AddRotateXYZOp().Set(Gf.Vec3f(62.0, 0.0, 38.0))

    cam.CreateFocalLengthAttr(28.0)

    # Dome light
    light = UsdGeom.Scope.Define(stage, "/World/Lights")
    dome = stage.DefinePrim("/World/Lights/DomeLight", "DomeLight")
    dome.CreateAttribute("inputs:intensity", Sdf.ValueTypeNames.Float).Set(750.0)

    # Distant light
    sun = stage.DefinePrim("/World/Lights/Sun", "DistantLight")
    sun.CreateAttribute("inputs:intensity", Sdf.ValueTypeNames.Float).Set(2200.0)
    sun.CreateAttribute("inputs:angle", Sdf.ValueTypeNames.Float).Set(0.35)


def create_ground(stage):
    plane = UsdGeom.Mesh.Define(stage, "/World/Ground")

    size = 8.0
    points = [
        Gf.Vec3f(-size, -size, 0.0),
        Gf.Vec3f( size, -size, 0.0),
        Gf.Vec3f( size,  size, 0.0),
        Gf.Vec3f(-size,  size, 0.0),
    ]

    plane.CreatePointsAttr(points)
    plane.CreateFaceVertexCountsAttr([4])
    plane.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    plane.CreateSubdivisionSchemeAttr("none")

    mat = create_omnipbr_material(
        stage,
        "/World/Materials/GroundMat",
        color=(0.45, 0.45, 0.43),
        opacity=1.0,
        roughness=0.75,
    )
    bind_material(plane.GetPrim(), mat)


def main():
    stage = get_stage()

    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Scope.Define(stage, "/World/Materials")
    UsdGeom.Xform.Define(stage, "/World/Boxes")

    setup_camera_and_light(stage)
    create_ground(stage)

    box_mat = create_omnipbr_material(
        stage,
        "/World/Materials/Cardboard",
        color=(0.62, 0.46, 0.28),
        opacity=1.0,
        roughness=0.62,
    )

    wrap_mat = create_omnipbr_material(
        stage,
        "/World/Materials/StretchWrapFilm",
        color=(0.92, 0.96, 1.0),
        opacity=0.28,
        roughness=0.12,
        metallic=0.0,
    )

    boxes = generate_irregular_box_pile()

    for i, box in enumerate(boxes):
        create_box_mesh(
            stage,
            f"/World/Boxes/box_{i:03d}",
            box,
            material=box_mat,
        )

    print("[INFO] Generating wrap mesh...")
    wrap_vertices, wrap_faces = generate_wrap_mesh_from_boxes(
        boxes=boxes,
        voxel_size=VOXEL_SIZE,
        wrap_gap=WRAP_GAP,
        smooth_iter=SMOOTH_ITER,
    )

    print(f"[INFO] Wrap vertices: {len(wrap_vertices)}, faces: {len(wrap_faces)}")

    create_mesh_from_vertices_faces(
        stage,
        "/World/StretchWrapFilm",
        wrap_vertices,
        wrap_faces,
        material=wrap_mat,
    )

    # Set camera as viewport camera
    omni.usd.get_context().get_selection().set_selected_prim_paths(["/World/StretchWrapFilm"], True)

    # Render a few frames
    for _ in range(120):
        simulation_app.update()

    # Save USD
    out_path = "/tmp/irregular_box_wrap.usda"
    stage.GetRootLayer().Export(out_path)
    print(f"[INFO] Saved scene to: {out_path}")

    while simulation_app.is_running():
        simulation_app.update()


if __name__ == "__main__":
    main()
    simulation_app.close()