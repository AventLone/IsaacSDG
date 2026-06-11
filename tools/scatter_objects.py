import numpy as np
from tools import common
import random
from isaacsim.core.utils import prims as prims_utils, stage as stage_utils
from omni import usd
from pxr import UsdGeom
from tools import path_generation
import cv2

def can_place_rect(grid: np.ndarray, x, y, w, h, margin=0):
    """
    x, y: top-left pixel coordinate
    w, h: rectangle size in pixels
    margin: extra safety distance around the rectangle
    """
    H, W = grid.shape[:2]

    x0 = max(0, x - margin)
    y0 = max(0, y - margin)
    x1 = min(W, x + w + margin)
    y1 = min(H, y + h + margin)

    if x < 0 or y < 0 or x + w > W or y + h > H:
        return False

    roi = grid[y0:y1, x0:x1]
    return np.all(roi == 0)    # valid only if the whole region is free


def scatter(prim_pathes: list[str], num_for_each, lower_bound, upper_bound, origin=(0, 0, 0)) -> np.ndarray:
    this_stage = stage_utils.get_current_stage()
    scatter_prim_path = "/World/Objects/Scatter"
    if not prims_utils.get_prim_at_path(scatter_prim_path).IsValid():
        prims_utils.create_prim(scatter_prim_path)

    resolution = 0.02

    rect_sizes = []
    for prim in prim_pathes:
        dim_x, dim_y, _ = common.get_dimensions(prim)
        rect_sizes.append((round(dim_x / resolution), round(dim_y / resolution)))
    rect_sizes = rect_sizes * num_for_each

    # grid_size = round((upper_bound[1] - lower_bound[1]) / resolution), 
    H = round((upper_bound[1] - lower_bound[1]) / resolution)
    W = round((upper_bound[0] - lower_bound[0]) / resolution)

    grid = np.zeros((H, W), dtype=np.uint8)
    for i, (w, h) in enumerate(rect_sizes):
        for _ in range(666):
            x = random.randint(0, W - w)
            y = random.randint(0, H - h)

            if can_place_rect(grid, x, y, w, h):
                grid[y: y + h, x: x + w] = 255   # place rect on the grid

                # convert pixel coordinate to real-world coordinate
                cx_px = float(x) + float(w) / 2.0
                cy_px = float(y) + float(h) / 2.0

                world_x = lower_bound[0] + cx_px * resolution + origin[0]
                world_y = upper_bound[1] - cy_px * resolution + origin[1]
                world_z = origin[2]

                # Copy prim
                dst_prim_path = f"{scatter_prim_path}/no_{i:04d}"
                target_prim_path = prim_pathes[i % len(prim_pathes)]
                usd.duplicate_prim(this_stage, prim_path=target_prim_path, path_to=dst_prim_path)
                # common.set_world_trasform(dst_prim_path, 
                #                           translation=(world_x, world_y, world_z), 
                #                           orientation=common.yaw2quat(random.uniform(-30.0, 30.0)))
                common.set_world_trasform(dst_prim_path, translation=(world_x, world_y, world_z))
                UsdGeom.Imageable(prims_utils.get_prim_at_path(dst_prim_path)).MakeVisible()
                
                break

    free, inflated = path_generation.inflate_obstacles(grid, resolution=resolution, camera_radius=0.06)
    waypoints = path_generation.generate_lawnmower_waypoints(free, spacing_px=30, margin_px=3, min_segment_px=20)
    path_px = path_generation.connect_waypoints_with_astar(free, waypoints)

    cv2.imwrite("images/camera_path.png", path_generation.visualize_path(grid, inflated, path_px, waypoints=waypoints))

    return grid, path_generation.pixel_path_to_world_poses(path_px, 
                                                           (origin[0] + lower_bound[0], origin[1] + upper_bound[1]),
                                                           resolution)
