import numpy as np
from typing import Sequence
import isaacsim.asset.gen.omap as gen_omap

from isaacsim.asset.gen.omap import 



def generate_orbit_positions(origin: np.ndarray, radius: float, count: int):
    # 在 0 到 2pi 之间均匀生成角度
    angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    # 计算对应的 X, Y 坐标
    return [(float(radius * np.cos(angle) + origin[0]), 
             float(radius * np.sin(angle) + origin[1]), 
             float(origin[2])) for angle in angles]

def generate_orbit_path(height: float, radiuses: list[float]) -> list:
    path = []
    count = 8
    origin = np.array([0.0, 0.0, height], dtype=np.float32)

    for radius in radiuses:
        path.extend(generate_orbit_positions(origin, radius, count))
    return path

def rectangular_contour_points(origin: Sequence[float], dimensions_x: float, dimensions_y: float, 
                               points_per_side=8) -> list[tuple[float, float, float]]:
    half_dimensions_x = dimensions_x / 2.0
    half_dimensions_y = dimensions_y / 2.0

    # points_per_side = total_points // 4
    points_x = np.linspace(origin[0] - half_dimensions_x, origin[0] +
                           half_dimensions_x, points_per_side, endpoint=False)
    points_y = np.linspace(origin[1] - half_dimensions_y, origin[1] +
                           half_dimensions_y, points_per_side, endpoint=False)
    
    points = []
    for point_x in points_x:
        points.append((float(point_x), origin[1] + half_dimensions_y, origin[2]))
    for point_y in points_y:
        points.append((origin[0] + half_dimensions_x, float(point_y), origin[2]))
    for point_y in points_y:
        points.append((origin[0] - half_dimensions_x, float(point_y), origin[2]))
    for point_x in points_x:
        points.append((float(point_x), origin[1] - half_dimensions_y, origin[2]))

    return points

def generate_rectangle_path(height: float, dimensions_list: list[tuple[float, float]]):
    path = []
    for dimensions_x, dimensions_y in dimensions_list:
        path.extend(rectangular_contour_points([0.0, 0.0, height], dimensions_x, dimensions_y, 8))
    return path
