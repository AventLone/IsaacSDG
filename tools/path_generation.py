import numpy as np
from typing import Sequence


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


import cv2
import numpy as np
import heapq
from tools.orthographic_views import OrthographicProject

def inflate_obstacles(occ_img: np.ndarray, resolution: float, camera_radius=0.025):
    occupied = occ_img > 127  # white = obstacle

    radius_px = int(np.ceil(camera_radius / resolution))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * radius_px + 1, 2 * radius_px + 1)
    )

    inflated = cv2.dilate(occupied.astype(np.uint8), kernel) > 0
    free = ~inflated

    return free, inflated


def astar(free: np.ndarray, start, goal):
    h, w = free.shape

    def heuristic(a, b):
        return np.hypot(a[0] - b[0], a[1] - b[1])

    neighbors = [
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1)
    ]

    open_set = []
    heapq.heappush(open_set, (0.0, start))

    came_from = {}
    g_score = {start: 0.0}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]

        for dy, dx in neighbors:
            ny = current[0] + dy
            nx = current[1] + dx

            if not (0 <= ny < h and 0 <= nx < w):
                continue
            if not free[ny, nx]:
                continue

            step_cost = np.hypot(dy, dx)
            tentative_g = g_score[current] + step_cost

            nxt = (ny, nx)
            if tentative_g < g_score.get(nxt, float("inf")):
                came_from[nxt] = current
                g_score[nxt] = tentative_g
                f = tentative_g + heuristic(nxt, goal)
                heapq.heappush(open_set, (f, nxt))

    return None


def extract_free_intervals(row_free, min_width_px=10):
    """
    Find continuous free intervals on one image row.
    """
    intervals = []
    in_interval = False
    start = 0

    for x, v in enumerate(row_free):
        if v and not in_interval:
            start = x
            in_interval = True
        elif not v and in_interval:
            end = x - 1
            if end - start + 1 >= min_width_px:
                intervals.append((start, end))
            in_interval = False

    if in_interval:
        end = len(row_free) - 1
        if end - start + 1 >= min_width_px:
            intervals.append((start, end))

    return intervals


def generate_lawnmower_waypoints(free: np.ndarray, spacing_px=20, margin_px=3, min_segment_px=20):
    """
    Generate coverage waypoints that sweep the whole free space.

    free: bool image, True = free
    spacing_px: distance between sweep lines
    margin_px: keep segment endpoints slightly away from obstacle boundary
    """

    h, w = free.shape
    waypoints = []

    sweep_id = 0

    for y in range(spacing_px // 2, h, spacing_px):
        intervals = extract_free_intervals(free[y], min_width_px=min_segment_px)

        if not intervals:
            continue

        # Alternate sweep direction
        left_to_right = sweep_id % 2 == 0

        if left_to_right:
            intervals = sorted(intervals, key=lambda p: p[0])
        else:
            intervals = sorted(intervals, key=lambda p: p[1], reverse=True)

        for x0, x1 in intervals:
            x0 += margin_px
            x1 -= margin_px

            if x1 <= x0:
                continue

            if left_to_right:
                waypoints.append((y, x0))
                waypoints.append((y, x1))
            else:
                waypoints.append((y, x1))
                waypoints.append((y, x0))

        sweep_id += 1

    return waypoints


def connect_waypoints_with_astar(free, waypoints):
    """
    Connect lawnmower waypoints using A*.
    """
    full_path = []
    for a, b in zip(waypoints[:-1], waypoints[1:]):
        path = astar(free, a, b)
        if path is None:
            continue

        if full_path:
            full_path.extend(path[1:])
        else:
            full_path.extend(path)
    return full_path


# def visualize_path(occ_img, inflated, path_px, waypoints=None):
#     vis = cv2.cvtColor(occ_img, cv2.COLOR_GRAY2BGR)

#     # inflated obstacles: gray
#     vis[inflated] = (120, 120, 120)
#     vis[occ_img > 0] = (255, 255, 255)

#     # path: green
#     if path_px is not None and len(path_px) > 1:
#         for p1, p2 in zip(path_px[:-1], path_px[1:]):
#             y1, x1 = p1
#             y2, x2 = p2
#             cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 0), 1)

#     # waypoints: red
#     if waypoints is not None:
#         for (y, x) in waypoints:
#             cv2.circle(vis, (x, y), 3, (0, 0, 255), -1)

#     return vis


def visualize_path(occ_img, inflated, path_px, waypoints=None):
    vis = cv2.cvtColor(occ_img, cv2.COLOR_GRAY2BGR)

    # inflated obstacles: gray
    vis[inflated] = (120, 120, 120)
    vis[occ_img > 0] = (255, 255, 255)

    # draw path only between adjacent pixels
    if path_px is not None and len(path_px) > 1:
        for p1, p2 in zip(path_px[:-1], path_px[1:]):
            y1, x1 = p1
            y2, x2 = p2

            if abs(y2 - y1) <= 1 and abs(x2 - x1) <= 1:
                cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 0), 1)

    if waypoints is not None:
        for y, x in waypoints:
            cv2.circle(vis, (x, y), 2, (0, 0, 255), -1)

    return vis


def pixel_path_to_world_poses(path_px, origin, resolution):
    origin_x, origin_y = origin

    poses = []
    for y_px, x_px in path_px:
        x = origin_x + resolution * float(x_px)
        y = origin_y - resolution * float(y_px)
        poses.append((x, y))

    return poses


def generate_lawnmower_path(objects_prim_path: str):
    resolution = 0.02
    projector = OrthographicProject(prim_path=objects_prim_path, cell_size=resolution, padding=0.3)
    occ_grid, meta = projector.projection_on_xy

    free, inflated = inflate_obstacles(occ_grid, resolution=resolution, camera_radius=0.3)
    waypoints = generate_lawnmower_waypoints(free, spacing_px=30, margin_px=3, min_segment_px=20)
    path_px = connect_waypoints_with_astar(free, waypoints)

    cv2.imwrite("images/camera_path.png", visualize_path(occ_grid, inflated, path_px, waypoints=waypoints))

    return pixel_path_to_world_poses(path_px, meta["image_origin_uv"], resolution)
