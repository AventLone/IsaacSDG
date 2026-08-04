import numpy as np
import tools.common, random, carb

def _sample_segment(start, end, step):
    """
    Sample points between two 2D points with approximately `step` spacing.
    """
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)

    length = float(np.linalg.norm(end - start))
    if length == 0.0:
        return [start.tolist()]

    n_steps = max(1, int(np.ceil(length / step)))
    t_values = np.linspace(0.0, 1.0, n_steps + 1)
    return (start + (end - start) * t_values[:, None]).tolist()


def generate_lawnmower_waypoints(x_min, x_max, y_min, y_max, lane_width, sample_step=0.1, start_from_left=True):
    """
    Generate lawnmower coverage waypoints.

    Args:
        x_min, x_max: X range of the coverage area
        y_min, y_max: Y range of the coverage area
        lane_width: Distance between adjacent lanes
        sample_step: Approximate spacing between consecutive waypoint samples
        start_from_left: If True, first lane goes from x_min -> x_max

    Returns:
        np.ndarray of shape (N, 2), each row is [x, y]
    """

    waypoints = []

    y_values = np.arange(y_min, y_max + lane_width * 0.5, lane_width)

    direction = start_from_left
    current_point = None

    for y in y_values:
        lane_start = [x_min, y] if direction else [x_max, y]
        lane_end = [x_max, y] if direction else [x_min, y]

        if current_point is None:
            waypoints.append(lane_start)
        elif not np.allclose(current_point, lane_start):
            transition = _sample_segment(current_point, lane_start, sample_step)
            waypoints.extend(transition[1:])

        lane_points = _sample_segment(lane_start, lane_end, sample_step)
        waypoints.extend(lane_points[1:])

        current_point = lane_end
        direction = not direction

    return np.array(waypoints)


import omni.physx


async def get_traverse_path(lower_boundary: tuple[float, float, float],
                            upper_boundary: tuple[float, float, float],
                            lane_gap: float, sample_step: float,
                            start_from_left=True) -> list[tuple[float, float, float]]:
    """
    Generate collision-free traverse waypoints within a 3D boundary volume.

    The function raster-scans the XY area using the provided step sizes (`dx`, `dy`) and,
    for each XY sample, randomly picks a Z value between `lower_boundary[2]` and
    `upper_boundary[2]`. A waypoint is accepted only if no overlap is detected by
    PhysX `overlap_sphere` query.

    Args:
        lower_boundary: Minimum corner of the sampling volume as (x, y, z).
        upper_boundary: Maximum corner of the sampling volume as (x, y, z).
        dx: Step size along X direction. Must be > 0.
        dy: Step size along Y direction. Must be > 0.
        sweep_axis_x: If True, outer loop sweeps Y then X; otherwise X then Y.

    Returns:
        A list of valid waypoints as (x, y, z) tuples.

    Raises:
        ValueError: If `dx` or `dy` is not positive.
        RuntimeError: If PhysX scene query interface is unavailable.
    """
    
    if lane_gap <= 0 or sample_step <= 0:
        raise ValueError("lane_gap and sample_step must be > 0")

    x_min, x_max = sorted((lower_boundary[0], upper_boundary[0]))
    y_min, y_max = sorted((lower_boundary[1], upper_boundary[1]))
    z_min, z_max = sorted((lower_boundary[2], upper_boundary[2]))

    y_values = np.arange(y_min, y_max + lane_gap * 0.5, lane_gap)

    waypoints = []
    direction = start_from_left
    current_point = None
    for y in y_values:
        lane_start = [x_min, y] if direction else [x_max, y]
        lane_end = [x_max, y] if direction else [x_min, y]

        if current_point is None:
            waypoints.append(lane_start)
        elif not np.allclose(current_point, lane_start):
            transition = _sample_segment(current_point, lane_start, sample_step)
            waypoints.extend(transition[1:])

        lane_points = _sample_segment(lane_start, lane_end, sample_step)
        waypoints.extend(lane_points[1:])

        current_point = lane_end
        direction = not direction

    tools.common.timeline.play()
    await tools.common.wait_for(2)
    scene_query = omni.physx.get_physx_scene_query_interface()
    if scene_query is None:
        raise RuntimeError("PhysX scene query interface is unavailable.")

    safe_path = []
    for waypoint in waypoints:
        point = (float(waypoint[0]), float(waypoint[1]), random.uniform(z_min, z_max))
        if scene_query.overlap_sphere(0.1, carb.Float3(*point), None, True) == 0:
            safe_path.append(point)

    tools.common.timeline.pause()
    return safe_path