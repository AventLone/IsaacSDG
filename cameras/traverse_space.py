import numpy as np
import matplotlib.pyplot as plt


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


def generate_lawnmower_waypoints(
    x_min, x_max,
    y_min, y_max,
    lane_width,
    sample_step=0.1,
    start_from_left=True
):
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

path = generate_lawnmower_waypoints(x_min=0.0, x_max=5.0, y_min=0.0, y_max=5.0, lane_width=0.1, sample_step=0.05)

# Visualize path
plt.figure(figsize=(8, 8))
plt.plot(path[:, 0], path[:, 1], '-o', markersize=2, linewidth=1)
plt.title('Lawnmower Path')
plt.xlabel('X')
plt.ylabel('Y')
plt.axis('equal')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()




