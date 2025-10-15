import heapq
import math
import numpy as np


def heuristic(a, b):
    return math.hypot(b[0] - a[0], b[1] - a[1])


def neighbors(node, grid, allow_diagonal=True):
    x, y = node
    nbrs = []
    steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if allow_diagonal:
        steps += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    for dx, dy in steps:
        nx, ny = x + dx, y + dy
        if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1] and grid[nx, ny] == 0:
            cost = math.hypot(dx, dy)
            nbrs.append(((nx, ny), cost))
    return nbrs


def a_star(grid, start, goal, allow_diagonal=True):
    """A* on a 2D numpy grid. grid==0 free, 1 obstacle.
    start and goal are (x,y) integer cell coordinates.
    Returns list of (x,y) or [] if no path.
    """
    if grid[start[0], start[1]] != 0 or grid[goal[0], goal[1]] != 0:
        return []

    open_set = []
    heapq.heappush(open_set, (0 + heuristic(start, goal), 0, start))
    came_from = {start: None}
    gscore = {start: 0}

    while open_set:
        _, current_g, current = heapq.heappop(open_set)
        if current == goal:
            # reconstruct
            path = []
            n = current
            while n is not None:
                path.append(n)
                n = came_from[n]
            path.reverse()
            return path

        for (nbr, cost) in neighbors(current, grid, allow_diagonal):
            tentative_g = gscore[current] + cost
            if nbr not in gscore or tentative_g < gscore[nbr]:
                came_from[nbr] = current
                gscore[nbr] = tentative_g
                f = tentative_g + heuristic(nbr, goal)
                heapq.heappush(open_set, (f, tentative_g, nbr))

    return []


def smooth_path_bezier(path, samples_per_segment=20):
    """Smooth a polyline `path` using cubic Bezier curves.

    This function computes tangents at each path point using a centered
    difference (similar to Catmull-Rom), converts each adjacent pair of
    points into a cubic Bezier segment with control points determined from
    the tangents, then samples `samples_per_segment` points on each cubic
    to produce a smooth, continuous path.

    Args:
        path: list of (x, y) points (ints or floats).
        samples_per_segment: number of sampled points to generate for each
            Bezier segment (integer >= 1). The returned list includes the
            first point of the path and then sampled points for each segment
            (excluding duplicate endpoints).

    Returns:
        List of (float, float) points representing the smoothed path.
    """
    if len(path) == 0:
        return []
    if len(path) == 1:
        return [(float(path[0][0]), float(path[0][1]))]
    # Convert to float tuples
    pts = [(float(x), float(y)) for x, y in path]

    # Compute tangents t_i = (p_{i+1} - p_{i-1}) / 2 for interior points.
    n = len(pts)
    tangents = [None] * n
    for i in range(n):
        if i == 0:
            # forward difference for first point
            dx = pts[1][0] - pts[0][0]
            dy = pts[1][1] - pts[0][1]
        elif i == n - 1:
            # backward difference for last point
            dx = pts[-1][0] - pts[-2][0]
            dy = pts[-1][1] - pts[-2][1]
        else:
            dx = 0.5 * (pts[i+1][0] - pts[i-1][0])
            dy = 0.5 * (pts[i+1][1] - pts[i-1][1])
        tangents[i] = (dx, dy)

    def bezier_point(t, p0, p1, p2, p3):
        """Compute cubic Bezier point at parameter t."""
        u = 1.0 - t
        b0 = (u * u * u)
        b1 = 3 * (u * u) * t
        b2 = 3 * u * (t * t)
        b3 = t * t * t
        x = b0 * p0[0] + b1 * p1[0] + b2 * p2[0] + b3 * p3[0]
        y = b0 * p0[1] + b1 * p1[1] + b2 * p2[1] + b3 * p3[1]
        return (x, y)

    smoothed = [pts[0]]

    # For each segment from pts[i] to pts[i+1], build Bezier control points
    # B0 = P_i
    # B1 = P_i + tangent_i / 3
    # B2 = P_{i+1} - tangent_{i+1} / 3
    # B3 = P_{i+1}
    for i in range(n - 1):
        p0 = pts[i]
        p3 = pts[i + 1]
        t0 = tangents[i]
        t1 = tangents[i + 1]
        b1 = (p0[0] + t0[0] / 3.0, p0[1] + t0[1] / 3.0)
        b2 = (p3[0] - t1[0] / 3.0, p3[1] - t1[1] / 3.0)

        # Sample segment. skip t==0 (we already have p0), include t==1 only
        # for the final segment to add the path endpoint.
        for s in range(1, samples_per_segment + 1):
            t = s / float(samples_per_segment)
            pt = bezier_point(t, p0, b1, b2, p3)
            smoothed.append(pt)

    return smoothed
