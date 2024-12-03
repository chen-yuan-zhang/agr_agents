import math
import heapq
from collections import namedtuple

Tile = namedtuple("Tile", ["i", "j"])
Voxel = namedtuple("Voxel", ["i", "j", "k"])
Label = namedtuple('Label', ['pos', 'dir', 'len', 'g', 'f', 'parent'])

def heuristic_euclidean(start, target):
    h = (start.i - target.i)**2 + (start.j - target.j)**2 + (start.k - target.k)**2
    return h

def heuristic_manhattan(start, target, fly_altitude=None):
    h = abs(start.i - target.i) + abs(start.j - target.j)
    return 10*h/4

def heuristic_euclidean_plane_flyaltitude(start, target, fly_altitude=None):
    h = (start.i - target.i)**2 + (start.j - target.j)**2
    if fly_altitude is not None:
        h += (start.k - fly_altitude)**2
    return math.sqrt(h)

def create_root_label(start, target, heuristic, dir, fly_altitude=None):
    next_g = 0
    next_h = heuristic(start, target, fly_altitude)
    next_f = next_g + next_h
    next = Label(start, dir, 0, next_g, next_f, None)
    return next

def extend(current, pos, target, cost, heuristic, dir, fly_altitude=None):
    next_g = current.g + cost
    next_h = heuristic(pos, target, fly_altitude)
    next_f = next_g + next_h
    len = current.len+1
    next = Label(pos, dir, len, next_g, next_f, current)
    return next

def get_path(label):
    path = []
    while label is not None:
        
        path.append(list(label.pos))
        label = label.parent

    path.reverse()
    return path

def astar3d(start, target, cost, voxel_grid, heuristic=heuristic_manhattan, fly_altitude=None, max_iter=None, dim2=False):

    if voxel_grid.ndim == 2:
        X,Y = voxel_grid.shape
    else:
        X,Y,Z = voxel_grid.shape

    if dim2:
        start = Voxel(start[0], start[1], fly_altitude)
        target = Voxel(target[0], target[1], fly_altitude)
    else:
        start = Voxel(*start)
        target = Voxel(*target)

    next = create_root_label(start, target, heuristic, dir=(0, 0), fly_altitude=fly_altitude)

    pq = []
    heapq.heappush(pq, (next.f, next))
    found = None
    visited = {}
    iter = 0

    while pq:

        _, current = heapq.heappop(pq)
        if current.pos == target:
            found = current
            break

        # Max number of nodes explored reached
        if max_iter and iter>max_iter:
            return None

        if dim2:
            directions = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0)]
        else:
            directions = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0),
                      (1, 0, 1), (-1, 0, 1), (0, 1, 1), (0, -1, 1), (1, 1, -1), (1, -1, 1), (-1, 1, 1), (-1, -1, 1),
                      (1, 0, -1), (-1, 0, -1), (0, 1, -1), (0, -1, -1), (1, 1, -1), (1, -1, -1), (-1, 1, -1), (-1, -1, -1)]
        
        for dir in directions:

            dx, dy, dz = dir
            next_pos = Voxel(current.pos.i + dx, current.pos.j + dy, current.pos.k + dz)
            
            if 0 <= next_pos.i < X and 0 <= next_pos.j < Y  and ((voxel_grid.ndim<=2 or int(Z/2) <= next_pos.k < Z)) \
                and next_pos not in visited:
                visited[next_pos] = True

                step_cost = 1 if abs(dx) + abs(dy) == 1 else 1.4
                
                turn_cost = abs(dx - current.dir[0]) + abs(dy - current.dir[1])
                if cost.ndim == 3:
                    edge_cost = cost[next_pos.i, next_pos.j, next_pos.k] #+ step_cost + turn_cost/4
                else:
                    edge_cost = cost[next_pos.i, next_pos.j] #+ step_cost + turn_cost/4
                next = extend(current, next_pos, target, edge_cost, heuristic, dir, fly_altitude)

                heapq.heappush(pq, (next.f, next))

        iter += 1

    if found:
        path = get_path(found)
        return path
    
    return None


def astar2d(start, target, cost, heuristic=heuristic_manhattan, max_iter=None):


    X,Y = cost.shape


    start = Tile(*start)
    target = Tile(*target)

    next = create_root_label(start, target, heuristic, dir=(0, 0))

    pq = []
    heapq.heappush(pq, (next.f, next))
    found = None
    visited = {}
    iter = 0

    while pq:

        _, current = heapq.heappop(pq)
        if current.pos == target:
            found = current
            break

        # Max number of nodes explored reached
        if max_iter and iter>max_iter:
            return None

        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        for dir in directions:

            dx, dy = dir
            next_pos = Tile(current.pos.i + dx, current.pos.j + dy)
            
            if 0 <= next_pos.i < X and 0 <= next_pos.j < Y and next_pos not in visited:

                visited[next_pos] = True
                edge_cost = cost[next_pos.i, next_pos.j] #+ step_cost + turn_cost/4
                next = extend(current, next_pos, target, edge_cost, heuristic, dir)

                heapq.heappush(pq, (next.f, next))

        iter += 1

    if found:
        path = get_path(found)
        return path
    
    return None

