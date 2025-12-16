import json
import math
import sys
import matplotlib.pyplot as plt
import heapq
from matplotlib.patches import Rectangle

def dist(p1, p2):
    return math.hypot(p2[0]-p1[0], p2[1]-p1[1])

def center(corners):
    return (
        sum(p[0] for p in corners) / 4,
        sum(p[1] for p in corners) / 4
    )

def base_angle(corners):
    dx = corners[1][0] - corners[0][0]
    dy = corners[1][1] - corners[0][1]
    return math.degrees(math.atan2(dy, dx)) % 360

def ang_diff(a, b, sym):
    if sym:
        d = abs((a - b) % 90)
        return min(d, 90 - d)
    d = abs((a - b) % 360)
    return min(d, 360 - d)

def linear_time(d, vmax, amax):
    if d <= 0:
        return 0.0
    d_switch = (vmax * vmax) / amax
    if d >= d_switch:
        return (2 * vmax / amax) + (d - d_switch) / vmax
    return 2 * math.sqrt(d / amax)

def angular_time(a, wmax, alphamax):
    if a <= 0:
        return 0.0
    a_switch = (wmax * wmax) / alphamax
    if a >= a_switch:
        return (2 * wmax / alphamax) + (a - a_switch) / wmax
    return 2 * math.sqrt(a / alphamax)

def move_cost(p1, a1, p2, a2, vmax, amax, wmax, alphamax, sym):
    return max(
        linear_time(dist(p1, p2), vmax, amax),
        angular_time(ang_diff(a1, a2, sym), wmax, alphamax)
    )

def is_within_wafer(center_pos, wafer_diameter):
    if wafer_diameter is None:
        return True
    radius = wafer_diameter / 2.0
    cx, cy = center_pos
    distance_from_center = math.hypot(cx, cy)  
    return distance_from_center <= radius

def greedy_nn(start_pos, start_ang, dies,
              vmax, amax, wmax, alphamax, sym, wafer_diameter):
    used = [False]*len(dies)
    pos, ang = start_pos, start_ang
    path = [(pos, ang)]

    for _ in range(len(dies)):
        best_i, best_ang, best_cost = -1, 0, float("inf")

        for i, (p, base) in enumerate(dies):
            if used[i]:
                continue
            
            if not is_within_wafer(p, wafer_diameter):
                continue

            angles = [(base + k*90) % 360] if not sym else \
                     [(base + k*90) % 360 for k in range(4)]

            for a in angles:
                cost = move_cost(pos, ang, p, a,
                               vmax, amax, wmax, alphamax, sym)
                if cost < best_cost:
                    best_cost = cost
                    best_i = i
                    best_ang = a

        if best_i == -1:  
            break
            
        used[best_i] = True
        pos, _ = dies[best_i]
        ang = best_ang
        path.append((pos, ang))

    return path

def total_time(path, vmax, amax, wmax, alphamax, sym):
    return sum(
        move_cost(path[i][0], path[i][1],
                  path[i+1][0], path[i+1][1],
                  vmax, amax, wmax, alphamax, sym)
        for i in range(len(path)-1)
    )

def two_opt(path, vmax, amax, wmax, alphamax, sym):
    best = path
    best_t = total_time(best, vmax, amax, wmax, alphamax, sym)

    improved = True
    while improved:
        improved = False
        for i in range(1, len(best)-2):
            for j in range(i+1, len(best)-1):
                cand = best[:i] + best[i:j][::-1] + best[j:]
                t = total_time(cand, vmax, amax, wmax, alphamax, sym)
                if t < best_t:
                    best, best_t = cand, t
                    improved = True
    return best

def optimize_angles(path, angle_map, sym):
    res = [path[0]]
    for i in range(1, len(path)):
        pos = path[i][0]
        base = angle_map[pos]
        prev = res[-1][1]

        angles = [(base + k*90) % 360] if not sym else [(base + k*90) % 360 for k in range(4)]

        best_ang = min(angles, key=lambda a: ang_diff(prev, a, sym))
        res.append((pos, best_ang))

    return res

def milestone3(data):
    sym = data.get("UseSymmetry90", True)
    wafer_diameter = data.get("WaferDiameter") 

    vmax = data["StageVelocity"]
    amax = data["StageAcceleration"]
    wmax = data["CameraVelocity"]
    alphamax = data["CameraAcceleration"]

    start_pos = tuple(data["InitialPosition"])
    start_ang = data["InitialAngle"]

    dies = []
    angle_map = {}

    for d in data["Dies"]:
        c = center(d["Corners"])
        a = base_angle(d["Corners"])
        dies.append((c, a))
        angle_map[c] = a

    path = greedy_nn(start_pos, start_ang, dies, vmax, amax, wmax, alphamax, 
                     sym, wafer_diameter)

    path = two_opt(path, vmax, amax, wmax, alphamax, sym)
    path = optimize_angles(path, angle_map, sym)

    return {
        "TotalTime": round(total_time(path, vmax, amax, wmax, alphamax, sym), 6),
        "Path": [list(p) for p, _ in path]
    }

# Additional functions for milestone4
def move_cost_idx(idx1, a1, idx2, a2, dists, wmax, alphamax, sym):
    return max(
        dists[idx1][idx2],
        angular_time(ang_diff(a1, a2, sym), wmax, alphamax)
    )

def line_intersects_rect(p1, p2, bl, tr):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    if dx == 0 and dy == 0:
        return False
    t_near = float('-inf')
    t_far = float('inf')
    for i, dir_, minb, maxb in [(0, dx, bl[0], tr[0]), (1, dy, bl[1], tr[1])]:
        if dir_ == 0:
            if p1[i] < minb or p1[i] > maxb:
                return False
        else:
            t1 = (minb - p1[i]) / dir_
            t2 = (maxb - p1[i]) / dir_
            t_near = max(t_near, min(t1, t2))
            t_far = min(t_far, max(t1, t2))
    if t_near > t_far:
        return False
    if t_far < 0 or t_near > 1:
        return False
    return t_near < t_far

def does_intersect_forbidden(p1, p2, zones, eps=1e-6):
    if dist(p1, p2) < eps:
        return False
    for zone in zones:
        bl = zone["BottomLeft"]
        tr = zone["TopRight"]
        bl_eps = [bl[0] + eps, bl[1] + eps]
        tr_eps = [tr[0] - eps, tr[1] - eps]

        if line_intersects_rect(p1, p2, bl_eps, tr_eps):
            return True
    return False

def dijkstra(adj, src, n):
    dist = [float('inf')] * n
    dist[src] = 0
    prev = [-1] * n
    pq = [(0, src)]
    while pq:
        dd, u = heapq.heappop(pq)
        if dd > dist[u]: continue
        for v, w in adj[u]:
            alt = dist[u] + w
            if alt < dist[v]:
                dist[v] = alt
                prev[v] = u
                heapq.heappush(pq, (alt, v))
    return dist, prev

def find_detour(p1, p2, zones, offset=5, max_attempts=10):
    mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    length = math.hypot(dx, dy)
    if length == 0:
        return None
    perp_dx, perp_dy = -dy / length, dx / length
    for i in range(1, max_attempts + 1):
        for sign in [1, -1]:
            detour = (mx + sign * offset * i * perp_dx, my + sign * offset * i * perp_dy)
            if not does_intersect_forbidden(p1, detour, zones) and not does_intersect_forbidden(detour, p2, zones):
                return detour
    return None

def get_path(src, tgt, prev):
    path = []
    cur = tgt
    while cur != -1:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path if path and path[0] == src else []

def greedy_nn_m4(start_idx, start_ang, dies, sym, dists, wmax, alphamax, node_list, pos_to_idx, angle_map):
    used = [False]*len(dies)
    current_idx = start_idx
    current_ang = start_ang
    path = [(current_idx, current_ang)]

    for _ in range(len(dies)):
        best_ii, best_ang, best_cost = -1, 0, float("inf")

        for ii, (p, base) in enumerate(dies):
            if used[ii]:
                continue
            idx = pos_to_idx[p]

            angles = [(base + k*90) % 360 for k in range(1 if not sym else 4)]

            for a in angles:
                cost = max(dists[current_idx][idx], angular_time(ang_diff(current_ang, a, sym), wmax, alphamax))
                if cost < best_cost:
                    best_cost = cost
                    best_ii = ii
                    best_ang = a
                    best_idx = idx

        if best_ii == -1:
            break

        used[best_ii] = True
        current_idx = best_idx
        current_ang = best_ang
        path.append((current_idx, current_ang))

    return path

def total_time_m4(path, dists, wmax, alphamax, sym):
    return sum(
        move_cost_idx(path[i][0], path[i][1], path[i+1][0], path[i+1][1], dists, wmax, alphamax, sym)
        for i in range(len(path)-1)
    )

def two_opt_m4(path, dists, wmax, alphamax, sym):
    best = path
    best_t = total_time_m4(best, dists, wmax, alphamax, sym)

    improved = True
    while improved:
        improved = False
        for i in range(1, len(best)-2):
            for j in range(i+1, len(best)-1):
                cand = best[:i] + best[i:j][::-1] + best[j:]
                t = total_time_m4(cand, dists, wmax, alphamax, sym)
                if t < best_t:
                    best, best_t = cand, t
                    improved = True
    return best

def optimize_angles_m4(path, angle_map, sym, node_list):
    res = [path[0]]
    for i in range(1, len(path)):
        idx = path[i][0]
        base = angle_map[node_list[idx]]
        prev = res[-1][1]

        angles = [(base + k*90) % 360 for k in range(1 if not sym else 4)]

        best_ang = min(angles, key=lambda a: ang_diff(prev, a, sym))
        res.append((idx, best_ang))

    return res

def milestone4(data):
    sym = data.get("UseSymmetry90", True)
    wafer_diameter = data.get("WaferDiameter")

    vmax = data["StageVelocity"]
    amax = data["StageAcceleration"]
    wmax = data["CameraVelocity"]
    alphamax = data["CameraAcceleration"]

    start_pos = tuple(data["InitialPosition"])
    start_ang = data["InitialAngle"]
    forbidden_zones = data.get("ForbiddenZones", [])

    dies_centers = []
    angle_map = {}  
    for d in data["Dies"]:
        c = center(d["Corners"])
        a = base_angle(d["Corners"])
        dies_centers.append(c)
        angle_map[c] = a

    points = [start_pos] + dies_centers

    corners = []
    for zone in forbidden_zones:
        bl = tuple(zone["BottomLeft"])
        tr = tuple(zone["TopRight"])
        tl = (bl[0], tr[1])
        br = (tr[0], bl[1])
        corners += [bl, tl, tr, br]

    all_nodes_set = set(points + corners)
    node_list = list(all_nodes_set)
    n_nodes = len(node_list)

    pos_to_idx = {pos: idx for idx, pos in enumerate(node_list)}

    start_idx = pos_to_idx[start_pos]

    die_indices = []
    for c in dies_centers:
        if c in pos_to_idx:
            die_indices.append(pos_to_idx[c])

    adj = [[] for _ in range(n_nodes)]
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            p1 = node_list[i]
            p2 = node_list[j]
            if not does_intersect_forbidden(p1, p2, forbidden_zones):
                d = dist(p1, p2)
                t = linear_time(d, vmax, amax)
                adj[i].append((j, t))
                adj[j].append((i, t))

    dists = {}
    prevs = {}
    visit_indices = [start_idx] + die_indices
    for src in set(visit_indices):
        d, p = dijkstra(adj, src, n_nodes)
        dists[src] = d
        prevs[src] = p

    filtered_dies = []
    for c, base in zip(dies_centers, [angle_map[c] for c in dies_centers]):
        if is_within_wafer(c, wafer_diameter):
            filtered_dies.append((c, base))

    path = greedy_nn_m4(start_idx, start_ang, filtered_dies, sym, dists, wmax, alphamax, node_list, pos_to_idx, angle_map)

    path = two_opt_m4(path, dists, wmax, alphamax, sym)
    path = optimize_angles_m4(path, angle_map, sym, node_list)
    full_path_idx = []
    for k in range(len(path) - 1):
        sub_path = get_path(path[k][0], path[k+1][0], prevs[path[k][0]])
        if k == 0:
            full_path_idx = sub_path
        else:
            full_path_idx += sub_path[1:]

    full_path_pos = [list(node_list[idx]) for idx in full_path_idx]

    # Inject detour logic to smooth grazing segments
    adjusted_path = []
    for i in range(len(full_path_pos) - 1):
        p1, p2 = full_path_pos[i], full_path_pos[i+1]
        if does_intersect_forbidden(p1, p2, forbidden_zones):
            detour = find_detour(p1, p2, forbidden_zones)
            if detour:
                adjusted_path.append(p1)
                adjusted_path.append(detour)
            else:
                adjusted_path.append(p1)
        else:
            adjusted_path.append(p1)
    adjusted_path.append(full_path_pos[-1])

    return {
        "TotalTime": round(total_time_m4(path, dists, wmax, alphamax, sym), 6),
        "Path": adjusted_path
    }

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python <milestone_number> <testcase_number>")
        sys.exit(1)

    milestone = sys.argv[1]
    testcase = sys.argv[2]
    
    if milestone == '3':
        input_file = f"C:\\Users\\kla_user\\Downloads\\KLA HACKATHON\\KLA HACKATHON\\Input_Milestone{milestone}_Testcase{testcase}.json"
        func = milestone3
    elif milestone == '4':
        input_file = f"C:\\Users\\kla_user\\Downloads\\KLA HACKATHON\\KLA HACKATHON\\Milestone4\\Input_Milestone{milestone}_Testcase{testcase}.json"
        func = milestone4
    else:
        print("Invalid milestone number. Use 3 or 4.")
        sys.exit(1)
    
    output_file = f"TestCase_{milestone}_{testcase}.json"

    with open(input_file) as f:
        data = json.load(f)
    result = func(data)

    print(f"Path length: {len(result['Path'])}")
    if "WaferDiameter" in data:
        print(f"Wafer diameter used: {data.get('WaferDiameter')}mm")
    all_corners=[]
    for die in data['Dies']:
        for corner in die['Corners']:
            all_corners.append(corner)
    x_coords = [point[0] for point in all_corners]
    y_coords = [point[1] for point in all_corners]

    plt.figure(figsize=(10, 10))
    plt.scatter(x_coords, y_coords, color='blue', s=10, label='Corners')
    for die in data['Dies']:
        corners = die['Corners']
        x_die = [c[0] for c in corners] + [corners[0][0]]
        y_die = [c[1] for c in corners] + [corners[0][1]]
        plt.plot(x_die, y_die, color='red', linewidth=1)
    plt.scatter(data['InitialPosition'][0], data['InitialPosition'][1],color='green', s=100, marker='x', label='Initial Position')
    if result["Path"]:
        path_x = [p[0] for p in result["Path"]]
        path_y = [p[1] for p in result["Path"]]
        plt.plot(path_x, path_y, color='purple', linewidth=2, marker='o', markersize=5, label='Optimized Path')
        for i, (x, y) in enumerate(zip(path_x, path_y), start=1):
            plt.text(x, y, str(i), fontsize=10, color='black', ha='right', va='bottom')
    
    if 'ForbiddenZones' in data and data['ForbiddenZones']:
        for zone in data['ForbiddenZones']:
            bl = zone['BottomLeft']   
            tr = zone['TopRight']     
            width = tr[0] - bl[0]
            height = tr[1] - bl[1]
            rect = Rectangle(
                (bl[0], bl[1]), width, height,
                linewidth=1.5, edgecolor='black',
                facecolor='red', alpha=0.3, label='Forbidden zone'
            )
            plt.gca().add_patch(rect)

        if len(data['ForbiddenZones']) > 1:
            proxy = Rectangle((0, 0), 1, 1,
                            linewidth=1.5, edgecolor='black',
                            facecolor='red', alpha=0.3)
            plt.legend(handles=[proxy], labels=['Forbidden zone'], loc='best')

        # initial position + path
        plt.scatter(data['InitialPosition'][0], data['InitialPosition'][1], color='green', s=100, marker='x', label='Initial Position')
        if result["Path"]:
            path_x = [p[0] for p in result["Path"]]
            path_y = [p[1] for p in result["Path"]]
            plt.plot(path_x, path_y, color='purple', linewidth=2, marker='o', markersize=5, label='Optimized Path')
            for i, (x, y) in enumerate(zip(path_x, path_y), start=1):
                plt.text(x, y, str(i), fontsize=10, color='black', ha='right', va='bottom')

    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Plot of Die Corners with Optimized Path (Initial Included)')
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.show()

    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)
