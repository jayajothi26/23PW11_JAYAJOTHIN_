import json
import math

# ---------------- BASIC GEOMETRY ---------------- #

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

# ---------------- ANGLE DIFF ---------------- #

def ang_diff(a, b, sym):
    if sym:
        d = abs((a - b) % 90)
        return min(d, 90 - d)
    d = abs((a - b) % 360)
    return min(d, 360 - d)

# ---------------- MOTION TIME ---------------- #

def linear_time(d, vmax, amax):
    if d <= 0:
        return 0.0
    d_switch = (vmax * vmax) / amax
    if d >= d_switch:
        return (2 * vmax / amax) + (d - d_switch) / vmax
    return 2 * math.sqrt(d / amax)

def angular_time(a, wmax, amax):
    if a <= 0:
        return 0.0
    a_switch = (wmax * wmax) / amax
    if a >= a_switch:
        return (2 * wmax / amax) + (a - a_switch) / wmax
    return 2 * math.sqrt(a / amax)

def move_cost(p1, a1, p2, a2, vmax, amax, wmax, alphamax, sym):
    return max(
        linear_time(dist(p1, p2), vmax, amax),
        angular_time(ang_diff(a1, a2, sym), wmax, alphamax)
    )

# ---------------- CHECK IF DIE IS WITHIN WAFER ---------------- #
def is_within_wafer(center_pos, wafer_diameter):
    """Check if die center is within wafer diameter bounds"""
    radius = wafer_diameter / 2.0
    cx, cy = center_pos
    distance_from_center = math.hypot(cx, cy)  # Distance from wafer center (0,0)
    return distance_from_center <= radius

# ---------------- GREEDY NN - FILTER OUT OF WAFER ---------------- #

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
            
            # SKIP DIES OUTSIDE WAFER DIAMETER - CRITICAL FIX
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

        if best_i == -1:  # No valid dies left within wafer
            break
            
        used[best_i] = True
        pos, _ = dies[best_i]
        ang = best_ang
        path.append((pos, ang))

    return path

# ---------------- TOTAL TIME ---------------- #

def total_time(path, vmax, amax, wmax, alphamax, sym):
    return sum(
        move_cost(path[i][0], path[i][1],
                  path[i+1][0], path[i+1][1],
                  vmax, amax, wmax, alphamax, sym)
        for i in range(len(path)-1)
    )

# ---------------- 2-OPT ---------------- #

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

# ---------------- ANGLE POLISH ---------------- #

def optimize_angles(path, angle_map, sym):
    res = [path[0]]
    for i in range(1, len(path)):
        pos = path[i][0]
        base = angle_map[pos]
        prev = res[-1][1]

        angles = [(base + k*90) % 360] if not sym else \
                 [(base + k*90) % 360 for k in range(4)]

        best_ang = min(angles, key=lambda a: ang_diff(prev, a, sym))
        res.append((pos, best_ang))

    return res

# ---------------- MILESTONE 3 ---------------- #

def milestone3(data):
    sym = data.get("UseSymmetry90", True)
    wafer_diameter = data.get("WaferDiameter", 300.0)  # Default 300mm wafer

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

    # Pass wafer_diameter to greedy_nn
    path = greedy_nn(start_pos, start_ang, dies, vmax, amax, wmax, alphamax, 
                     sym, wafer_diameter)

    path = two_opt(path, vmax, amax, wmax, alphamax, sym)
    path = optimize_angles(path, angle_map, sym)

    return {
        "TotalTime": round(total_time(path, vmax, amax, wmax, alphamax, sym), 6),
        "Path": [list(p) for p, _ in path]
    }

# ---------------- MAIN ---------------- #

if __name__ == "__main__":
    with open("Input_Milestone3_Testcase4.json") as f:
        data = json.load(f)

    result = milestone3(data)
    
    print(f"Path length: {len(result['Path'])}")
    print(f"Wafer diameter used: {data.get('WaferDiameter', 300.0)}mm")
    
    with open("TestCase_3_4.json", "w") as f:
        json.dump(result, f, indent=2)
