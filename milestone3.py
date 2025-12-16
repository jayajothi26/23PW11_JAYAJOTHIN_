import json
import math
import sys
import matplotlib.pyplot as plt
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

def is_within_wafer(center_pos, wafer_diameter):
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


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python  <milestone_number> <testcase_number>")
        sys.exit(1)

    milestone = sys.argv[1]
    testcase = sys.argv[2]
    input_file = f"C:\\Users\\kla_user\\Downloads\\KLA HACKATHON\\KLA HACKATHON\\Input_Milestone{milestone}_Testcase{testcase}.json"
    output_file = f"TestCase_{milestone}_{testcase}.json"

    with open(input_file) as f:
        data = json.load(f)
    result = milestone3(data)

    print(f"Path length: {len(result['Path'])}")
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
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Plot of Die Corners with Optimized Path (Initial Included)')
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.show()

   

    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)



