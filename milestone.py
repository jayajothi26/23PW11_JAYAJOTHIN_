import json
import math
import matplotlib.pyplot as plt

def distance_of_die(x, y):
    return math.sqrt((x[0]-y[0])**2 + (x[1]-y[1])**2)

def center_die(corners):
    x_sum = [c[0] for c in corners]
    y_sum = [c[1] for c in corners]
    center_x = sum(x_sum) / 4
    center_y = sum(y_sum) / 4
    return [center_x, center_y]

def milestone1(lines):
    curr = lines["InitialPosition"]
    stage_velocity = lines["StageVelocity"]
    total_time = 0
    path = [curr]
    for die in lines["Dies"]:
        centre = center_die(die["Corners"])
        distance = distance_of_die(curr, centre)
        time = distance / stage_velocity
        total_time += time
        curr = centre
        path.append(centre)
    return {
        "TotalTime": total_time,
        "Path": path
    }

def calculate_die_properties(corners):
    cx = sum(c[0] for c in corners) / 4
    cy = sum(c[1] for c in corners) / 4
    center = (cx, cy)
    dx = corners[1][0] - corners[0][0]
    dy = corners[1][1] - corners[0][1]
    angle = math.degrees(math.atan2(dy, dx)) % 360
    return center, angle

def distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

def angular_diff_sym_90(a1, a2):
    d = abs((a1 - a2) % 90)
    return min(d, 90 - d)

def angular_diff_360(a1, a2):
    d = abs((a1 - a2) % 360)
    return min(d, 360 - d)

def move_time_with_symmetry(from_state, to_center, to_base_angle, stage_v, cam_v, use_symmetry=True):
    (pos1, ang1) = from_state
    stage_time = distance(pos1, to_center) / stage_v
    if use_symmetry:
        candidate_angles = [(to_base_angle + k * 90) % 360 for k in range(4)]
        cam_time = min(angular_diff_sym_90(ang1, ang2) / cam_v for ang2 in candidate_angles)
    else:
        cam_time = angular_diff_360(ang1, to_base_angle) / cam_v
    return max(stage_time, cam_time)

def nearest_neighbor_angle_aware(start_state, dies, stage_v, cam_v, use_symmetry=True):
    n = len(dies)
    visited = [False] * n
    path = [start_state] 
    curr_state = start_state
    total_time = 0.0
    for _ in range(n):
        best_idx, best_step, best_angle = None, float("inf"), None
        for i in range(n):
            if visited[i]:
                continue
            center_i, base_angle_i = dies[i]
            candidate_angles = [(base_angle_i + k * 90) % 360] if not use_symmetry else [
                (base_angle_i + k * 90) % 360 for k in range(4)
            ]
            for ang_i in candidate_angles:
                step = max(
                    distance(curr_state[0], center_i) / stage_v,
                    (angular_diff_sym_90(curr_state[1], ang_i) / cam_v if use_symmetry
                     else angular_diff_360(curr_state[1], ang_i) / cam_v)
                )
                if step < best_step:
                    best_idx, best_step, best_angle = i, step, ang_i

        visited[best_idx] = True
        center, _ = dies[best_idx]
        curr_state = (center, best_angle)
        path.append(curr_state)
        total_time += best_step
    return path, total_time

def total_path_time_angle_aware(path, stage_v, cam_v, use_symmetry=True):
    total = 0.0
    for i in range(len(path) - 1):
        stage_t = distance(path[i][0], path[i+1][0]) / stage_v
        if use_symmetry:
            cam_t = angular_diff_sym_90(path[i][1], path[i+1][1]) / cam_v
        else:
            cam_t = angular_diff_360(path[i][1], path[i+1][1]) / cam_v
        total += max(stage_t, cam_t)
    return total

def two_opt(path, stage_v, cam_v, use_symmetry=True):
    improved = True
    current_time = total_path_time_angle_aware(path, stage_v, cam_v, use_symmetry)
    while improved:
        improved = False
        for i in range(1, len(path) - 2):
            for j in range(i + 1, len(path) - 1):
                new_path = path[:i] + path[i:j][::-1] + path[j:]
                new_time = total_path_time_angle_aware(new_path, stage_v, cam_v, use_symmetry)
                if new_time + 1e-12 < current_time:
                    path = new_path
                    current_time = new_time
                    improved = True
    return path

def or_opt(path, stage_v, cam_v, use_symmetry=True, max_seg_len=3):
    improved = True
    while improved:
        improved = False
        best_time = total_path_time_angle_aware(path, stage_v, cam_v, use_symmetry)
        n = len(path)
        for seg_len in range(1, max_seg_len + 1):
            for i in range(1, n - seg_len):
                seg = path[:i] + path[i+seg_len:]
                remainder = path[:i] + path[i+seg_len:]
                for j in range(1, len(remainder)):
                    new_path = remainder[:j] + seg + remainder[j:]
                    new_time = total_path_time_angle_aware(new_path, stage_v, cam_v, use_symmetry)
                    if new_time + 1e-12 < best_time:
                        path = new_path
                        best_time = new_time
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    return path

def optimize_angles_for_order(path, stage_v, cam_v, dies_map, use_symmetry=True):
    optimized = [path[0]]
    for i in range(1, len(path)):
        center_i = path[i][0]
        base_angle = dies_map.get(center_i, path[i][1])
        prev_ang = optimized[-1][1]

        if use_symmetry:
            candidate_angles = [(base_angle + k * 90) % 360 for k in range(4)]
            next_ang = path[i+1][1] if i + 1 < len(path) else None
            best_ang, best_cost = None, float("inf")
            for ang in candidate_angles:
                incoming = angular_diff_sym_90(prev_ang, ang) / cam_v
                if next_ang is not None:
                    outgoing = angular_diff_sym_90(ang, next_ang) / cam_v
                    cost = max(incoming, outgoing)
                else:
                    cost = incoming
                if cost < best_cost:
                    best_ang, best_cost = ang, cost
        else:
            best_ang = base_angle

        optimized.append((center_i, best_ang))
    return optimized

def milestone2(data, use_symmetry=True):
    initial_pos = tuple(data["InitialPosition"])
    initial_angle = data["InitialAngle"]
    stage_v = data["StageVelocity"]
    cam_v = data["CameraVelocity"]
    dies = [calculate_die_properties(d["Corners"]) for d in data["Dies"]]
    start_state = (initial_pos, initial_angle)
    if "UseSymmetry90" in data:
        use_symmetry = bool(data["UseSymmetry90"])
    dies_map = {center: angle for center, angle in dies}
    path, _ = nearest_neighbor_angle_aware(start_state, dies, stage_v, cam_v, use_symmetry)
    path = two_opt(path, stage_v, cam_v, use_symmetry)
    path = or_opt(path, stage_v, cam_v, use_symmetry)
    path = optimize_angles_for_order(path, stage_v, cam_v, dies_map, use_symmetry)
    total_time = total_path_time_angle_aware(path, stage_v, cam_v, use_symmetry)
    full_positions = [state[0] for state in path]
    return {"TotalTime": round(total_time, 3), "Path": full_positions}

if __name__ == "__main__":
    # Run Milestone 1
    with open('D:\\KLA HACKATHON\\Input_Milestone1_Testcase4.json', 'r') as f:
        data1 = json.load(f)
    result1 = milestone1(data1)
    print("Milestone 1 Result:", result1)
    with open('TestCase_1_4.json', 'w') as f:
        json.dump(result1, f)

    # Run Milestone 2
    with open(r"D:\KLA HACKATHON\Input_Milestone2_Testcase3.json") as f:
        data2 = json.load(f)

    result2 = milestone2(data2, use_symmetry=True)
    print("Milestone 2 Result:", result2)

    with open("TestCase_2_3.json", "w") as f:
        json.dump(result2, f, indent=2)

    # Plotting for Milestone 2
    all_corners = []
    for die in data2['Dies']:
        for corner in die['Corners']:
            all_corners.append(corner)
    x_coords = [point[0] for point in all_corners]
    y_coords = [point[1] for point in all_corners]

    plt.figure(figsize=(10, 10))
    plt.scatter(x_coords, y_coords, color='blue', s=10, label='Corners')

    for die in data2['Dies']:
        corners = die['Corners']
        x_die = [c[0] for c in corners] + [corners[0][0]]
        y_die = [c[1] for c in corners] + [corners[0][1]]
        plt.plot(x_die, y_die, color='red', linewidth=1)

    plt.scatter(data2['InitialPosition'][0], data2['InitialPosition'][1], color='green', s=100, marker='x', label='Initial Position')

    if result2["Path"]:
        path_x = [p[0] for p in result2["Path"]]
        path_y = [p[1] for p in result2["Path"]]
        plt.plot(path_x, path_y, color='purple', linewidth=2, marker='o', markersize=5, label='Optimized Path')

        for i, (x, y) in enumerate(zip(path_x, path_y), start=1):
            plt.text(x, y, str(i), fontsize=10, color='black', ha='right', va='bottom')

    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Plot of Die Corners with Optimized Path (Milestone 2)')
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.show()
