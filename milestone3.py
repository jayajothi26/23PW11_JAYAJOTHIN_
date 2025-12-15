import json
import math

def distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

def die_center(corners):
    x = sum(c[0] for c in corners) / 4
    y = sum(c[1] for c in corners) / 4
    return [x, y]

def die_angle(corners):
    x1, y1 = corners[0]
    x2, y2 = corners[1]
    dx = x2 - x1
    dy = y2 - y1
    return math.degrees(math.atan2(dy, dx))

def angular_diff(a1, a2):
    diff = (a1 - a2) % 90
    return min(diff, 90 - diff)

def trap_time(dist, vmax, amax):
    if dist <= 0:
        return 0.0
    if amax <= 0:
        raise ValueError("Acceleration must be positive")
    trap_dist = vmax ** 2 / (2*amax)
    if dist <= trap_dist:
        return math.sqrt(4 * dist / amax)
    else:
        t_accel = vmax / amax
        cruise_dist = dist - trap_dist
        t_cruise = cruise_dist / vmax
        return 2 * t_accel + t_cruise

def milestone3(data):
    vmax = data["StageVelocity"]
    amax = data["StageAcceleration"]
    cam_vmax = data["CameraVelocity"]
    cam_amax = data["CameraAcceleration"]
    curr_pos = data["InitialPosition"]
    curr_angle = data["InitialAngle"]
    total_time = 0.0
    print(curr_pos)
    path = [curr_pos]
    
    for die in data["Dies"]:
        center = die_center(die["Corners"])
        dist = distance(curr_pos, center)
        stage_time = trap_time(dist, vmax, amax)
        
        target_angle = die_angle(die["Corners"])
        angle_delta = angular_diff(curr_angle, target_angle)
        cam_time = trap_time(angle_delta, cam_vmax, cam_amax)
        
        step_time = max(stage_time, cam_time)
        total_time += step_time
        
        curr_pos = center[:]
        curr_angle = target_angle
        path.append(center[:])
    
    return {
        "TotalTime": round(total_time, 15),
        "Path": path
    }

with open(r"D:\KLA HACKATHON\Input_Milestone3_Testcase1.json") as f: 
    data = json.load(f)

result = milestone3(data)
print(result)
with open("TestCase_3_1.json", "w") as f:
    json.dump(result, f, indent=2)