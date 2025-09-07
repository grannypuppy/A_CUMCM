import numpy as np

def solve_smoke_deployment(missiles_info, drones_info, flight_info, bombs_info, num_bombs_per_drone, time_horizon, time_step, verbose=True):
    """
    Sets up and solves the smoke bomb deployment optimization problem.

    Args:
        missiles_info (dict): Dictionary with missile names as keys and initial positions as values.
        drones_info (dict): Dictionary with drone names as keys and initial positions as values.
        num_bombs_per_drone (int): The number of bombs each drone can deploy.
        time_horizon (int): The total simulation time in seconds.
        time_step (float): The step size for time discretization.
    """
    # --- 1. Constants and Initialization ---
    V_MISSILE = 300.0  # m/s
    V_SINK = 3.0  # m/s
    R_SMOKE = 10.0  # m
    G = 9.8  # m/s^2
    MIN_DROP_INTERVAL = 1.0 # s
    SMOKE_DURATION = 20.0 # s

    # Target Definition
    TARGET_CENTER_BASE = np.array([0, 200, 0])
    TARGET_RADIUS = 7.0
    TARGET_HEIGHT = 10.0

    # Time and Object Discretization
    time_steps = np.arange(0, time_horizon, time_step)
    num_time_steps = len(time_steps)
    
    # Generate 48 target points as per the user's method
    target_points = []
    for h in [0, TARGET_HEIGHT]:
        for i in range(24):
            angle = 2 * np.pi * i / 24
            x = TARGET_CENTER_BASE[0] + TARGET_RADIUS * np.cos(angle)
            y = TARGET_CENTER_BASE[1] + TARGET_RADIUS * np.sin(angle)
            z = TARGET_CENTER_BASE[2] + h
            target_points.append(np.array([x, y, z]))

    # --- 3. Decision Variables ---
    # Drone flight parameters
    drone_speed = dict.fromkeys(drones_info.keys(), 120.0)
    drone_angle = dict.fromkeys(drones_info.keys(), np.pi) # theta

    # We need to model cos and sin of the angle. Gurobi's general constraints are good for this.
    drone_cos_angle = {j: np.cos(drone_angle[j]) for j in drones_info.keys()}
    drone_sin_angle = {j: np.sin(drone_angle[j]) for j in drones_info.keys()}

    # Bomb deployment variables
    bombs = [(j, k) for j in drones_info.keys() for k in range(num_bombs_per_drone)]
    t_drop = dict.fromkeys(bombs, 1.5)
    fusetime = dict.fromkeys(bombs, 3.6)

    for j, (sp, theta) in flight_info.items():
        drone_speed[j] = sp
        drone_angle[j] = theta
        drone_cos_angle[j] = np.cos(theta)
        drone_sin_angle[j] = np.sin(theta)

    for (j, k), (t_drop_val, fusetime_val) in bombs_info.items():
        t_drop[j, k] = t_drop_val
        fusetime[j, k] = fusetime_val


    t_det = {(j,k): t_drop[j,k] + fusetime[j,k] for j, k in bombs}


    ##
    if verbose:
        for j in drones_info.keys():
            print(f"Drone {j} speed: {drone_speed[j]}, angle: {np.rad2deg(drone_angle[j]):.2f} degrees")
            for k in range(num_bombs_per_drone):
                print(f"Bomb {j},{k} drop time: {t_drop[(j,k)]}, fuse time: {fusetime[(j,k)]}")
                print(f"Bomb {j},{k} detonation time: {t_det[(j,k)]}")
    ##
    # Binary variable to indicate if a time step is effectively obscured
    is_obscured = dict.fromkeys(range(num_time_steps), 0)
    bombs_job = {b: {t: set() for t in range(num_time_steps)} for b in bombs}
    # Obscuration Logic (this is the complex part)
    for t_idx, t in enumerate(time_steps):
        # This variable is 1 if all lines of sight are blocked at time t

        # Intermediate variables for positions
        # Missile positions at time t
        missile_pos = {}
        for i, p0_i in missiles_info.items():
            p0_vec = np.array(p0_i)
            dist_to_target = np.linalg.norm(p0_vec)
            time_to_target = dist_to_target / V_MISSILE
            if t < time_to_target:
                 missile_pos[i] = p0_vec * (1 - (V_MISSILE * t) / dist_to_target)
            else:
                 missile_pos[i] = np.array([0,0,0]) # Reached target
        
        p_drop_x = dict.fromkeys(bombs, 0)
        p_drop_y = dict.fromkeys(bombs, 0)
        p_det_x = dict.fromkeys(bombs, 0)
        p_det_y = dict.fromkeys(bombs, 0)
        p_det_z = dict.fromkeys(bombs, 0)
        p_cloud_x = dict.fromkeys(bombs, 0)
        p_cloud_y = dict.fromkeys(bombs, 0)
        p_cloud_z = dict.fromkeys(bombs, 0)

        for j, k in bombs:
            if t_det[j,k] > t or t > t_det[j,k] + SMOKE_DURATION:
                continue
            p_drop_x[j,k] = drones_info[j][0] + drone_speed[j] * drone_cos_angle[j] * t_drop[j,k]
            p_drop_y[j,k] = drones_info[j][1] + drone_speed[j] * drone_sin_angle[j] * t_drop[j,k]
            p_det_x[j,k] = p_drop_x[j,k] + drone_speed[j] * drone_cos_angle[j] * fusetime[j,k]
            p_det_y[j,k] = p_drop_y[j,k] + drone_speed[j] * drone_sin_angle[j] * fusetime[j,k]
            p_det_z[j,k] = drones_info[j][2] - 0.5 * G * fusetime[j,k] * fusetime[j,k]
            p_cloud_x[j,k] = p_det_x[j,k]
            p_cloud_y[j,k] = p_det_y[j,k]
            p_cloud_z[j,k] = p_det_z[j,k] - V_SINK * (t - t_det[j,k])


        for i in missiles_info.keys():
            for p_target in target_points:
                line_blocked = 0
                missile_A = missile_pos[i]
                target_B = p_target
                AB = target_B - missile_A 
                for j, k in bombs:
                    if t_det[j,k] > t or t > t_det[j,k] + SMOKE_DURATION:
                        continue

                    # 1. Define the line segment from missile (A) to target point (B)
                    cloud_C = np.array([p_cloud_x[j,k], p_cloud_y[j,k], p_cloud_z[j,k]])
                    AC = cloud_C - missile_A
                    BC = cloud_C - target_B
                    
                    # Project AC onto AB to find the closest point on the line AB
                    s = np.dot(AC, AB) / np.dot(AB, AB)
                    
                    dist_sq = 0
                    if s < 0:
                        # Closest point is A (missile)
                        dist_sq = np.dot(AC, AC)
                    elif s > 1:
                        # Closest point is B (target)
                        dist_sq = np.dot(BC, BC)
                    else:
                        # Closest point is on the segment, use projection
                        projection = missile_A + s * AB
                        dist_sq = np.dot(cloud_C - projection, cloud_C - projection)

                    # print(f" time: {t}, target_point: {p_target}, cloud_C {j,k}: {cloud_C}")

                    if dist_sq <= R_SMOKE * R_SMOKE:
                        line_blocked = 1
                        bombs_job[j,k][t_idx].add(i)
                        # if t < 6 :
                        #     print(f" time: {t}, target_point: {p_target}, cloud_C {j,k}: {cloud_C}")
                if line_blocked == 0:
                    break
            if line_blocked == 0:
                break
        is_obscured[t_idx] = line_blocked

        # print(f"Time {t}s 目标是否被遮挡: {is_obscured[t_idx]}")

    total_coverage = sum(is_obscured[t_idx] * time_step for t_idx in range(num_time_steps)) 

    # --- 8. Print Results ---
    if verbose:
        print("Optimization finished.")
        print(f"Total effective obscuration time: {total_coverage:.2f} seconds")
        
        time_blocks = dict.fromkeys(bombs,0)
        for j, k in bombs:
            for t_idx,blocked in bombs_job[j,k].items():
                if is_obscured[t_idx] and len(blocked) > 0:
                    # print(f"Bomb ({j},{k}) blocks missile {blocked} at time {time_steps[t_idx]}")
                    time_blocks[j,k] += time_step
        print(f"Total time blocks: {time_blocks}")

    return total_coverage

# --- Example Usage (corresponds to Problem 4) ---
if __name__ == '__main__':
    # Initial positions from the problem description
    missiles = {
        'M1': (20000, 0, 2000),
    }
    drones = {
        'FY1': (17800, 0, 1800),
        'FY2': (12000, 1400, 1400),
        'FY3': (6000, -3000, 700),
    }
    
    flight = {
        'FY1': (85.50, np.pi * 178 / 180),
        'FY2': (137.55, np.pi * 287.4 / 180),
        'FY3': (139.93, np.pi * 85.12 / 180),
    }

    bombs = {
        ('FY1', 0): (0.14, 2.72),
        ('FY2', 0): (4.76, 5.54),
        ('FY3', 0): (18.58, 3.67)
    }

    # To run for other problems, change the inputs here.
    # For example, for Problem 5:
    # missiles = {'M1':..., 'M2':..., 'M3':...}
    # drones = {'FY1':..., 'FY2':..., ... 'FY5':...}
    # num_bombs = 3
    
    solve_smoke_deployment(
        missiles_info=missiles,
        drones_info=drones,
        flight_info=flight,
        bombs_info=bombs,
        num_bombs_per_drone=1, 
        time_horizon=50,      # e.g., 80 seconds simulation
        time_step=0.01,         # Coarse time step to make it solvable
        verbose=True
    )