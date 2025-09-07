import numpy as np

def run_simulation_for_viz(missiles_info, drones_info, flight_info, bombs_info, num_bombs_per_drone, time_horizon, time_step):
    """
    Runs the smoke bomb deployment simulation and returns a detailed log for visualization.
    """
    # --- 1. Constants and Initialization ---
    V_MISSILE = 300.0  # m/s
    V_SINK = 3.0  # m/s
    R_SMOKE = 10.0  # m
    G = 9.8  # m/s^2
    SMOKE_DURATION = 20.0 # s

    # Target Definition
    TARGET_CENTER_BASE = np.array([0, 200, 0])
    TARGET_RADIUS = 7.0
    TARGET_HEIGHT = 10.0

    # Time and Object Discretization
    time_steps = np.arange(0, time_horizon, time_step)
    
    # Generate 48 target points
    target_points = []
    for h in [0, TARGET_HEIGHT]:
        for i in range(24):
            angle = 2 * np.pi * i / 24
            x = TARGET_CENTER_BASE[0] + TARGET_RADIUS * np.cos(angle)
            y = TARGET_CENTER_BASE[1] + TARGET_RADIUS * np.sin(angle)
            z = TARGET_CENTER_BASE[2] + h
            target_points.append(np.array([x, y, z]))

    # --- 3. Process Input Parameters ---
    drone_speed = {j: 120.0 for j in drones_info.keys()}
    drone_angle = {j: np.pi for j in drones_info.keys()}
    for j, (sp, theta) in flight_info.items():
        drone_speed[j] = sp
        drone_angle[j] = theta
    
    drone_cos_angle = {j: np.cos(drone_angle[j]) for j in drones_info.keys()}
    drone_sin_angle = {j: np.sin(drone_angle[j]) for j in drones_info.keys()}

    bombs = [(j, k) for j in drones_info.keys() for k in range(num_bombs_per_drone)]
    t_drop = {b: 1.5 for b in bombs}
    fusetime = {b: 3.6 for b in bombs}
    for (j, k), (t_drop_val, fusetime_val) in bombs_info.items():
        t_drop[j, k] = t_drop_val
        fusetime[j, k] = fusetime_val

    t_det = {(j,k): t_drop[j,k] + fusetime[j,k] for j, k in bombs}
    
    simulation_log = []

    # --- Main Simulation Loop ---
    for t_idx, t in enumerate(time_steps):
        
        # --- Calculate positions at time t ---
        
        # Missile positions
        missile_pos = {}
        for i, p0_i in missiles_info.items():
            p0_vec = np.array(p0_i)
            dist_to_target = np.linalg.norm(p0_vec)
            time_to_target = dist_to_target / V_MISSILE
            if t < time_to_target:
                 missile_pos[i] = p0_vec * (1 - (V_MISSILE * t) / dist_to_target)
            else:
                 missile_pos[i] = np.array([0,0,0])

        # Drone positions
        drone_pos = {}
        for j, p0_j in drones_info.items():
            drone_pos[j] = np.array([
                p0_j[0] + drone_speed[j] * drone_cos_angle[j] * t,
                p0_j[1] + drone_speed[j] * drone_sin_angle[j] * t,
                p0_j[2]
            ])

        # Cloud positions (only for active clouds)
        active_clouds = {}
        for j, k in bombs:
            if t_det[j,k] <= t < t_det[j,k] + SMOKE_DURATION:
                p_drop_x = drones_info[j][0] + drone_speed[j] * drone_cos_angle[j] * t_drop[j,k]
                p_drop_y = drones_info[j][1] + drone_speed[j] * drone_sin_angle[j] * t_drop[j,k]
                p_det_x = p_drop_x + drone_speed[j] * drone_cos_angle[j] * fusetime[j,k]
                p_det_y = p_drop_y + drone_speed[j] * drone_sin_angle[j] * fusetime[j,k]
                p_det_z = drones_info[j][2] - 0.5 * G * fusetime[j,k]**2
                
                cloud_pos = np.array([
                    p_det_x,
                    p_det_y,
                    p_det_z - V_SINK * (t - t_det[j,k])
                ])
                active_clouds[(j,k)] = cloud_pos
        
        # --- Obscuration Logic ---
        is_fully_obscured = 1
        lines_of_sight_status = []

        for i in missiles_info.keys():
            for p_target in target_points:
                line_blocked_by_any_cloud = 0
                missile_A = missile_pos[i]
                target_B = p_target
                AB = target_B - missile_A 
                
                for cloud_C in active_clouds.values():
                    AC = cloud_C - missile_A
                    
                    # Project AC onto AB to find the closest point on the line AB
                    s = np.dot(AC, AB) / np.dot(AB, AB)
                    
                    dist_sq = 0
                    if s < 0:
                        dist_sq = np.dot(AC, AC)
                    elif s > 1:
                        BC = cloud_C - target_B
                        dist_sq = np.dot(BC, BC)
                    else:
                        projection = missile_A + s * AB
                        dist_sq = np.dot(cloud_C - projection, cloud_C - projection)

                    if dist_sq <= R_SMOKE**2:
                        line_blocked_by_any_cloud = 1
                        break # This line is blocked, no need to check other clouds
                
                lines_of_sight_status.append({
                    'missile': i,
                    'start': missile_A,
                    'end': p_target,
                    'is_blocked': bool(line_blocked_by_any_cloud)
                })

                if not line_blocked_by_any_cloud:
                    is_fully_obscured = 0
                    break # One line is not blocked, so the target is not fully obscured
            if not is_fully_obscured:
                break
        
        # --- Store log for this time step ---
        simulation_log.append({
            'time': t,
            'missiles': missile_pos,
            'drones': drone_pos,
            'active_clouds': active_clouds,
            'is_fully_obscured': bool(is_fully_obscured),
            # Note: Storing all lines of sight for all steps can be memory intensive.
            # For animation, we usually only need the positions and the final obscured status.
            # This detailed line status is more for static "explainer" visualizations.
            'lines_of_sight_status': lines_of_sight_status if not is_fully_obscured else [] # To save space
        })

    # --- 8. Return Results ---
    print("Simulation for visualization finished.")
    total_coverage = sum(
        step['is_fully_obscured'] * time_step for step in simulation_log
    )
    print(f"Total effective obscuration time: {total_coverage:.2f} seconds")

    static_info = {
        'target_center_base': TARGET_CENTER_BASE,
        'target_radius': TARGET_RADIUS,
        'target_height': TARGET_HEIGHT,
        'target_points': target_points,
        'smoke_radius': R_SMOKE,
        'missiles_info': missiles_info,
        'drones_info': drones_info,
    }

    return simulation_log, static_info

# --- Example Usage (corresponds to Problem 4) ---
if __name__ == '__main__':
    # Initial positions from the problem description
    missiles = {
        'M1': (20000, 0, 2000),
    }
    drones = {
        'FY1': (17800, 0, 1800),
        # 'FY2': (12000, 1400, 1400),
        # 'FY3': (6000, -3000, 700),
    }
    
    flight = {
        'FY1': (139.14, np.pi * 188.67 / 180),
    }

    bombs = {
        ('FY1', 0): (0, 2.80),
        ('FY1', 1): (4.72, 6.17),
        ('FY1', 2): (10.00, 5.05)
    }

    # To run for other problems, change the inputs here.
    # For example, for Problem 5:
    # missiles = {'M1':..., 'M2':..., 'M3':...}
    # drones = {'FY1':..., 'FY2':..., ... 'FY5':...}
    # num_bombs = 3
    
    log, static_data = run_simulation_for_viz(
        missiles_info=missiles,
        drones_info=drones,
        flight_info=flight,
        bombs_info=bombs,
        num_bombs_per_drone=3, 
        time_horizon=50,      # e.g., 80 seconds simulation
        time_step=0.5         # Using a larger time step for faster testing
    )

    # Example of how to access the log data
    print(f"Generated {len(log)} log entries.")
    if log:
        print("First time step log:")
        print(log[0])
        print("\nLast time step log:")
        print(log[-1])