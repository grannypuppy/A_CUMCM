import numpy as np
import gurobipy as gp
from gurobipy import GRB

def solve_smoke_deployment(missiles_info, drones_info, num_bombs_per_drone, time_horizon, time_step):
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

    # --- 2. Gurobi Model Setup ---
    model = gp.Model("SmokeScreenOptimization")
    model.params.LogToConsole = 1

    # --- 3. Decision Variables ---
    # Drone flight parameters
    drone_speed = model.addVars(drones_info.keys(), lb=70, ub=140, name="drone_speed")
    drone_angle = model.addVars(drones_info.keys(), lb=-np.pi, ub=np.pi, name="drone_angle") # theta

    # We need to model cos and sin of the angle. Gurobi's general constraints are good for this.
    drone_cos_angle = model.addVars(drones_info.keys(), lb=-1, ub=1, name="drone_cos_angle")
    drone_sin_angle = model.addVars(drones_info.keys(), lb=-1, ub=1, name="drone_sin_angle")
    for j in drones_info.keys():
        model.addGenConstrSin(drone_angle[j], drone_sin_angle[j])
        model.addGenConstrCos(drone_angle[j], drone_cos_angle[j])

    # Bomb deployment variables
    bombs = [(j, k) for j in drones_info.keys() for k in range(num_bombs_per_drone)]
    t_drop = model.addVars(bombs, vtype=GRB.CONTINUOUS, lb=0, ub=time_horizon, name="t_drop")
    fusetime = model.addVars(bombs, vtype=GRB.CONTINUOUS, lb=0, name="fusetime")
    t_det = model.addVars(bombs, vtype=GRB.CONTINUOUS, lb=0, ub=time_horizon, name="t_det")

    # Binary variable to indicate if a time step is effectively obscured
    is_obscured = model.addVars(num_time_steps, vtype=GRB.BINARY, name="is_obscured")

    # --- 4. Objective Function ---
    # Maximize the total time the target is obscured
    model.setObjective(gp.quicksum(is_obscured[t_idx] for t_idx in range(num_time_steps)) * time_step, GRB.MAXIMIZE)

    # --- 5. Constraints ---
    # Link drop, fuse, and detonation times
    model.addConstrs((t_det[j,k] == t_drop[j,k] + fusetime[j,k] for j,k in bombs), "detonation_time")

    # Minimum 1s interval between drops from the same drone
    for j in drones_info.keys():
        for k1 in range(num_bombs_per_drone):
            for k2 in range(k1 + 1, num_bombs_per_drone):
                 # To model |t_drop1 - t_drop2| >= 1, we use a binary variable
                 b = model.addVar(vtype=GRB.BINARY)
                 M = time_horizon 
                 model.addConstr(t_drop[j, k1] - t_drop[j, k2] >= MIN_DROP_INTERVAL - M * b)
                 model.addConstr(t_drop[j, k2] - t_drop[j, k1] >= MIN_DROP_INTERVAL - M * (1 - b))

    # Obscuration Logic (this is the complex part)
    for t_idx, t in enumerate(time_steps):
        # This variable is 1 if all lines of sight are blocked at time t
        # all_lines_blocked_at_t = model.addVar(vtype=GRB.BINARY, name=f"all_blocked_at_t{t_idx}")
        
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
        
        # We need a binary var for each line of sight (missile, target_point) being blocked
        lines_blocked_this_step = []
        for i in missiles_info.keys():
            for p_idx, p_target in enumerate(target_points):
                line_blocked = model.addVar(vtype=GRB.BINARY, name=f"line_blocked_t{t_idx}_m{i}_p{p_idx}")
                lines_blocked_this_step.append(line_blocked)

                # This line is blocked if AT LEAST ONE cloud intercepts it.
                cloud_intercepts = []
                for j, k in bombs:
                    cloud_intercepts_this = model.addVar(vtype=GRB.BINARY, name=f"intercept_t{t_idx}_m{i}_p{p_idx}_b{j}{k}")
                    cloud_intercepts.append(cloud_intercepts_this)

                    # Cloud active condition
                    # cloud_is_active 应该等价于 (t_det[j,k] <= t) 且 (t <= t_det[j,k] + SMOKE_DURATION)
                    cloud_is_active = model.addVar(vtype=GRB.BINARY, name=f"cloud_is_active_{t_idx}_{j}{k}")
                    # 用Big-M方法将区间条件转为与二进制变量关联的约束
                    M = time_horizon * 2 # Use a more reasonable M
                    model.addConstr(t_det[j,k] - t <= M * (1 - cloud_is_active), name=f"active_start_{t_idx}_{j}{k}")
                    model.addConstr(t - (t_det[j,k] + SMOKE_DURATION) <= M * (1 - cloud_is_active), name=f"active_end_{t_idx}_{j}{k}")

                    # --- Position Calculations (Requires helper variables) ---
                    # Drone position at drop time
                    p_fy_drop_x = drones_info[j][0] + drone_speed[j] * drone_cos_angle[j] * t_drop[j,k]
                    p_fy_drop_y = drones_info[j][1] + drone_speed[j] * drone_sin_angle[j] * t_drop[j,k]
                    
                    # Detonation point
                    p_det_x = p_fy_drop_x + drone_speed[j] * drone_cos_angle[j] * fusetime[j,k]
                    p_det_y = p_fy_drop_y + drone_speed[j] * drone_sin_angle[j] * fusetime[j,k]
                    p_det_z = drones_info[j][2] - 0.5 * G * fusetime[j,k] * fusetime[j,k]

                    # Cloud center at time t
                    p_cloud_x = p_det_x
                    p_cloud_y = p_det_y
                    p_cloud_z = p_det_z - V_SINK * (t - t_det[j,k])

                    # --- Geometric check using the Parametric Line Segment ("moving point") method ---
                    # This method is generally more efficient for solvers than the vector projection method.
                    M = 1e5 # A large constant for the Big-M method

                    # 1. Define the line segment from missile (A) to target point (B)
                    missile_A = missile_pos[i]
                    target_B = p_target
                    AB = target_B - missile_A # This is a constant vector for the current loop iteration

                    # 2. Introduce a parameter 's' to define a moving point Q on the line segment AB
                    # Q = A + s * AB, where s is in [0, 1]
                    s = model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name=f"s_{t_idx}_{i}_{p_idx}_{j}{k}")

                    # Coordinates of the moving point Q. These are Gurobi expressions.
                    Qx = missile_A[0] + s * AB[0]
                    Qy = missile_A[1] + s * AB[1]
                    Qz = missile_A[2] + s * AB[2]

                    # 3. Calculate the vector from the moving point (Q) to the smoke cloud center (P)
                    dx = p_cloud_x - Qx
                    dy = p_cloud_y - Qy
                    dz = p_cloud_z - Qz

                    # 4. Apply the Big-M constraint.
                    # If cloud_intercepts_this is 1, then the squared distance d2 = dx^2+dy^2+dz^2 must be <= R_SMOKE^2.
                    # Otherwise, the constraint is relaxed by the large M term.
                    # The entire expression is quadratic, so we use addQConstr.
                    dist_sq = model.addVar(name=f"dist_sq_{t_idx}_{i}_{p_idx}_{j}{k}")
                    model.addConstr(dist_sq == (dx*dx + dy*dy + dz*dz), name=f"dist_sq_const_{t_idx}_{i}_{p_idx}_{j}{k}")
                    model.addConstr(dist_sq - R_SMOKE**2 <= M * (1 - cloud_intercepts_this),
                                     name=f"dist_check_{t_idx}_{i}_{p_idx}_{j}{k}")
                    
                    # Link to active status (This constraint remains from the previous logic)
                    model.addConstr(cloud_intercepts_this <= cloud_is_active)

                # The line is blocked if the sum of intercepts is >= 1
                model.addConstr(gp.quicksum(cloud_intercepts) >= line_blocked)

        # LOGIC FIX: The target is obscured if AT LEAST ONE line of sight is blocked.
        # The original logic required ALL lines to be blocked, which is too strict.
        model.addConstr(gp.quicksum(lines_blocked_this_step) >= is_obscured[t_idx])


    # --- 6. Set Initial Solution (Warm Start) ---
    print("--- Setting Initial Feasible Solution ---")
    try:
        # Drone FY1 parameters from user
        drone_j = 'FY1'
        bomb_k = 0 # Assuming the first bomb of the first drone

        # User-provided values
        initial_speed = 120.0
        initial_t_drop = 1.5
        initial_fusetime = 3.6
        # Direction "towards (0,0,z)" from a positive x-coordinate implies a flight angle of 180 degrees (pi radians).
        initial_angle_rad = np.pi

        # Set the .Start attribute for each core variable
        drone_speed[drone_j].Start = initial_speed
        drone_angle[drone_j].Start = initial_angle_rad
        t_drop[drone_j, bomb_k].Start = initial_t_drop
        fusetime[drone_j, bomb_k].Start = initial_fusetime

        # It's also helpful to set derived variables for a complete warm start
        t_det[drone_j, bomb_k].Start = initial_t_drop + initial_fusetime
        drone_cos_angle[drone_j].Start = np.cos(initial_angle_rad)
        drone_sin_angle[drone_j].Start = np.sin(initial_angle_rad)
        print("Initial solution has been set for Gurobi.")
    except Exception as e:
        print(f"Could not set initial solution (maybe drone/bomb doesn't exist in this scenario): {e}")


    # --- 7. Solve ---
    model.setParam('NonConvex', 2) # Important for quadratic constraints
    model.setParam('TimeLimit', 3000) # Set a time limit (e.g., 5 minutes)
    model.setParam('MIPGap', 0.1) # Stop when a 10% optimality gap is reached
    # model.setParam('NumericFocus', 2) # Increase numerical stability
    
    model.optimize()

    # --- 8. Print Results ---
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT or model.status == GRB.INTERRUPTED:
        if model.SolCount > 0:
            print("Optimization finished.")
            print(f"Total effective obscuration time: {model.ObjVal:.2f} seconds")
            
            for j in drones_info.keys():
                print(f"\n--- Drone {j} Strategy ---")
                print(f"  Flight Speed: {drone_speed[j].X:.2f} m/s")
                print(f"  Flight Angle: {np.rad2deg(drone_angle[j].X):.2f} degrees")
                for k in range(num_bombs_per_drone):
                    print(f"  Bomb {k+1}:")
                    print(f"    Drop Time: {t_drop[j,k].X:.2f} s")
                    print(f"    Fuse Time: {fusetime[j,k].X:.2f} s")
                    print(f"    Detonation Time: {t_det[j,k].X:.2f} s")
        else:
            print("Optimization stopped but no solution was found.")
    elif model.status == GRB.INFEASIBLE:
        print("Model is infeasible. Computing IIS to find conflicting constraints...")
        model.computeIIS()
        model.write("infeasible_model.ilp")
        print("IIS written to infeasible_model.ilp")
    else:
        print(f"No solution found. Status code: {model.status}")
        
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
    
    # To run for other problems, change the inputs here.
    # For example, for Problem 5:
    # missiles = {'M1':..., 'M2':..., 'M3':...}
    # drones = {'FY1':..., 'FY2':..., ... 'FY5':...}
    # num_bombs = 3
    
    solve_smoke_deployment(
        missiles_info=missiles,
        drones_info=drones,
        num_bombs_per_drone=1, 
        time_horizon=80,      # e.g., 80 seconds simulation
        time_step=0.5         # Coarse time step to make it solvable
    )