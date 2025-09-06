import numpy as np
import time
import math
import cma
import warnings

# 忽略 UserWarning
warnings.filterwarnings("ignore", category=UserWarning)

# -------------------- 全局常量与默认设置 --------------------
G = 9.8
R_SMOKE = 10.0
V_SINK = 3.0
T_EFFECTIVE = 20.0

UAV_INIT_POSITIONS = [
    np.array([17800.0, 0.0, 1800.0]),
    np.array([12000.0, 1400.0, 1400.0]),
    np.array([6000.0, -3000.0, 700.0])
]

MISSILE_INIT = np.array([20000.0, 0.0, 2000.0])
FAKE_TARGET = np.array([0.0, 0.0, 0.0])
CYL_CENTER = np.array([0.0, 200.0, 0.0])
R_CYL = 7.0
H_CYL = 10.0
N_THETA = 36
N_Z = 2
DT = 0.01
CHUNK_SIZE = 1024


# -------------------- 辅助函数 --------------------
def sample_side_points(n_theta, n_z, r_cyl, h_cyl, cyl_center):
    thetas = np.linspace(0.0, 2 * np.pi, n_theta, endpoint=False)
    zs = np.linspace(0.0, h_cyl, n_z)
    pts = []
    for z in zs:
        for theta in thetas:
            x = r_cyl * np.cos(theta)
            y = r_cyl * np.sin(theta) + cyl_center[1]
            pts.append([x, y, z])
    return np.array(pts)


def all_points_blocked_chunkwise(smoke_center, missile_pos, target_points, R_smoke_sq, chunk_size):
    A = missile_pos
    N = target_points.shape[0]
    for start in range(0, N, chunk_size):
        B = target_points[start:start + chunk_size]
        AB = B - A
        AP = smoke_center - A
        ab2 = np.sum(AB * AB, axis=1)
        dot_AP_AB = np.dot(AB, AP)
        s = np.zeros_like(dot_AP_AB)
        nonzero_mask = ab2 > 1e-12
        s[nonzero_mask] = dot_AP_AB[nonzero_mask] / ab2[nonzero_mask]
        s = np.clip(s, 0.0, 1.0)
        Q = A + (s[:, None] * AB)
        d2 = np.sum((smoke_center - Q) ** 2, axis=1)
        if np.any(d2 > R_smoke_sq): return False
    return True


# -------------------- 核心计算函数 --------------------
def calculate_multi_smoke_coverage(drone_deployments):
    u_M = (FAKE_TARGET - MISSILE_INIT) / np.linalg.norm(FAKE_TARGET - MISSILE_INIT)
    vM = u_M * 300.0

    def missile_pos(t):
        return MISSILE_INIT + vM * t

    smoke_center_funcs, explosion_times = [], []

    for i, deployment in enumerate(drone_deployments):
        direction = deployment['direction']
        speed = deployment['speed']
        dir_uav_xy = np.array(direction) / np.linalg.norm(direction)
        vU = np.array([dir_uav_xy[0] * speed, dir_uav_xy[1] * speed, 0.0])

        t_release = deployment['t_release']
        t_fuze = deployment['t_fuze']
        t_b = t_release + t_fuze
        explosion_times.append(t_b)

        uav_start_pos = UAV_INIT_POSITIONS[i]
        P_release = uav_start_pos + vU * t_release

        def bomb_position(t, _P_release=P_release, _t_release=t_release, _vU=vU):
            dt_local = t - _t_release
            return _P_release + np.array([_vU[0] * dt_local, _vU[1] * dt_local, -0.5 * G * (dt_local ** 2)])

        P_explode = bomb_position(t_b)

        def smoke_center_pos(t, _P_explode=P_explode, _t_b=t_b):
            if t < _t_b: return None
            return _P_explode + np.array([0.0, 0.0, -V_SINK * (t - _t_b)])

        smoke_center_funcs.append(smoke_center_pos)

    target_points = sample_side_points(N_THETA, N_Z, R_CYL, H_CYL, CYL_CENTER)
    R_smoke_sq = R_SMOKE * R_SMOKE
    scan_start_time, scan_end_time = min(explosion_times), max(explosion_times) + T_EFFECTIVE
    times = np.arange(scan_start_time, scan_end_time + 1e-9, DT)

    num_smokes = len(drone_deployments)
    individual_masks = [np.zeros(len(times), dtype=bool) for _ in range(num_smokes)]

    for i, t in enumerate(times):
        current_missile_pos = missile_pos(t)
        for j, smoke_func in enumerate(smoke_center_funcs):
            current_smoke_center = smoke_func(t)
            if current_smoke_center is not None and all_points_blocked_chunkwise(current_smoke_center,
                                                                                 current_missile_pos, target_points,
                                                                                 R_smoke_sq, CHUNK_SIZE):
                individual_masks[j][i] = True

    detailed_times = []
    for j in range(num_smokes):
        current_mask = individual_masks[j]
        other_masks = [m for k, m in enumerate(individual_masks) if k != j]
        other_masks_combined = np.logical_or.reduce(other_masks) if other_masks else np.zeros_like(current_mask)
        solo_mask = np.logical_and(current_mask, np.logical_not(other_masks_combined))
        solo_time = np.sum(solo_mask) * DT
        shared_mask = np.logical_and(current_mask, other_masks_combined)
        shared_time = np.sum(shared_mask) * DT
        detailed_times.append({"solo": solo_time, "shared": shared_time})

    combined_mask = np.logical_or.reduce(individual_masks)
    total_coverage_time = np.sum(combined_mask) * DT

    return {"total_coverage_time": total_coverage_time, "detailed_times": detailed_times}


# -------------------- CMA-ES 求解部分 --------------------
def objective_function_for_minimizer(params):
    angle1, speed1, t_rel1, fuz1, \
        angle2, speed2, t_rel2, fuz2, \
        angle3, speed3, t_rel3, fuz3 = params

    drone_deployments = [
        {'direction': np.array([math.cos(math.radians(angle1)), math.sin(math.radians(angle1))]), 'speed': speed1,
         't_release': t_rel1, 't_fuze': fuz1},
        {'direction': np.array([math.cos(math.radians(angle2)), math.sin(math.radians(angle2))]), 'speed': speed2,
         't_release': t_rel2, 't_fuze': fuz2},
        {'direction': np.array([math.cos(math.radians(angle3)), math.sin(math.radians(angle3))]), 'speed': speed3,
         't_release': t_rel3, 't_fuze': fuz3}
    ]
    results = calculate_multi_smoke_coverage(drone_deployments=drone_deployments)
    return -results['total_coverage_time']


def run_3_uavs_cma_optimizer():
    print("\n" + "=" * 50)
    print("开始为【3架无人机】场景执行 CMA-ES 优化...")
    print(f"  - FY1 初始位置: {UAV_INIT_POSITIONS[0]}")
    print(f"  - FY2 初始位置: {UAV_INIT_POSITIONS[1]}")
    print(f"  - FY3 初始位置: {UAV_INIT_POSITIONS[2]}")
    print("=" * 50)

    bounds_per_uav = [(0, 360), (70, 140), (0.1, 40), (0.1, 15)]
    bounds_tuples = bounds_per_uav * 3
    lower_bounds = [b[0] for b in bounds_tuples]
    upper_bounds = [b[1] for b in bounds_tuples]
    initial_guess = [178.61, 88.57, 0.61, 3.04, 279.27, 133.22, 4.37, 5.90, 84.65, 139.50, 18.73, 3.61]
    sigma0_scalar = 1
    stds_per_uav = [4, 4, 0.1, 0.1]
    stds_per_coordinate = stds_per_uav * 3

    print(f"使用的初始解 (12个参数): {initial_guess}")
    print(f"全局初始步长 (sigma0): {sigma0_scalar}")
    print(f"【自定义】各坐标轴初始步长 (CMA_stds): {stds_per_coordinate}")

    options = {
        'bounds': [lower_bounds, upper_bounds],
        'CMA_stds': stds_per_coordinate,
        'maxfevals': 50000,
        'tolfun': 1e-9,
        'tolx': 1e-9,
        'verbose': 1,
    }

    xbest, es = cma.fmin2(
        objective_function_for_minimizer,
        initial_guess,
        sigma0_scalar,
        options=options
    )

    print("\n" + "*" * 50)
    print("【3架无人机】场景 CMA-ES 优化完成！")

    best_params = xbest
    max_coverage_time = -es.result.fbest
    angle1, speed1, t_rel1, fuz1, \
        angle2, speed2, t_rel2, fuz2, \
        angle3, speed3, t_rel3, fuz3 = best_params

    best_deployments = [
        {'direction': np.array([math.cos(math.radians(angle1)), math.sin(math.radians(angle1))]), 'speed': speed1,
         't_release': t_rel1, 't_fuze': fuz1},
        {'direction': np.array([math.cos(math.radians(angle2)), math.sin(math.radians(angle2))]), 'speed': speed2,
         't_release': t_rel2, 't_fuze': fuz2},
        {'direction': np.array([math.cos(math.radians(angle3)), math.sin(math.radians(angle3))]), 'speed': speed3,
         't_release': t_rel3, 't_fuze': fuz3}
    ]
    final_details = calculate_multi_smoke_coverage(drone_deployments=best_deployments)
    detailed_times = final_details['detailed_times']

    print(f"\n  最大总有效遮蔽时间: {max_coverage_time:.4f} s")
    print("\n  对应的最优参数组合:")
    params_per_drone = [
        (angle1, speed1, t_rel1, fuz1),
        (angle2, speed2, t_rel2, fuz2),
        (angle3, speed3, t_rel3, fuz3)
    ]

    for i, (params, details) in enumerate(zip(params_per_drone, detailed_times)):
        angle, speed, t_rel, fuz = params
        total_individual_time = details['solo'] + details['shared']
        print(f"    - 无人机 FY{i + 1} (初始位置: {UAV_INIT_POSITIONS[i]}) (总参与时长: {total_individual_time:.4f}s):")
        print(f"      - 飞行参数: 方向角度={angle:.2f} 度, 飞行速度={speed:.2f} m/s")
        print(f"      - 投弹参数: 投放时间={t_rel:.2f}s, 引信延时={fuz:.2f}s")
        print(f"      - 【单独遮蔽时长】: {details['solo']:.4f} s")
        print(f"      - 【协同遮蔽时长】: {details['shared']:.4f} s")

    # 【修改 1】新增一个部分，用于打印完整的、未经处理的12个参数列表
    print("\n  --------------------------------------------------")
    print(f"  【完整参数列表】 (可用于下次初始解):")
    # .tolist() 将 numpy array 转换为 python list，显示更友好
    print(f"  {best_params.tolist()}")
    print("  --------------------------------------------------")

    print(f"\n    - 停止原因: {es.result.stop.get('reason', 'NA')}")
    print("*" * 50)

    # 【修改 2】让函数返回关键结果，而不仅仅是打印
    return best_params, max_coverage_time, final_details


# -------------------- 主程序入口 --------------------
if __name__ == "__main__":
    # 【修改 3】调用函数并接收其返回的结果
    best_parameters, total_time, detailed_results = run_3_uavs_cma_optimizer()

    # 现在您可以在程序主体部分使用这些返回的变量
    # 例如，可以再次打印一个简洁的摘要
    print("\n--- 函数返回结果摘要 ---")
    print(f"捕获到的最优参数 (列表格式): {best_parameters.tolist()}")
    print(f"捕获到的最大遮蔽时间: {total_time:.4f} s")
    print(f"捕获到的详细时间分析: {detailed_results['detailed_times']}")