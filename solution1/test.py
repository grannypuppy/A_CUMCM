import numpy as np
from gurobipy import GRB, Model

# 1. 参数设置
g = 9.8
v_sink = 3.0
R_smoke = 10.0
t_effective = 20.0
dt = 0.1
total_time_steps = int(t_effective / dt) + 1

# 2. 模型初始化
model = Model("SmokeCoverageOptimization")
# model.params.NonConvex = 2
# model.params.LogToConsole = 1
# model.params.TimeLimit = 30
# model.params.NumericFocus = 3

# 3. 决策变量（关键修改：变量上下界使用固定数值）
theta = model.addVar(lb=0.0, ub=2 * np.pi, vtype=GRB.CONTINUOUS, name='theta')
speed = model.addVar(lb=70.0, ub=140.0, vtype=GRB.CONTINUOUS, name='speed')
droptime = model.addVar(lb=0.0, ub=100.0, vtype=GRB.CONTINUOUS, name='droptime')
fuzetime = model.addVar(lb=0.0, ub=100.0, vtype=GRB.CONTINUOUS, name='fuzetime')  # 5=droptime下界，13=10+3

# 三角函数辅助变量
cos_theta = model.addVar(lb=-1.0, ub=1.0, vtype=GRB.CONTINUOUS, name='cos_theta')
sin_theta = model.addVar(lb=-1.0, ub=1.0, vtype=GRB.CONTINUOUS, name='sin_theta')

# 二进制变量
is_blocked = model.addVars(total_time_steps, vtype=GRB.BINARY, name='is_blocked')

# 4. 约束（新增 fuzetime 与 droptime 的关系约束）
model.addGenConstrCos(theta, cos_theta, name="cos_theta_const")
model.addGenConstrSin(theta, sin_theta, name="sin_theta_const")

# 表达 fuzetime >= droptime 和 fuzetime <= droptime + 3.0 的逻辑
model.addConstr(fuzetime >= droptime, "fuzetime_after_droptime")  # 下界约束
# model.addConstr(fuzetime <= droptime + 3.0, "fuzetime_max_gap")  # 上界约束（替代原 ub=droptime+3.0）

# 5. 物理模型（省略，保持不变）
uav_init = np.array([17800.0, 0.0, 1800.0])
missile_init = np.array([20000.0, 0.0, 2000.0])
fake_target = np.array([0.0, 0.0, 0.0])
cyl_center = np.array([0.0, 200.0, 0.0])
r_cyl = 7.0
h_cyl = 10.0

missile_dir = (fake_target - missile_init) / np.linalg.norm(fake_target - missile_init)
vM = missile_dir * 300.0

vU_x = cos_theta * speed
vU_y = sin_theta * speed

P_release_x = uav_init[0] + vU_x * droptime
P_release_y = uav_init[1] + vU_y * droptime
P_release_z = uav_init[2]


# 6. 采样点（保持不变）
def sample_side_points(n_theta=360, n_z=2):
    thetas = np.linspace(0.0, 2*np.pi, n_theta, endpoint=False)
    zs = np.linspace(0.0, h_cyl, n_z)
    pts = []
    for z in zs:
        for theta in thetas:
            x = r_cyl * np.cos(theta)
            y = r_cyl * np.sin(theta) + cyl_center[1]
            pts.append([x, y, z])
    return np.array(pts)   # shape (n_theta*n_z, 3)


points = sample_side_points()
n_points = points.shape[0]
R2 = R_smoke * R_smoke
M = 1e5

# 7. 核心约束（保持不变）
is_point_blocked = model.addVars(total_time_steps, n_points, vtype=GRB.BINARY, name='is_point_blocked')

for k in range(total_time_steps):
    t = fuzetime + k * dt
    dt_explode = k * dt
    dt_drop = fuzetime - droptime

    dt_drop_sq = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f'dt_drop_sq_{k}')
    model.addConstr(dt_drop_sq == dt_drop * dt_drop, name=f'dt_drop_sq_const_{k}')

    explode_x = P_release_x + vU_x * dt_drop
    explode_y = P_release_y + vU_y * dt_drop
    explode_z = P_release_z - 0.5 * g * dt_drop_sq

    smoke_x = explode_x
    smoke_y = explode_y
    smoke_z = explode_z - v_sink * dt_explode

    missile_x = missile_init[0] + vM[0] * t
    missile_y = missile_init[1] + vM[1] * t
    missile_z = missile_init[2] + vM[2] * t

    for i in range(n_points):
        px, py, pz = points[i]

        ABx = px - missile_x
        ABy = py - missile_y
        ABz = pz - missile_z
        APx = smoke_x - missile_x
        APy = smoke_y - missile_y
        APz = smoke_z - missile_z

        ABx_sq = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f'ABx_sq_{k}_{i}')
        ABy_sq = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f'ABy_sq_{k}_{i}')
        ABz_sq = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f'ABz_sq_{k}_{i}')
        model.addConstr(ABx_sq == ABx * ABx, name=f'ABx_sq_const_{k}_{i}')
        model.addConstr(ABy_sq == ABy * ABy, name=f'ABy_sq_const_{k}_{i}')
        model.addConstr(ABz_sq == ABz * ABz, name=f'ABz_sq_const_{k}_{i}')
        ab2 = ABx_sq + ABy_sq + ABz_sq

        dot_AP_AB = ABx * APx + ABy * APy + ABz * APz

        s = model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name=f's_{k}_{i}')
        mul_s_ab2 = model.addVar(vtype=GRB.CONTINUOUS, name=f'mul_s_ab2_{k}_{i}')
        model.addConstr(mul_s_ab2 == s * ab2, name=f'mul_s_ab2_const_{k}_{i}')

        lhs_s_upper = model.addVar(vtype=GRB.CONTINUOUS, name=f'lhs_s_upper_{k}_{i}')
        lhs_s_lower = model.addVar(vtype=GRB.CONTINUOUS, name=f'lhs_s_lower_{k}_{i}')
        model.addConstr(lhs_s_upper == mul_s_ab2 - dot_AP_AB, name=f'lhs_s_upper_const_{k}_{i}')
        model.addConstr(lhs_s_lower == mul_s_ab2 - dot_AP_AB, name=f'lhs_s_lower_const_{k}_{i}')
        model.addConstr(lhs_s_upper <= M * (1 - is_point_blocked[k, i]), name=f's_upper_{k}_{i}')
        model.addConstr(lhs_s_lower >= -M * (1 - is_point_blocked[k, i]), name=f's_lower_{k}_{i}')

        Qx = model.addVar(vtype=GRB.CONTINUOUS, name=f'Qx_{k}_{i}')
        Qy = model.addVar(vtype=GRB.CONTINUOUS, name=f'Qy_{k}_{i}')
        Qz = model.addVar(vtype=GRB.CONTINUOUS, name=f'Qz_{k}_{i}')
        model.addConstr(Qx == missile_x + s * ABx, name=f'Qx_const_{k}_{i}')
        model.addConstr(Qy == missile_y + s * ABy, name=f'Qy_const_{k}_{i}')
        model.addConstr(Qz == missile_z + s * ABz, name=f'Qz_const_{k}_{i}')

        dx = smoke_x - Qx
        dy = smoke_y - Qy
        dz = smoke_z - Qz

        d2 = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f'd2_{k}_{i}')
        model.addConstr(d2 == dx * dx + dy * dy + dz * dz, name=f'd2_const_{k}_{i}')

        lhs_dist = model.addVar(vtype=GRB.CONTINUOUS, name=f'lhs_dist_{k}_{i}')
        model.addConstr(lhs_dist == d2 - R2, name=f'lhs_dist_const_{k}_{i}')
        model.addConstr(lhs_dist <= M * (1 - is_point_blocked[k, i]), name=f'distance_{k}_{i}')

    for i in range(n_points):
        model.addConstr(is_blocked[k] <= is_point_blocked[k, i], name=f'blocked_require_{k}_{i}')

# 8. 目标函数与求解
total_coverage = sum(is_blocked[k] * dt for k in range(total_time_steps))
model.setObjective(total_coverage, GRB.MAXIMIZE)
model.optimize()

# 9. 结果输出
status = model.status
status_map = {
    GRB.OPTIMAL: "最优解",
    GRB.INFEASIBLE: "无可行解",
    GRB.TIME_LIMIT: "求解超时",
    GRB.UNBOUNDED: "目标函数无界"
}
print(f"\n求解状态：{status_map.get(status, f'未知状态（码：{status}）')}")

if status == GRB.OPTIMAL:
    print("=" * 50)
    print("最优结果：")
    print(f"总遮蔽时间：{model.ObjVal:.2f} 秒")
    print(f"无人机角度：{theta.X:.4f} 弧度（≈{theta.X * 180 / np.pi:.2f}°）")
    print(f"无人机速度：{speed.X:.2f} 单位/秒")
    print(f"投放时间：{droptime.X:.2f} 秒")
    print(f"引爆时间：{fuzetime.X:.2f} 秒")
    print("=" * 50)
elif status == GRB.INFEASIBLE:
    model.computeIIS()
    model.write("infeasible.ilp")
    print("不可行约束已保存至 infeasible.ilp")
