import numpy as np
import plotly.graph_objects as go

# --- 模式选择: 'concept' (当前静态示意) 或 'paper'（论文公式驱动动画） ---
MODE = 'paper'  # 可切换为 'concept'

# --- 通用常量（两种模式共享） ---
R_SMOKE = 12.0  # 烟雾半径
TARGET_CENTER_BASE = np.array([0, 0, 0])  # 圆柱基座放在地平面
TARGET_RADIUS = 10.0
TARGET_HEIGHT = 20.0
NUM_POINTS_ON_CIRCLE = 24

# --- 论文公式函数 ---
def sample_target_points_top_bottom(center_base, radius, height, num_points):
    points = []
    for h in [center_base[2], center_base[2] + height]:
        for i in range(num_points):
            angle = 2 * np.pi * i / num_points
            x = center_base[0] + radius * np.cos(angle)
            y = center_base[1] + radius * np.sin(angle)
            z = h
            points.append(np.array([x, y, z]))
    return points

def smoke_center_at_time(t, uav_init, v_fy, theta, tau1, tau2, v_sink=3.0, g=9.8):
    """依据论文：投放点->起爆点->沉降，返回 t 时刻烟雾中心坐标；若 t<tau2 返回 None。"""
    # 投放点
    rx = uav_init[0] + v_fy * np.cos(theta) * tau1
    ry = uav_init[1] + v_fy * np.sin(theta) * tau1
    rz = uav_init[2]
    # 起爆点
    td = tau2 - tau1
    ex = rx + v_fy * np.cos(theta) * td
    ey = ry + v_fy * np.sin(theta) * td
    ez = rz - 0.5 * g * (td ** 2)
    if t < tau2:
        return None
    # t 时刻中心（仅Z变化）
    sz = ez - v_sink * (t - tau2)
    return np.array([ex, ey, sz])

def missile_pos_at_time(t, missile_init, v_m_vec):
    return missile_init + v_m_vec * t

def compute_block_status_for_point(missile_pos, target_point, smoke_centers, r_smoke):
    """按论文分段投影s计算是否被任一烟雾遮蔽。"""
    A = target_point - missile_pos  # PMiP0
    A_sq = float(np.dot(A, A))
    if A_sq == 0.0:
        return False
    for S in smoke_centers:
        if S is None:
            continue
        AP = S - missile_pos  # PMiPSk
        # 三段式 s：端点在段外的情形
        # 若 PMiPSk·PMiP0 <= 0 -> s=0
        # 若 P0PSk·P0PMi <= 0 -> s=1  (等价 (S-P0)·(P0-m) <= 0)
        if np.dot(AP, A) <= 0:
            s = 0.0
        elif np.dot(S - target_point, -A) <= 0:
            s = 1.0
        else:
            s = float(np.dot(AP, A) / A_sq)
        Q = missile_pos + s * A
        d_sq = float(np.dot(S - Q, S - Q))
        if d_sq <= r_smoke * r_smoke:
            return True
    return False

def build_cylinder_surface(center_base, radius, height, n_r=50, n_h=50):
    z = np.linspace(center_base[2], center_base[2] + height, n_h)
    theta = np.linspace(0, 2 * np.pi, n_r)
    theta, z_mesh = np.meshgrid(theta, z)
    x = center_base[0] + radius * np.cos(theta)
    y = center_base[1] + radius * np.sin(theta)
    return x, y, z_mesh

def create_sphere_surface(center, radius, color='rgba(128, 128, 128, 0.3)'):
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 60)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
    return go.Surface(x=x, y=y, z=z, colorscale=[[0, color], [1, color]], showscale=False, hoverinfo='skip')

# --- 概念模式渲染（保留现有静态示意） ---
def render_concept():
    target_points = sample_target_points_top_bottom(TARGET_CENTER_BASE, TARGET_RADIUS, TARGET_HEIGHT, NUM_POINTS_ON_CIRCLE)
    missile_pos = np.array([80, 20, 35])
    active_smoke_clouds = [
        np.array([45, 5, 28]),
        np.array([40, 30, 15]),
    ]

    fig = go.Figure()
    # 地面
    plane_size = 60
    fig.add_trace(go.Surface(x=[-plane_size, plane_size], y=[-plane_size, plane_size], z=[[0, 0], [0, 0]],
                             colorscale=[[0, 'lightgrey'], [1, 'lightgrey']], showscale=False, opacity=0.5, hoverinfo='none'))
    # 圆柱
    cx, cy, cz = build_cylinder_surface(TARGET_CENTER_BASE, TARGET_RADIUS, TARGET_HEIGHT)
    fig.add_trace(go.Surface(x=cx, y=cy, z=cz, colorscale=[[0, 'lightgrey'], [1, 'lightgrey']], showscale=False, opacity=0.3, hoverinfo='none'))
    # 目标点
    t_np = np.array(target_points)
    fig.add_trace(go.Scatter3d(x=t_np[:,0], y=t_np[:,1], z=t_np[:,2], mode='markers', marker=dict(size=1, color='#6082B6'), name='目标点'))
    # 导弹
    fig.add_trace(go.Scatter3d(x=[missile_pos[0]], y=[missile_pos[1]], z=[missile_pos[2]], mode='markers', marker=dict(size=8, color='black', symbol='diamond'), name='导弹'))
    # 烟雾球
    for _c in active_smoke_clouds:
        fig.add_trace(create_sphere_surface(_c, R_SMOKE))
        fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode='markers', marker=dict(color='gray', size=10, opacity=0.4), name='烟雾球'))
    # 视线
    blocked_x, blocked_y, blocked_z, clear_x, clear_y, clear_z = [], [], [], [], [], []
    for P0 in target_points:
        A = P0 - missile_pos
        A_sq = float(np.dot(A, A))
        is_blocked = False
        if A_sq > 0:
            for S in active_smoke_clouds:
                AP = S - missile_pos
                if np.dot(AP, A) <= 0:
                    s = 0.0
                elif np.dot(S - P0, -A) <= 0:
                    s = 1.0
                else:
                    s = float(np.dot(AP, A) / A_sq)
                Q = missile_pos + s * A
                if float(np.dot(S - Q, S - Q)) <= R_SMOKE * R_SMOKE:
                    is_blocked = True
                    break
        if is_blocked:
            blocked_x.extend([missile_pos[0], P0[0], None]); blocked_y.extend([missile_pos[1], P0[1], None]); blocked_z.extend([missile_pos[2], P0[2], None])
        else:
            clear_x.extend([missile_pos[0], P0[0], None]); clear_y.extend([missile_pos[1], P0[1], None]); clear_z.extend([missile_pos[2], P0[2], None])
    fig.add_trace(go.Scatter3d(x=blocked_x, y=blocked_y, z=blocked_z, mode='lines', line=dict(color='#8FBC8F', width=2), name='被遮蔽的视线'))
    fig.add_trace(go.Scatter3d(x=clear_x, y=clear_y, z=clear_z, mode='lines', line=dict(color='#BC8F8F', width=2), name='通畅的视线'))

    fig.update_layout(title='遮蔽原理概念可视化 (带垂直落差)',
                      scene=dict(xaxis=dict(title='X 轴', showticklabels=False),
                                 yaxis=dict(title='Y 轴', showticklabels=False),
                                 zaxis=dict(title='Z 轴', range=[0, 45], showticklabels=False),
                                 aspectmode='data',
                                 camera=dict(eye=dict(x=1.8, y=-1.8, z=0.8))),
                      legend_title='图例', margin=dict(l=0, r=0, b=0, t=40))
    fig.show()

# --- 论文模式渲染（动画） ---
def render_paper():
    # 参数（可按论文/实例调整）
    g = 9.8
    v_sink = 3.0
    v_fy = 70.0
    theta = np.deg2rad(185.0)
    tau1 = 0.5
    tau2 = 3.0
    uav_init = np.array([-50.0, -10.0, 30.0])

    # 导弹：从较远处以恒速飞向圆柱中心
    missile_init = np.array([80.0, 20.0, 35.0])
    to_center = (TARGET_CENTER_BASE + np.array([0.0, 0.0, TARGET_HEIGHT/2])) - missile_init
    to_center = to_center / np.linalg.norm(to_center)
    v_m_speed = 40.0
    v_m_vec = to_center * v_m_speed

    # 采样点：只取顶/底圆周
    target_points = sample_target_points_top_bottom(TARGET_CENTER_BASE, TARGET_RADIUS, TARGET_HEIGHT, NUM_POINTS_ON_CIRCLE)

    # 时间轴
    dt = 0.2
    t_start = 0.0
    t_end = tau2 + 20.0
    times = np.arange(t_start, t_end + 1e-9, dt)

    # 统计 Bl 与累计时长
    frames = []
    total_covered = 0.0

    # 静态底图
    fig = go.Figure()
    plane_size = 80
    fig.add_trace(go.Surface(x=[-plane_size, plane_size], y=[-plane_size, plane_size], z=[[0, 0], [0, 0]],
                             colorscale=[[0, 'lightgrey'], [1, 'lightgrey']], showscale=False, opacity=0.5, hoverinfo='none'))
    cx, cy, cz = build_cylinder_surface(TARGET_CENTER_BASE, TARGET_RADIUS, TARGET_HEIGHT)
    fig.add_trace(go.Surface(x=cx, y=cy, z=cz, colorscale=[[0, 'lightgrey'], [1, 'lightgrey']], showscale=False, opacity=0.3, hoverinfo='none'))
    t_np = np.array(target_points)
    fig.add_trace(go.Scatter3d(x=t_np[:,0], y=t_np[:,1], z=t_np[:,2], mode='markers', marker=dict(size=1, color='#6082B6'), name='目标点'))

    # 初始占位：导弹、烟雾球、两类视线
    missile_trace = go.Scatter3d(x=[missile_init[0]], y=[missile_init[1]], z=[missile_init[2]], mode='markers', marker=dict(size=8, color='black', symbol='diamond'), name='导弹')
    fig.add_trace(missile_trace)

    # 先放一个空烟雾（后续帧替换）
    empty_cloud = create_sphere_surface(np.array([0,0,-999]), R_SMOKE)
    fig.add_trace(empty_cloud)

    blocked_lines = go.Scatter3d(x=[], y=[], z=[], mode='lines', line=dict(color='#8FBC8F', width=2), name='被遮蔽的视线')
    clear_lines = go.Scatter3d(x=[], y=[], z=[], mode='lines', line=dict(color='#BC8F8F', width=2), name='通畅的视线')
    fig.add_trace(blocked_lines)
    fig.add_trace(clear_lines)

    # 逐帧生成
    for t in times:
        # 位置
        m_pos = missile_pos_at_time(t, missile_init, v_m_vec)
        S = smoke_center_at_time(t, uav_init, v_fy, theta, tau1, tau2, v_sink=v_sink, g=g)
        smoke_centers = [S] if S is not None else []

        # 遮蔽判定
        bx, by, bz, cx_, cy_, cz_ = [], [], [], [], [], []
        all_blocked = True
        for P0 in target_points:
            blocked = compute_block_status_for_point(m_pos, P0, smoke_centers, R_SMOKE)
            if blocked:
                bx.extend([m_pos[0], P0[0], None]); by.extend([m_pos[1], P0[1], None]); bz.extend([m_pos[2], P0[2], None])
            else:
                all_blocked = False
                cx_.extend([m_pos[0], P0[0], None]); cy_.extend([m_pos[1], P0[1], None]); cz_.extend([m_pos[2], P0[2], None])
        if all_blocked:
            total_covered += dt

        # 更新烟雾球表面
        if S is None:
            cloud_surface = create_sphere_surface(np.array([0,0,-999]), R_SMOKE)  # 放到地平面下以“隐藏”
        else:
            cloud_surface = create_sphere_surface(S, R_SMOKE)

        frame = go.Frame(data=[
            # 地面、圆柱、目标点不变 → 不放入 frame.data
            go.Scatter3d(x=[m_pos[0]], y=[m_pos[1]], z=[m_pos[2]], mode='markers', marker=dict(size=8, color='black', symbol='diamond')),  # 导弹
            cloud_surface,  # 烟雾
            go.Scatter3d(x=bx, y=by, z=bz, mode='lines', line=dict(color='#8FBC8F', width=2)),
            go.Scatter3d(x=cx_, y=cy_, z=cz_, mode='lines', line=dict(color='#BC8F8F', width=2)),
        ], name=f"{t:.2f}")
        frames.append(frame)

    fig.frames = frames

    # 播放控件与布局
    fig.update_layout(
        title=f"论文模型可视化 | 覆盖总时长(离散): {total_covered:.2f}s",
        scene=dict(xaxis=dict(title='X 轴', showticklabels=False),
                   yaxis=dict(title='Y 轴', showticklabels=False),
                   zaxis=dict(title='Z 轴', range=[0, 45], showticklabels=False),
                   aspectmode='data',
                   camera=dict(eye=dict(x=1.8, y=-1.8, z=0.8))),
        legend_title='图例',
        margin=dict(l=0, r=0, b=0, t=40),
        updatemenus=[{
            'type': 'buttons',
            'buttons': [
                {'args': [None, {'frame': {'duration': 60, 'redraw': True}, 'fromcurrent': True, 'transition': {'duration': 0}}], 'label': '播放', 'method': 'animate'},
                {'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate', 'transition': {'duration': 0}}], 'label': '暂停', 'method': 'animate'}
            ],
            'direction': 'left', 'pad': {'r': 10, 't': 70}, 'showactive': False, 'x': 0.05, 'y': 0, 'xanchor': 'right', 'yanchor': 'top'
        }],
        sliders=[{
            'steps': [
                {'args': [[f.name], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate', 'transition': {'duration': 0}}], 'label': f.name, 'method': 'animate'}
                for f in frames
            ],
            'active': 0,
            'transition': {'duration': 0},
            'x': 0.12, 'y': 0, 'currentvalue': {'font': {'size': 12}, 'prefix': 't=', 'visible': True, 'xanchor': 'left'},
            'len': 0.8
        }]
    )

    # 初始帧可见图元（对齐frames中顺序）
    if frames:
        first = frames[0].data
        # 将初始帧中的动态图元追加到静态图后面
        for d in first:
            fig.add_trace(d)
    fig.show()

if MODE == 'concept':
    render_concept()
else:
    render_paper()
