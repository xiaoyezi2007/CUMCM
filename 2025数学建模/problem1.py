import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

# --- 辅助函数定义 ---
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def point_segment_distance(p, a, b):
    """
    计算点p到线段ab的距离。
    p, a, b 均为numpy array形式的3D坐标。
    """
    ab = b - a
    ap = p - a
    # np.dot(ab, ab) might be zero if a and b are the same point.
    # Add a small epsilon to avoid division by zero.
    dot_product = np.dot(ab, ab)
    if dot_product < 1e-9:
        return np.linalg.norm(p - a)

    t = np.dot(ap, ab) / dot_product

    if t < 0.0:
        return np.linalg.norm(p - a)
    elif t > 1.0:
        return np.linalg.norm(p - b)
    else:
        projection = a + t * ab
        return np.linalg.norm(p - projection)


def generate_cylinder_points(center, radius, height, num_points=100):
    """
    在圆柱体表面进行均匀采样。

    与原始方法不同，此函数根据侧面和底面的实际表面积，
    按比例分配采样点的数量，从而实现真正的均匀分布。

    参数:
    center (np.array): 圆柱体下底面的圆心坐标 [x, y, z]。
    radius (float): 圆柱体的半径。
    height (float): 圆柱体的高度。
    num_points (int): 要生成的总点数。

    返回:
    np.array: 一个 (num_points, 3) 的Numpy数组，包含点的坐标。
    """
    # 1. 计算各个部分的表面积
    area_side = 2 * np.pi * radius * height
    area_cap = np.pi * radius ** 2
    total_area = area_side + 2 * area_cap

    # 2. 根据面积比例，计算分配到每个部分的点数
    # 使用round()确保点数为整数，并将剩余点数加到最大的区域（侧面）
    num_side = int(round(num_points * (area_side / total_area)))
    num_cap_each = int(round(num_points * (area_cap / total_area)))

    # 由于取整可能会有误差，重新分配以确保总点数不变
    num_caps_total = num_points - num_side
    num_top_cap = num_caps_total // 2
    num_bot_cap = num_caps_total - num_top_cap

    points = []

    # 3. 在圆柱体侧面生成点
    for _ in range(num_side):
        # 在 [0, 2*pi] 范围内均匀选择角度
        angle = np.random.uniform(0, 2 * np.pi)
        # 在 [0, height] 范围内均匀选择高度
        z = center[2] + np.random.uniform(0, height)

        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)
        points.append([x, y, z])

    # 4. 在圆柱体上底面生成点
    z_top = center[2] + height
    for _ in range(num_top_cap):
        # 在 [0, 2*pi] 范围内均匀选择角度
        angle = np.random.uniform(0, 2 * np.pi)
        # 为了保证点在圆内均匀分布，半径r的取值需要开根号
        # 这是因为圆的面积与半径的平方成正比
        r = radius * np.sqrt(np.random.uniform(0, 1))

        x = center[0] + r * np.cos(angle)
        y = center[1] + r * np.sin(angle)
        points.append([x, y, z_top])

    # 5. 在圆柱体下底面生成点
    z_bot = center[2]
    for _ in range(num_bot_cap):
        angle = np.random.uniform(0, 2 * np.pi)
        r = radius * np.sqrt(np.random.uniform(0, 1))

        x = center[0] + r * np.cos(angle)
        y = center[1] + r * np.sin(angle)
        points.append([x, y, z_bot])

    return np.array(points)


def visualize_cylinder(points):
    """
    将生成的圆柱体点进行三维可视化 (美化增强版)。
    """
    print("正在显示圆柱体采样点三维图，请关闭图形窗口以继续仿真...")

    # 使用更现代的绘图风格
    plt.style.use('seaborn-v0_8-whitegrid')

    fig = plt.figure(figsize=(10, 8))  # 使用更大的画布
    ax = fig.add_subplot(111, projection='3d')

    # --- 核心美化：使用颜色映射和透明度 ---
    # 根据Z坐标(高度)给每个点上色，使用viridis色谱
    scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                         c=points[:, 2],  # 颜色值映射到Z坐标
                         cmap='viridis',  # 使用viridis色谱
                         marker='.',
                         s=5,  # 点的大小可以稍大一些
                         alpha=0.6)  # 设置透明度

    # --- 添加结构辅助线，增强立体感 ---
    # 1. 计算圆柱体的基本参数
    center_x = (points[:, 0].max() + points[:, 0].min()) / 2
    center_y = (points[:, 1].max() + points[:, 1].min()) / 2
    z_min, z_max = points[:, 2].min(), points[:, 2].max()
    radius = np.sqrt((points[:, 0] - center_x) ** 2 + (points[:, 1] - center_y) ** 2).max()

    # 2. 绘制中心轴
    # ax.plot([center_x, center_x], [center_y, center_y], [z_min, z_max],
    #         color='red', linestyle='--', linewidth=2, label='中心轴')

    # 3. 绘制上下底面轮廓
    theta = np.linspace(0, 2 * np.pi, 100)
    x_circle = center_x + radius * np.cos(theta)
    y_circle = center_y + radius * np.sin(theta)
    ax.plot(x_circle, y_circle, z_min, color='gray', linestyle='-', linewidth=1.5, label='bottom')
    ax.plot(x_circle, y_circle, z_max, color='gray', linestyle='-', linewidth=1.5)

    # --- 优化标签、标题和图例 ---
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_zlabel('Z (m)', fontsize=12)
    ax.set_title('cylinder frame', fontsize=16, pad=20)
    ax.legend(loc='upper left')

    # 添加颜色条，说明颜色与高度的对应关系
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label('Z (m)', fontsize=10)

    # --- 设置背景和视角 ---
    # 改变背景面板颜色
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((0.95, 0.95, 0.95, 1.0))  # 让Z平面有淡淡的底色

    # 设置一个更美观的初始视角
    ax.view_init(elev=25., azim=-135)

    # 保持坐标轴比例一致
    max_range = np.array([points[:, 0].max() - points[:, 0].min(),
                          points[:, 1].max() - points[:, 1].min(),
                          points[:, 2].max() - points[:, 2].min()]).max() / 2.0
    mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
    mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
    mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.show()

valid = []

def simulate(t1=0.753, t2=0.247, t3=20):
    # --- 初始状态和参数定义 ---
    ANS = 0

    # 时间和速度
    # 根据效率优化建议，将时间步长从0.001增加到0.05，可大幅提升计算速度
    INTERVAL = 0.01
    V_M1 = 300.0
    V_FY1 = 125.0
    A_G_Z = -9.8

    # 初始位置
    P_M1_start = np.array([20000.0, 0.0, 2000.0])
    P_FY1_start = np.array([17800.0, 0, 1800.0])

    # 目标位置
    TARGET_M1 = np.array([0.0, 0.0, 0.0])
    TARGET_FY1 = np.array([0.0, 0.0, P_FY1_start[2]])

    # 圆柱体参数
    CYLINDER_CENTER_BASE = np.array([0.0, 200.0, 0.0])
    CYLINDER_RADIUS = 7.0
    CYLINDER_HEIGHT = 10.0

    temp_alpha = 0.10
    temp_alpha = np.radians(temp_alpha)

    # 预计算单位方向向量
    DIR_M1 = (TARGET_M1 - P_M1_start) / np.linalg.norm(TARGET_M1 - P_M1_start)
    #DIR_FY1 = np.array([-math.cos(temp_alpha), math.sin(temp_alpha), 0.0])  # (0.9888, 0.1491, 0.0)
    DIR_FY1 = np.array([0.9959, 0.0904, 0.0])

    # 预生成圆柱体表面采样点并进行可视化
    cylinder_surface_points = generate_cylinder_points(
        CYLINDER_CENTER_BASE, CYLINDER_RADIUS, CYLINDER_HEIGHT
    )
    #visualize_cylinder(cylinder_surface_points)

    # --- 模拟开始 ---
    # --- 第一阶段 ---
    t_phase1_start = 0.0
    t_phase1_end = t_phase1_start + t1
    t_phase1_duration = t_phase1_end - t_phase1_start
    pos_m1_p1 = P_M1_start + DIR_M1 * V_M1 * t_phase1_duration
    pos_fy1_p1 = P_FY1_start + DIR_FY1 * V_FY1 * t_phase1_duration

    print(f"--- 第一阶段结束 (t = {t_phase1_duration}s) ---")
    print(f"M1 坐标: {np.round(pos_m1_p1, 2)}")
    print(f"FY1 坐标 (即G的投放点): {np.round(pos_fy1_p1, 2)}")
    print("\n" + "=" * 40 + "\n")

    # --- 第二阶段 (干扰弹做抛体运动) ---
    t_phase2_start = t_phase1_end
    t_phase2_end = t_phase2_start + t2
    t_phase2_duration = t_phase2_end - t_phase2_start

    # M1 和 FY1 在 t=5.1s 的位置
    pos_m1_p2 = P_M1_start + DIR_M1 * V_M1 * t_phase2_end
    pos_fy1_p2 = P_FY1_start + DIR_FY1 * V_FY1 * t_phase2_end

    # === 【核心物理模型修正】 ===
    # 干扰弹(G)在投放后做抛体运动，其轨迹独立于无人机(FY1)。
    # G的初始位置是在 t=1.5s 时无人机的位置
    pos_g_launch = pos_fy1_p1
    # G的初始速度继承自无人机 (矢量)
    vel_g_launch = DIR_FY1 * V_FY1
    # 根据抛体运动公式计算G在第二阶段结束(t=5.1s)时的位置
    # 水平位移: pos_xy = pos_start_xy + v_xy * t
    pos_g_p2_xy = pos_g_launch[:2] + vel_g_launch[:2] * t_phase2_duration
    # 垂直位移: z = z0 + v0_z*t + 0.5*a*t^2 (垂直初速度v0_z为0)
    pos_g_p2_z = pos_g_launch[2] + 0.5 * A_G_Z * (t_phase2_duration ** 2)
    # 组合成G的最终坐标，即烟幕起爆点
    pos_g_p2 = np.array([pos_g_p2_xy[0], pos_g_p2_xy[1], pos_g_p2_z])

    print(f"--- 第二阶段结束 (t = {t2}s) ---")
    print(f"M1 坐标: {np.round(pos_m1_p2, 2)}")
    print(f"FY1 坐标: {np.round(pos_fy1_p2, 2)}")
    print(f"G 坐标 (烟幕起爆点): {np.round(pos_g_p2, 2)}")
    print("\n" + "=" * 40 + "\n")

    # --- 第三阶段 (烟幕云团下沉并进行遮蔽判断) ---
    t_phase3_start = t_phase2_end
    t_phase3_end = t_phase3_start + t3
    V_SMOKE_Z = -3.0  # 烟幕下沉速度

    # 记录烟幕云团开始下沉的初始位置
    pos_smoke_start = pos_g_p2

    print(f"--- 第三阶段模拟开始 (t = {t_phase3_start}s to {t_phase3_end}s, 步长={INTERVAL}s) ---")
    for t in np.arange(t_phase3_start, t_phase3_end, INTERVAL):
        # 计算当前时刻各物体位置
        pos_m1_current = P_M1_start + DIR_M1 * V_M1 * t

        t_since_detonation = t - t_phase3_start
        pos_smoke_current = pos_smoke_start + np.array([0, 0, V_SMOKE_Z * t_since_detonation])

        # 执行遮蔽判定
        is_obscured = True
        for cyl_point in cylinder_surface_points:
            dist = point_segment_distance(pos_smoke_current, pos_m1_current, cyl_point)
            if dist > 10.0:
                is_obscured = False
                break

        if is_obscured:
            ANS += INTERVAL
            valid.append(t)

    print("\n" + "=" * 40 + "\n")
    print(f"模拟结束。")
    print(f"最终计算出的有效遮蔽总时长为: {ANS:.4f} 秒")
    print(f'区间为：[{valid[0]:.2f}, {valid[-1]:.2f}]')
    return ANS

def t1():
    drop_times_to_test = np.arange(0, 5.25, 0.25)
    obscuration_durations = []

    print("开始进行参数敏感性分析...")
    for i, t_d in enumerate(drop_times_to_test):
        print(f"正在计算: 投放时间 = {t_d:.2f}s ({i + 1}/{len(drop_times_to_test)})")
        duration = simulate(t1=float(t_d))
        obscuration_durations.append(duration)

    # 寻找最佳结果
    max_duration = max(obscuration_durations)
    best_t_drop = drop_times_to_test[np.argmax(obscuration_durations)]

    print(f"最佳投放时间点: {best_t_drop:.2f}秒")
    print(f"对应的最长有效遮蔽时长: {max_duration:.4f}秒")

    # --- 结果可视化 ---
    plt.figure(figsize=(12, 7))
    plt.plot(drop_times_to_test, obscuration_durations, marker='o', linestyle='-', label='bottom')

    # 突出显示最大值点
    plt.plot(best_t_drop, max_duration, 'ro', markersize=10, label=f'best: {max_duration:.2f}s')
    plt.annotate(f'({best_t_drop:.2f}s, {max_duration:.2f}s)',
                 xy=(best_t_drop, max_duration),
                 xytext=(best_t_drop + 0.3, max_duration - 0.1),
                 )

    plt.title('有效遮蔽时长 vs. 干扰弹投放时间', fontsize=16)
    plt.xlabel('第一阶段飞行时间 (投放时机 t_drop) / 秒', fontsize=12)
    plt.ylabel('有效遮蔽总时长 / 秒', fontsize=12)
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    simulate()
    # visualize_cylinder(generate_cylinder_points(
    #     np.array([0.0, 200.0, 0.0]), 7.0, 10.0, num_points=10000
    # ))