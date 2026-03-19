from typing import Any

import numpy as np
from shapely.geometry import Polygon, LineString, Point
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
from tqdm import tqdm


# --- (您的所有辅助函数和全局变量定义保持不变) ---
def point_segment_distance(p, a, b):
    ab = b - a
    if np.dot(ab, ab) == 0: return np.linalg.norm(p - a)
    ap = p - a
    t = np.dot(ap, ab) / np.dot(ab, ab)
    if t < 0.0:
        return np.linalg.norm(p - a)
    elif t > 1.0:
        return np.linalg.norm(p - b)
    else:
        projection = a + t * ab
        return np.linalg.norm(p - projection)


def generate_cylinder_points(center, radius, height, num_points):
    points = []
    num_side = int(num_points * 0.7)
    for _ in range(num_side):
        angle = np.random.uniform(0, 2 * np.pi)
        z = center[2] + np.random.uniform(0, height)
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)
        points.append([x, y, z])
    num_cap = (num_points - num_side) // 2
    for _ in range(num_cap):
        angle = np.random.uniform(0, 2 * np.pi)
        r = np.sqrt(np.random.uniform(0, 1)) * radius
        x_bot = center[0] + r * np.cos(angle)
        y_bot = center[1] + r * np.sin(angle)
        points.append([x_bot, y_bot, center[2]])
        x_top = center[0] + r * np.cos(angle)
        y_top = center[1] + r * np.sin(angle)
        points.append([x_top, y_top, center[2] + height])
    return np.array(points)


def calculate_ans_for_phase3(t1=0.00714286, t2=0, t3=20, temp_alpha=0, V_FY1=140):
    # print(t1,t2,t3,temp_alpha)
    # --- 初始状态和参数定义 ---
    valid = []
    ANS = 0

    # 时间和速度
    # 根据效率优化建议，将时间步长从0.001增加到0.05，可大幅提升计算速度
    INTERVAL = 0.01
    V_M1 = 300.0
    A_G_Z = -9.8

    # 初始位置
    P_M1_start = np.array([20000.0, 0.0, 2000.0])
    P_FY1_start = np.array([17800.0, 0.0, 1800.0])

    # 目标位置
    TARGET_M1 = np.array([0.0, 0.0, 0.0])
    TARGET_FY1 = np.array([0.0, 0.0, P_FY1_start[2]])

    # 圆柱体参数
    CYLINDER_CENTER_BASE = np.array([0.0, 200.0, 0.0])
    CYLINDER_RADIUS = 7.0
    CYLINDER_HEIGHT = 10.0

    temp_alpha = np.radians(temp_alpha)

    # 预计算单位方向向量
    DIR_M1 = (TARGET_M1 - P_M1_start) / np.linalg.norm(TARGET_M1 - P_M1_start)
    DIR_FY1 = np.array([-math.cos(temp_alpha), math.sin(temp_alpha), 0.0])  # (0.9888, 0.1491, 0.0)

    # --- 模拟开始 ---
    # --- 第一阶段 ---
    t_phase1_start = 0.0
    t_phase1_end = t_phase1_start + t1
    t_phase1_duration = t_phase1_end - t_phase1_start
    pos_m1_p1 = P_M1_start + DIR_M1 * V_M1 * t_phase1_duration
    pos_fy1_p1 = P_FY1_start + DIR_FY1 * V_FY1 * t_phase1_duration

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

    # --- 第三阶段 (烟幕云团下沉并进行遮蔽判断) ---
    t_phase3_start = t_phase2_end
    t_phase3_end = t_phase3_start + t3
    V_SMOKE_Z = -3.0  # 烟幕下沉速度

    # 记录烟幕云团开始下沉的初始位置
    pos_smoke_start = pos_g_p2

    # print(f"--- 第三阶段模拟开始 (t = {t_phase3_start}s to {t_phase3_end}s, 步长={INTERVAL}s) ---")
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

        elif len(valid) != 0:
            break
    if len(valid) == 0:
        ans = -1
    else:
        ans = valid[-1]
    return ANS, ans


mode = 0

V_INTERVAL = 10 if mode == 0 else 5
T_INTERVAL = 1 if mode == 0 else 0.1
ANGLE_STEP_DEG = 1
Z_STEP = 0.5
P3_INTERVAL = 0.1 if mode == 0 else 0.005
NUM_CYLINDER_POINTS = 100 if mode == 0 else 500

V_M1 = 300.0
V_G_Z_p3 = 3.0
A_G_Z = -9.8
P_M1_start = np.array([20000.0, 0.0, 2000.0])
P_FY1_start = np.array([17800.0, 0.0, 1800.0])
TARGET_M1 = np.array([0.0, 0.0, 0.0])
CYLINDER_CENTER = np.array([0.0, 200.0, 0.0])
DIR_M1 = (TARGET_M1 - P_M1_start) / np.linalg.norm(TARGET_M1 - P_M1_start)
cylinder_surface_points = generate_cylinder_points(
    CYLINDER_CENTER, 7.0, 10.0, NUM_CYLINDER_POINTS
)

global_max_ans = -1
global_best_params = []

v_min = 70
v_max = 140


def function1(x: int, V: int, alpha: float):
    return (3 * x) / V + 20 + (math.cos(alpha) / P_FY1_start[0]) * (P_FY1_start[2] - 20) * x + \
        (3 / (math.cos(math.atan2(P_M1_start[2], P_M1_start[0])) * V_M1)) * \
        (CYLINDER_CENTER[0] + (CYLINDER_CENTER[1] / (CYLINDER_CENTER[1] - x * math.sin(alpha))) * (
                P_FY1_start[0] - x * math.cos(alpha) - CYLINDER_CENTER[0]) - P_M1_start[0])


def function2(x: int, V: int):
    return (9.8 / (2 * V * V)) * x * x


def function3(x: int, alpha: float):
    return (math.cos(np.radians(alpha)) * x * (P_FY1_start[2] - 20)) / P_FY1_start[0] + 20


def function4(x: int, alpha: float):
    return math.cos(alpha) / P_FY1_start[0] * (P_FY1_start[2] - 20) * x + 20


# *** 新增：独立的绘图函数 ***
def plot_heatmap(alpha_range, v_range, results_grid):
    """
    绘制二维颜色矩阵（热力图）
    """
    print("\n" + "*" * 30)
    print("正在生成最终结果的热力图...")

    plt.figure(figsize=(10, 8))

    # --- 核心修改在此处 ---
    # 将 cmap 参数从 'inferno' (或默认) 修改为您喜欢的方案
    # 推荐选项: 'viridis', 'plasma', 'magma', 'cividis', 'gray'
    im = plt.imshow(results_grid.T, origin='lower', aspect='auto',
                    extent=[alpha_range.min(), alpha_range.max(), v_range.start, v_range.stop],
                    cmap='viridis')  # <--- 修改此处

    # 添加颜色条，并为其添加标签
    cbar = plt.colorbar()
    cbar.set_label('Max ANS (ans1+ans2+ans3)')

    # 设置图表标题和坐标轴标签
    plt.title('Max ANS Heatmap for Angle vs. Speed', fontsize=16)
    plt.xlabel('Angle (degrees)', fontsize=12)
    plt.ylabel('Speed (V_FY1)', fontsize=12)

    plt.show()


def check(a: float, b: float, tmp: float, v: float, alpha: float) -> bool:
    """
    根据给定的五个参数，计算M1和G点的瞬时位置，并进行成功判定。

    参数:
        a (float): 用于计算G点位置的参数，可理解为水平位移相关量。
        b (float): 用于计算G点位置的参数，可理解为初始Z坐标下降量。
        tmp (float): 绝对时间戳 (t)。
        v (float): FY系列飞机的速度 (V_FY)。
        alpha (float): FY系列飞机的飞行角度 (角度制)。

    返回:
        bool: 如果判定为 "yes"，返回 True；否则返回 False。
    """

    alpha_rad = math.radians(alpha)

    T = a / v
    t2 = math.sqrt(2 * b / 9.8)
    t1 = T - t2
    g_x, g_y, g_z = P_FY1_start[0], P_FY1_start[1], P_FY1_start[2]
    if tmp < t1:
        g_x -= v * tmp * math.cos(alpha_rad)
        g_y += v * tmp * math.sin(alpha_rad)
    elif tmp < T:
        g_x -= v * tmp * math.cos(alpha_rad)
        g_y += v * tmp * math.sin(alpha_rad)
        g_z -= 9.8 * (tmp - t1) * (tmp - t1) / 2
    else:
        # print("!")
        g_x -= v * T * math.cos(alpha_rad)
        g_y += v * T * math.sin(alpha_rad)
        g_z -= 9.8 * t2 * t2 / 2 + 3 * (tmp - T)
    # print(g_x)
    # print(g_y)
    # print(g_z)
    pos_g_current = np.array([g_x, g_y, g_z])

    # 3. 计算M1在 tmp 时刻的坐标
    pos_m1_current = P_M1_start + DIR_M1 * V_M1 * tmp

    # 4. 执行判定逻辑
    for cyl_point in cylinder_surface_points:
        dist = point_segment_distance(pos_g_current, pos_m1_current, cyl_point)
        # 如果任何一个距离大于等于10，则判定失败，立即返回 False
        if dist >= 10.0:
            return False

    # 如果循环正常结束，说明所有点都满足条件，判定成功，返回 True
    return True


def search():
    global global_max_ans, global_best_params, global_best_alpha, global_best_V
    ANGLE_INTERVAL = 0.1 if mode == 0 else 0.01
    buffer = 0.1

    # *** 新增：定义用于存储绘图数据的变量 ***
    alpha_range_deg = np.arange(0.0, 1, ANGLE_INTERVAL)
    v_range = range(v_min, v_max + 1, V_INTERVAL)
    # 创建一个网格来存储每个(alpha, V)组合的最大ANS (mx)
    results_grid = np.zeros((len(alpha_range_deg), len(v_range)))

    # *** 修改：使用enumerate来获取索引，以便填充数据网格 ***
    for i, alpha in enumerate(tqdm(alpha_range_deg, desc="Angle Progress")):
        alpha_rad = np.radians(alpha)
        for j, V in enumerate(v_range):
            mx = -1
            best = []
            for x1 in np.arange(0.0, 100, 1):
                # (您的核心计算逻辑完全不变)
                y1 = function1(x1, V, alpha_rad)
                y1 = max(y1, 0)
                y2 = function2(x1, V)
                if y1 > y2:
                    break
                y = min(y1, y2)
                p_g = np.array(
                    [P_FY1_start[0] - x1 * math.cos(alpha_rad), x1 * math.sin(alpha_rad), P_FY1_start[2] - y])
                if V == 0:
                    t = 0
                else:
                    t = x1 / V
                t1 = x1 / V - math.sqrt(2 * y / 9.8)
                ans1, tmp = calculate_ans_for_phase3(t1, t - t1, 20, alpha, V)

                print(f"alpha={alpha:.1f}°, V_FY1={V:.1f}, ANS1={ans1:.3f}, x1={x1:.2f},tmp = {tmp:.2f}")
                x2_l, x2_r = x1, 1500
                while x2_r - x2_l > 0.1:
                    mid = (x2_l + x2_r) / 2
                    y_mid = function1(mid, V, alpha_rad)
                    if mid / V - math.sqrt(2 * y_mid / 9.8) < t1 + 1 or tmp - (
                            function3(mid, alpha) - y_mid) / 3 > mid / V or check(mid, y_mid, tmp, V, alpha):
                        x2_l = mid
                    else:
                        x2_r = mid
                x2, y2 = x2_r + 5, function1(x2_r + 5, V, alpha_rad)
                p_g_2 = np.array(
                    [P_FY1_start[0] - x2 * math.cos(alpha_rad), x2 * math.sin(alpha_rad), P_FY1_start[2] - y2])
                t, t2 = x2 / V, x2 / V - math.sqrt(2 * y2 / 9.8)

                ans2, tmp = calculate_ans_for_phase3(t2, t - t2, 20, alpha, V)
                print(f"alpha={alpha:.1f}°, V_FY1={V:.1f}, ANS2={ans2:.3f}, x2={x2:.2f}")
                x3_l, x3_r = x2, 2000
                while x3_r - x3_l > 0.1:
                    mid = (x3_r + x3_l) / 2
                    y_mid = function1(mid, V, alpha_rad)
                    if mid / V - math.sqrt(2 * y_mid / 9.8) < t2 + 1 or tmp - (
                            function3(mid, alpha) - y_mid) / 3 > mid / V or check(mid, y_mid, tmp, V, alpha):
                        x3_l = mid
                    else:
                        x3_r = mid
                x3, y3 = x3_r + 5, function1(x3_r + 5, V, alpha_rad)

                p_g_3 = np.array(
                    [P_FY1_start[0] - x3 * math.cos(alpha_rad), x3 * math.sin(alpha_rad), P_FY1_start[2] - y3])
                t, t3 = x3 / V, x3 / V - math.sqrt(2 * y3 / 9.8)

                ans3, _ = calculate_ans_for_phase3(t3, t - t3, 20, alpha, V)
                print(f"alpha={alpha:.1f}°, V_FY1={V:.1f}, ANS3={ans3:.3f}, x3={x3:.2f}")
                if ans1 + ans2 + ans3 > mx:
                    mx = ans1 + ans2 + ans3
                    best.clear()
                    best.append([x1, y, x2, y2, x3, y3, ans1, ans2, ans3])

            # (您的打印逻辑完全不变)
            print(f"ALL:alpha={alpha:.3f}°, V_FY1={V:.1f}, max={mx:.3f}, best={best}")

            # *** 新增：将当前(alpha, V)算出的最大ANS (mx) 存入网格 ***
            if mx > -1:
                results_grid[i, j] = mx

            # (您的全局最优解更新逻辑完全不变)
            if mx > global_max_ans:
                global_max_ans = mx
                global_best_params = best
                global_best_alpha = alpha
                global_best_V = V

    # (您的最终结果打印逻辑完全不变)
    print(
        f"GLOBAL:bestalpha={global_best_alpha:.3f}°, bestV_FY1={global_best_V:.1f}, ANS={global_max_ans:.3f}, params={global_best_params}")

    # *** 新增：在所有计算结束后，调用绘图函数 ***
    plot_heatmap(alpha_range_deg, v_range, results_grid)


def draw():
    x_start = 0
    x_end = 600

    # 设置生成点的数量（这个值越大，函数图像就越平滑）
    num_points = 4000

    # --- 3. 生成数据和绘图 (通常无需修改这部分) ---

    # a. 在指定的范围内，均匀地生成x坐标点
    x_values = np.linspace(x_start, x_end, num_points)

    # b. 分别计算两个函数对应的y值
    y1_values = function1(x_values, 100, 0)
    y2_values = function2(x_values, 100)
    y3_values = function3(x_values, 0)

    x_values = [-x for x in x_values]
    y1_values = [-y for y in y1_values]
    y2_values = [-y for y in y2_values]
    y3_values = [-y for y in y3_values]

    # c. 开始绘图
    print("正在生成函数图像...")

    # 设置一个美观的绘图风格
    plt.style.use('seaborn-v0_8-whitegrid')

    # 创建一个图形窗口(画布)和坐标轴(幕布)，并指定大小
    fig, ax = plt.subplots(figsize=(12, 8))

    # 绘制x=0和y=0的参考线，使其更具数学感
    ax.axhline(0, color='black', linewidth=0.7, zorder=0)
    ax.axvline(0, color='black', linewidth=0.7, zorder=0)

    # d. 在同一个坐标轴(ax)上绘制两个函数图像
    # 为每个函数选择不同的、专业的颜色和标签
    ax.plot(x_values, y1_values, label='L1', color='#3498db', linewidth=2.5)
    ax.plot(x_values, y2_values, label='L2', color='#e74c3c', linewidth=2.5)
    ax.plot(x_values, y3_values, label='L3', color='#2ecc71', linewidth=2.5)

    # e. 添加美化和说明元素
    # 添加标题和坐标轴标签
    ax.set_title('profile map', fontsize=18, fontweight='bold', pad=15)
    ax.set_xlabel('x ', fontsize=14, labelpad=10)
    ax.set_ylabel('y ', fontsize=14, labelpad=10)

    # 动态调整y轴范围以适应两个函数，并增加10%的边距
    y_min = min(np.min(y1_values), np.min(y2_values))
    y_max = max(np.max(y1_values), np.max(y2_values))
    y_range = y_max - y_min
    ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

    # 显示图例，这是区分两条曲线的关键
    ax.legend(fontsize=12, frameon=True, facecolor='white', framealpha=0.9, shadow=True)

    # 显示网格
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # 自动调整布局并显示图形
    plt.tight_layout()
    plt.show()

    print("图像已生成并显示。")


if __name__ == "__main__":
    search()
    # draw()
