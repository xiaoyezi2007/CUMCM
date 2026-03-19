import numpy as np
from shapely.geometry import Polygon, LineString, Point
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys 


class Logger(object):
    def __init__(self, filename="record.txt"):
        self.terminal = sys.stdout
       
        self.log = open(filename, "w", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
       
        self.terminal.flush()
        self.log.flush()

# 将标准输出重定向到我们自定义的Logger实例
sys.stdout = Logger("record.txt")

# --- 辅助函数定义 ---
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


def calculate_ans_for_phase3(pos_g_start, T_start):
    ans = 0.0
    t_phase3_start = T_start
    t_phase3_end = t_phase3_start + 20.0
    for t in np.arange(t_phase3_start, t_phase3_end + P3_INTERVAL / 2, P3_INTERVAL):
        t_since_p3_start = t - t_phase3_start
        pos_m1_current = P_M1_start + DIR_M1 * V_M1 * t
        pos_g_current = np.copy(pos_g_start)
        pos_g_current[2] -= V_G_Z_p3 * t_since_p3_start
        all_ok = True
        for cyl_point in cylinder_surface_points:
            dist = point_segment_distance(pos_g_current, pos_m1_current, cyl_point)
            if dist > 10.0:
                all_ok = False
                break
        if all_ok:
            ans += P3_INTERVAL
    return ans


V_INTERVAL = 5
T_INTERVAL = 1  # T的步长可以适当增大，因为优化算法会帮助我们找到Z
ANGLE_STEP_DEG = 1  # 角度步长
Z_STEP = 2  # Z坐标步长
P3_INTERVAL = 0.01
NUM_CYLINDER_POINTS = 100

# --- 全局常量 ---
V_M1 = 300.0
V_G_Z_p3 = 3.0
A_G_Z = -9.8
P_M1_start = np.array([20000.0, 0.0, 2000.0])
P_FY1_start = np.array([12000.0, 1400.0, 1400.0])
TARGET_M1 = np.array([0.0, 0.0, 0.0])
CYLINDER_CENTER = np.array([0.0, 200.0, 0.0])
DIR_M1 = (TARGET_M1 - P_M1_start) / np.linalg.norm(TARGET_M1 - P_M1_start)
cylinder_surface_points = generate_cylinder_points(
    CYLINDER_CENTER, 7.0, 10.0, NUM_CYLINDER_POINTS
)

# --- 主搜索循环 ---
global_max_ans = -1
global_best_params = {}

v_min = 70
v_max = 140

for V_FY1 in np.arange(v_min, v_max + V_INTERVAL, V_INTERVAL):
    print(f"\n{'=' * 20} Testing V_FY1 = {V_FY1:.1f} {'=' * 20}")
    T_min = 5
    T_max = 25
    # 用于记录上一个T的最优z
    z_optimal_for_previous_T = None

    for T in np.arange(T_min, T_max, T_INTERVAL):
        pos_m1_T = P_M1_start + DIR_M1 * V_M1 * T
        pos_m1_T_plus_20 = P_M1_start + DIR_M1 * V_M1 * (T + 20.0)
        cir_center_xy = P_FY1_start[:2]
        cir_radius = V_FY1 * T

        v1 = pos_m1_T[:2]
        v2 = pos_m1_T_plus_20[:2]
        v3 = CYLINDER_CENTER[:2]
        dist_to_edge1 = point_segment_distance(cir_center_xy, v1, v2)
        dist_to_edge2 = point_segment_distance(cir_center_xy, v2, v3)
        dist_to_edge3 = point_segment_distance(cir_center_xy, v3, v1)
        min_dist_to_triangle = min(dist_to_edge1, dist_to_edge2, dist_to_edge3)
        if min_dist_to_triangle > cir_radius:
            continue

        # --- 计算几何相交区域 ---
        triangle = Polygon([v1, v2, v3])
        circle_boundary = LineString(
            [(cir_center_xy[0] + cir_radius * np.cos(a), cir_center_xy[1] + cir_radius * np.sin(a)) for a in
             np.linspace(0, 2 * np.pi, 360)])
        intersection = triangle.intersection(circle_boundary)
        if intersection.is_empty:
            continue

        # --- 确定G点Z坐标范围 ---
        z_max = P_FY1_start[2]
        z_min = P_FY1_start[2] + 0.5 * A_G_Z * (T ** 2)

        # --- 5. 遍历候选点，寻找最大ANS ---
        max_ans_for_this_T = -1
        best_g_for_this_T = None

        # 优化1：自适应调整Z的搜索起点
        if z_optimal_for_previous_T is not None:
            z_start_guess = z_optimal_for_previous_T + 50.0
            z_start = max(z_min, min(z_start_guess, z_max))
        else:
            z_start = z_max

        candidate_arcs = [intersection] if not hasattr(intersection, 'geoms') else intersection.geoms
        for arc in candidate_arcs:
            start_angle = math.atan2(arc.coords[0][1] - cir_center_xy[1], arc.coords[0][0] - cir_center_xy[0])
            end_angle = math.atan2(arc.coords[-1][1] - cir_center_xy[1], arc.coords[-1][0] - cir_center_xy[0])
            if end_angle < start_angle: end_angle += 2 * np.pi
            num_angle_steps = int(np.rad2deg(abs(end_angle - start_angle)) / ANGLE_STEP_DEG) + 1

            for angle in np.linspace(start_angle, end_angle, num_angle_steps):
                g_xy = np.array(
                    [cir_center_xy[0] + cir_radius * np.cos(angle), cir_center_xy[1] + cir_radius * np.sin(angle)])

                # 优化2：为提前退出做准备
                prev_ans_1 = -1.0
                prev_ans_2 = -1.0
                best_z_for_this_angle = None
                max_ans_for_this_angle = -1

                for z in np.arange(z_min, z_start, Z_STEP)[::-1]:
                    pos_g_start = np.array([g_xy[0], g_xy[1], z])
                    ans = calculate_ans_for_phase3(pos_g_start, T)

                    if ans < prev_ans_1 < prev_ans_2:
                        break  # 提前退出
                    prev_ans_2 = prev_ans_1
                    prev_ans_1 = ans

                    if ans > max_ans_for_this_angle:
                        max_ans_for_this_angle = ans
                        best_z_for_this_angle = z

                    if ans > max_ans_for_this_T:
                        max_ans_for_this_T = ans
                        best_g_for_this_T = pos_g_start
                        if ans > global_max_ans:
                            global_max_ans = ans
                            global_best_params = {'V_FY1': V_FY1, 'T': T, 'P_G': pos_g_start}

                        # 在T循环结束时，更新用于下一个T的z_optimal

        z_optimal_for_previous_T = best_g_for_this_T[2]
        if best_g_for_this_T is not None:
            print(
                f"V_FY1={V_FY1:.1f}, T={T:.1f}, Max ANS={max_ans_for_this_T:.3f}, G_start=({best_g_for_this_T[0]:.2f}, {best_g_for_this_T[1]:.2f}, {best_g_for_this_T[2]:.2f})")

# --- 全局最优解分析与可视化 ---
print("\n" + "*" * 30)
print("全局最优解分析:")
if global_max_ans > -1:
    V_FY1_opt, T_opt, P_G_opt = global_best_params['V_FY1'], global_best_params['T'], global_best_params['P_G']
    displacement_fy1_xy = P_G_opt[:2] - P_FY1_start[:2]
    norm = np.linalg.norm(displacement_fy1_xy)
    dir_fy1_opt = displacement_fy1_xy / norm if norm > 0 else np.array([0.0, 0.0])
    delta_z = P_G_opt[2] - P_FY1_start[2]
    t2_opt = np.sqrt(2 * delta_z / A_G_Z) if delta_z <= 0 else 0
    t1_opt = T_opt - t2_opt
    P_M1_p3_start_opt = P_M1_start + DIR_M1 * V_M1 * T_opt

    print(f"最大 ANS: {global_max_ans:.3f}")
    print("\n达成此结果的四个核心变量为:")
    print(f"  1. FY1 飞行速度: {V_FY1_opt:.2f} 单位/秒")
    print(f"  2. FY1 飞行方向向量 (XY): ({dir_fy1_opt[0]:.4f}, {dir_fy1_opt[1]:.4f})")
    print(f"  3. 第一阶段持续时间 (t1): {t1_opt:.3f} 秒")
    print(f"  4. 第二阶段持续时间 (t2): {t2_opt:.3f} 秒")

    print("\n正在生成最优解的关键点位置图...")
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(P_FY1_start[0], P_FY1_start[1], P_FY1_start[2], c='blue', s=100, label='FY1 Start', marker='s')
    ax.scatter(P_G_opt[0], P_G_opt[1], P_G_opt[2], c='green', s=150, label='G Start (P3)', marker='*')
    ax.scatter(P_M1_p3_start_opt[0], P_M1_p3_start_opt[1], P_M1_p3_start_opt[2], c='red', s=100, label='M1 Start (P3)',
               marker='o')
    ax.scatter(0, 0, 0, c='black', s=80, label='Origin', marker='+')
    ax.scatter(CYLINDER_CENTER[0], CYLINDER_CENTER[1], CYLINDER_CENTER[2], c='gray', s=80, label='Cylinder Center',
               marker='D')
    ax.text(P_FY1_start[0], P_FY1_start[1], P_FY1_start[2], '  FY1 Start', color='blue')
    ax.text(P_G_opt[0], P_G_opt[1], P_G_opt[2], '  G Start (P3)', color='green', fontsize=12)
    ax.text(P_M1_p3_start_opt[0], P_M1_p3_start_opt[1], P_M1_p3_start_opt[2], '  M1 Start (P3)', color='red')
    ax.text(0, 0, 0, '  Origin', color='black')
    ax.text(CYLINDER_CENTER[0], CYLINDER_CENTER[1], CYLINDER_CENTER[2], '  Cylinder Center', color='gray')
    ax.plot([P_FY1_start[0], P_G_opt[0]], [P_FY1_start[1], P_G_opt[1]], [P_FY1_start[2], P_FY1_start[2]], 'b--',
            alpha=0.5, label='FY1 Path (XY Projection)')
    ax.plot([P_G_opt[0]], [P_G_opt[1]], [P_FY1_start[2], P_G_opt[2]], 'g--', alpha=0.5, label='G Z-Drop')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')
    ax.set_title('Key Positions for Global Optimal Solution')
    ax.legend()
    ax.invert_yaxis()
    ax.view_init(elev=25, azim=-75)
    plt.show()
else:
    print("在搜索范围内未找到任何有效的（ANS > 0）解。")