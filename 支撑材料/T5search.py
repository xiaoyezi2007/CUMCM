import numpy as np
import math
import matplotlib.pyplot as plt
from tqdm import tqdm


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

def func(x, v, alpha, xm, ym, zm, xf, yf, zf):
    
    try:
        cos_alpha = math.cos(alpha)
        sin_alpha = math.sin(alpha)
        if cos_alpha == 0: return None

        t = xf - x * cos_alpha
        vm = 300.0

        denominator = ym * t / xm - (sin_alpha * (xf - t) / cos_alpha + yf - 200)
        if denominator == 0: return None

        k = 200.0 / denominator
        xm1 = k * t
        ym1 = 200.0 + k * (yf - 200.0 + sin_alpha * (xf - t) / cos_alpha)
        zm1 = k * zm * t / xm

        dis_squared = (xm - xm1) ** 2 + (ym - ym1) ** 2 + (zm - zm1) ** 2
        if dis_squared < 0: return None
        dis = math.sqrt(dis_squared)

        result = 3 * x / v + zf - t * zm / xm - 3 * dis / vm
        return result
    except (ValueError, ZeroDivisionError):
        return None

def calculate_ans_for_phase3(t1=0.00714286, t2=0, t3=20,temp_alpha=1,V_FY1=1,P_M1_start = [],P_FY1_start=[]):
   
    ANS = 0
    valid = []
   
    INTERVAL = 0.01
    V_M1 = 300.0
    A_G_Z = -9.8

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
    #visualize_cylinder(cylinder_surface_points)
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
    return ANS


# --- 全局常量和参数 ---
MODE = 0
V_INTERVAL = 10 if MODE == 0 else 5
ANGLE_INTERVAL = 10 if MODE == 0 else 0.5
X_INTERVAL = 50 if MODE == 0 else 20
NUM_CYLINDER_POINTS = 100
V_M1 = 300.0
A_G_Z = -9.8
P_M1_start = np.array([20000.0, 0.0, 2000.0])
P_M2_start = np.array([19000.0, 600, 2100])
P_M3_start = np.array([18000, -600, 1900])
P_FY1_start = np.array([17800.0, 0.0, 1800.0])
P_FY2_start = np.array([12000,1400,1400])
P_FY3_start = np.array([6000,-3000,700])
P_FY4_start = np.array([11000,2000,1800])
P_FY5_start = np.array([13000,-2000,1300])
TARGET_M1 = np.array([0.0, 0.0, 0.0])
CYLINDER_CENTER = np.array([0.0, 200.0, 0.0])
DIR_M1 = (TARGET_M1 - P_M1_start) / np.linalg.norm(TARGET_M1 - P_M1_start)
cylinder_surface_points = generate_cylinder_points(
    CYLINDER_CENTER, 7.0, 10.0, NUM_CYLINDER_POINTS
)

global_max_ans = -1
global_best_params = {}


def plot_heatmap(alpha_range, v_range, results_grid):
    print("\n" + "*" * 30)
    print("正在生成最终结果的热力图...")
    plt.figure(figsize=(12, 9))
    im = plt.imshow(results_grid.T, origin='lower', aspect='auto',
                    extent=[alpha_range.min(), alpha_range.max(), v_range.min(), v_range.max()],
                    cmap='viridis')
    cbar = plt.colorbar()
    cbar.set_label('Max ANS (s)')
    plt.title('Max ANS Heatmap for Angle vs. Speed', fontsize=16)
    plt.xlabel('Angle (degrees)', fontsize=12)
    plt.ylabel('Speed (V_FY1)', fontsize=12)
    plt.show()


def search():
    global global_max_ans, global_best_params
    FY = P_FY5_start

    # --- 定义搜索范围 ---
    v_min, v_max = 70, 140
    alpha_min_deg, alpha_max_deg = 20, 150
    x_min, x_max = 1000, 3500

    alpha_range_deg = np.arange(alpha_min_deg, alpha_max_deg + ANGLE_INTERVAL, ANGLE_INTERVAL)
    v_range = np.arange(v_min, v_max + V_INTERVAL, V_INTERVAL)
    x_range = np.arange(x_min, x_max + X_INTERVAL, X_INTERVAL)

    results_grid = np.full((len(alpha_range_deg), len(v_range)), -1.0)

    for i, alpha in enumerate(tqdm(alpha_range_deg, desc="Angle Progress")):
        alpha_rad = np.radians(alpha)
        for j, V in enumerate(v_range):
            if V == 0: continue

            best_x_for_pair1 = -1
            best_x_for_pair2 = -1
            best_x_for_pair3 = -1
            max_ans_for_pair1 = -1
            max_ans_for_pair2 = -1
            max_ans_for_pair3 = -1

            for x in x_range:
             
                y1 = func(x, V, alpha_rad, P_M1_start[0], P_M1_start[1], P_M1_start[2], FY[0], FY[1], FY[2])
                y2 = func(x, V, alpha_rad, P_M2_start[0], P_M2_start[1], P_M2_start[2], FY[0], FY[1], FY[2])
                y3 = func(x, V, alpha_rad, P_M3_start[0], P_M3_start[1], P_M3_start[2], FY[0], FY[1], FY[2])
                T = x/V
                y = 9.8*T*T/2
                y1,y2,y3 = min(y1,y),min(y2,y),min(y3,y)
             
                if (y1 is None or y1 <= 0) and (y2 is None or y2 <= 0) and (y3 is None or y3 <= 0):
                    continue

              
                if y1 >= FY[2] and y2 >= FY[2] and y3 >= FY[2]:
                    continue

                if y1 < 0 or y1 is None:
                    t21 = 0
                else:
                    t21 = math.sqrt(2 * y1 / abs(A_G_Z))
                # 释放前飞行时间 t1
                t11 = T - t21
                if y2 < 0 or y2 is None:
                    t22 = 0
                else:
                    t22 = math.sqrt(2 * y2 / abs(A_G_Z))
                # 释放前飞行时间 t1
                t12 = T - t22
                if y3 < 0 or y3 is None:
                    t23 = 0
                else:
                    t23 = math.sqrt(2 * y3 / abs(A_G_Z))
                
                t13 = T - t23

                # 如果t1为负，说明无人机到达该位置前，干扰弹就已落地，物理上不可能
                if t11 < 0 and t12 < 0 and t13 < 0:
                    continue

                # --- 4. 运行模拟并更新最优解 ---
                current_ans1 = calculate_ans_for_phase3(t11, t21, 20, alpha, V, P_M1_start, FY)
                current_ans2 = calculate_ans_for_phase3(t12, t22, 20, alpha, V, P_M2_start, FY)
                current_ans3 = calculate_ans_for_phase3(t13, t23, 20, alpha, V, P_M3_start, FY)
                #print(t1, t2, alpha, V,current_ans)
                if current_ans1 > max_ans_for_pair1:
                    max_ans_for_pair1 = current_ans1
                    best_x_for_pair1 = x
                if current_ans2 > max_ans_for_pair2:
                    max_ans_for_pair2 = current_ans2
                    best_x_for_pair2 = x
                if current_ans3 > max_ans_for_pair3:
                    max_ans_for_pair3 = current_ans3
                    best_x_for_pair3 = x

            if max_ans_for_pair1 > -1 and max_ans_for_pair2 > -1 and max_ans_for_pair3 > -1:
                results_grid[i, j] = max_ans_for_pair1 + max_ans_for_pair2 + max_ans_for_pair3
                print(
                    f"Angle={alpha:.2f}°, V={V:.1f} | Found Best ANS1={max_ans_for_pair1:.3f} ANS2={max_ans_for_pair2:.3f} ANS3={max_ans_for_pair3:.3f} at x1={best_x_for_pair1:.1f}m x2={best_x_for_pair2:.1f}m x3={best_x_for_pair3:.1f}m")

                if max_ans_for_pair1 + max_ans_for_pair2 + max_ans_for_pair3 > global_max_ans:
                    global_max_ans = max_ans_for_pair1 + max_ans_for_pair2 + max_ans_for_pair3
                    global_best_params = {
                        'alpha_deg': alpha,
                        'V_FY1': V,
                        'x1': best_x_for_pair1,
                        'x2': best_x_for_pair2,
                        'x3': best_x_for_pair3,
                        'ANS': max_ans_for_pair1 + max_ans_for_pair2 + max_ans_for_pair3
                    }

    print("\n" + "=" * 50)
    print("SEARCH COMPLETED!")
    if global_max_ans > -1:
        print(f"Global Best Result:")
        print(f"  - Max ANS: {global_best_params['ANS']:.4f} s")
        print(f"  - Angle: {global_best_params['alpha_deg']:.2f} degrees")
        print(f"  - Speed: {global_best_params['V_FY1']:.1f} m/s")
        print(f"  - Horizontal Distance (x1): {global_best_params['x1']:.1f} m")
        print(f"  - Horizontal Distance (x2): {global_best_params['x2']:.1f} m")
        print(f"  - Horizontal Distance (x3): {global_best_params['x3']:.1f} m")
    else:
        print("No valid solution found in the given parameter range.")
    print("=" * 50)

    plot_heatmap(alpha_range_deg, v_range, results_grid)


if __name__ == "__main__":
    search()