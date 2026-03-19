import numpy as np
import matplotlib.pyplot as plt
import math
import time
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import numba


# --- 辅助函数定义 ---

@numba.jit(nopython=True, fastmath=True)
def point_segment_distance(p, a, b):
    ab = b - a
    dot_product = np.dot(ab, ab)
    if dot_product == 0: return np.linalg.norm(p - a)
    ap = p - a
    t = np.dot(ap, ab) / dot_product
    if t < 0.0:
        return np.linalg.norm(p - a)
    elif t > 1.0:
        return np.linalg.norm(p - b)
    else:
        projection = a + t * ab
        return np.linalg.norm(p - projection)


# generate_cylinder_points 只在开始时运行一次，无需JIT
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


# --- 【核心】: 恢复您未经修改的原始函数，以保证逻辑100%正确 ---
# 这个函数不再被JIT编译，每个子进程将调用这个原始版本。
# 它内部的硬编码变量（如V_FY1=140.0, INTERVAL=0.01）是原始逻辑的一部分，必须保留。
def calculate_ans_for_phase3(t1=0.00714286, t2=0, t3=20, temp_alpha=0, V_FY1=140):
    valid = []
    ANS = 0
    INTERVAL = 0.005 if mode == 1 else 0.01  # 使用您原版的INTERVAL

    temp_alpha_rad = np.radians(temp_alpha)
    DIR_FY1 = np.array([-math.cos(temp_alpha_rad), math.sin(temp_alpha_rad), 0.0])

    t_phase1_end = t1
    pos_fy1_p1 = P_FY1_start + DIR_FY1 * V_FY1 * t_phase1_end

    t_phase2_end = t1 + t2
    t_phase2_duration = t2
    vel_g_launch = DIR_FY1 * V_FY1
    pos_g_p2_xy = pos_fy1_p1[:2] + vel_g_launch[:2] * t_phase2_duration
    pos_g_p2_z = pos_fy1_p1[2] + 0.5 * A_G_Z * (t_phase2_duration ** 2)
    pos_g_p2 = np.array([pos_g_p2_xy[0], pos_g_p2_xy[1], pos_g_p2_z])

    t_phase3_start = t_phase2_end
    t_phase3_end = t_phase3_start + t3
    V_SMOKE_Z = -3.0
    pos_smoke_start = pos_g_p2

    for t in np.arange(t_phase3_start, t_phase3_end, INTERVAL):
        pos_m1_current = P_M1_start + DIR_M1 * V_M1 * t
        t_since_detonation = t - t_phase3_start
        pos_smoke_current = pos_smoke_start + np.array([0, 0, V_SMOKE_Z * t_since_detonation])

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


# --- 全局常量定义 ---
mode = 0
V_INTERVAL = 2 if mode == 0 else 1
V_M1 = 300.0
NUM_CYLINDER_POINTS = 500 if mode == 0 else 1000

A_G_Z = -9.8
P_M1_start = np.array([20000.0, 0.0, 2000.0])
P_FY1_start = np.array([17800.0, 0.0, 1800.0])
TARGET_M1 = np.array([0.0, 0.0, 0.0])
CYLINDER_CENTER = np.array([0.0, 200.0, 0.0])
DIR_M1 = (TARGET_M1 - P_M1_start) / np.linalg.norm(TARGET_M1 - P_M1_start)
cylinder_surface_points = generate_cylinder_points(
    CYLINDER_CENTER, 7.0, 10.0, NUM_CYLINDER_POINTS
)
v_min = 70
v_max = 140


# --- JIT 编译的辅助函数 (这些是安全的，已将全局变量替换为字面量以确保JIT安全) ---
@numba.jit(nopython=True, fastmath=True)
def function1(x, V, alpha_rad):
    return (3 * x) / V + 20 + (math.cos(alpha_rad) / 17800.0) * (1800.0 - 20) * x + \
        (3 / (math.cos(math.atan2(2000.0, 20000.0)) * 300.0)) * \
        (0.0 + (200.0 / (200.0 - x * math.sin(alpha_rad))) * (
                17800.0 - x * math.cos(alpha_rad) - 0.0) - 20000.0)


@numba.jit(nopython=True, fastmath=True)
def function2(x, V):
    return (9.8 / (2 * V * V)) * x * x


@numba.jit(nopython=True, fastmath=True)
def function3(x, alpha_rad):
    return (math.cos(alpha_rad) * x * (1800.0 - 20)) / 17800.0 + 20


@numba.jit(nopython=True, fastmath=True)
def check(a, b, tmp, v, alpha_rad, P_FY1_start_local, P_M1_start_local, DIR_M1_local, V_M1_local,
          cylinder_surface_points_local):
    T = a / v
    t2 = math.sqrt(2 * b / 9.8) if (2 * b / 9.8) >= 0 else -1
    if t2 < 0: return False
    t1 = T - t2

    pos_g_current = np.empty(3, dtype=np.float64)
    if tmp < t1:
        pos_g_current[0] = P_FY1_start_local[0] - v * tmp * math.cos(alpha_rad)
        pos_g_current[1] = P_FY1_start_local[1] + v * tmp * math.sin(alpha_rad)
        pos_g_current[2] = P_FY1_start_local[2]
    elif tmp < T:
        pos_at_t1 = P_FY1_start_local + np.array([-math.cos(alpha_rad), math.sin(alpha_rad), 0.0]) * v * t1
        vel_at_t1 = np.array([-math.cos(alpha_rad), math.sin(alpha_rad), 0.0]) * v
        dt = tmp - t1
        pos_g_current = pos_at_t1 + vel_at_t1 * dt + np.array([0.0, 0.0, 0.5 * -9.8 * dt ** 2])
    else:
        pos_at_t1 = P_FY1_start_local + np.array([-math.cos(alpha_rad), math.sin(alpha_rad), 0.0]) * v * t1
        vel_at_t1 = np.array([-math.cos(alpha_rad), math.sin(alpha_rad), 0.0]) * v
        pos_at_T = pos_at_t1 + vel_at_t1 * t2 + np.array([0.0, 0.0, 0.5 * -9.8 * t2 ** 2])
        dt = tmp - T
        pos_g_current = pos_at_T + np.array([0.0, 0.0, -3.0 * dt])

    pos_m1_current = P_M1_start_local + DIR_M1_local * V_M1_local * tmp
    for j in range(cylinder_surface_points_local.shape[0]):
        dist = point_segment_distance(pos_g_current, pos_m1_current, cylinder_surface_points_local[j])
        if dist >= 10.0: return False
    return True


# --- 绘图函数 (不变) ---
def plot_heatmap(alpha_range, v_range, results_grid):
    print("\n" + "*" * 30 + "\n正在生成最终结果的热力图...")
    plt.figure(figsize=(12, 10))
    im = plt.imshow(results_grid.T, origin='lower', aspect='auto',
                    extent=[alpha_range.min(), alpha_range.max(), v_range.min(), v_range.max()],
                    cmap='viridis')
    cbar = plt.colorbar(im)
    cbar.set_label('Max ANS (ans1+ans2+ans3)', fontsize=12)
    plt.title('Max ANS Heatmap for Angle vs. Speed', fontsize=16)
    plt.xlabel('Angle (degrees)', fontsize=12)
    plt.ylabel('Speed (V_FY1)', fontsize=12)
    plt.show()


# --- 并行的工作函数 (Worker) ---
def worker_function(params):
    alpha_deg, V, i, j = params
    alpha_rad = np.radians(alpha_deg)

    mx = -1.0
    best_params_local = []

    for x1 in np.arange(0.0, 100, 1):
        y1_val = function1(x1, V, alpha_rad)
        y1_val = max(y1_val, 0)
        y2_val = function2(x1, V)
        if y1_val > y2_val: break

        y = min(y1_val, y2_val)

        if V == 0:
            t_val = 0
        else:
            t_val = x1 / V

        t1_val = t_val - math.sqrt(2 * y / 9.8) if (2 * y / 9.8) >= 0 else -1
        if t1_val < 0: continue

        ans1, tmp = calculate_ans_for_phase3(t1_val, t_val - t1_val, 20, alpha_deg, V)
        if tmp < 0: continue

        x2_l, x2_r = x1, 1500
        while x2_r - x2_l > 0.1:
            mid = (x2_l + x2_r) / 2
            y_mid = function1(mid, V, alpha_rad)
            if y_mid < 0:
                x2_r = mid
                continue
            t_mid_val = mid / V
            t1_mid_val = t_mid_val - math.sqrt(2 * y_mid / 9.8) if (2 * y_mid / 9.8) >= 0 else -1
            if t1_mid_val < 0:
                x2_r = mid
                continue
            if t1_mid_val < t1_val + 1 or tmp - (function3(mid, alpha_rad) - y_mid) / 3 > t_mid_val or check(mid, y_mid,
                                                                                                             tmp, V,
                                                                                                             alpha_rad,
                                                                                                             P_FY1_start,
                                                                                                             P_M1_start,
                                                                                                             DIR_M1,
                                                                                                             V_M1,
                                                                                                             cylinder_surface_points):
                x2_l = mid
            else:
                x2_r = mid
        x2, y2 = x2_r + 5 , function1(x2_r + 5, V, alpha_rad)
        if y2 < 0: y2 = 0

        t_val2 = x2 / V
        t2_val = t_val2 - math.sqrt(2 * y2 / 9.8) if (2 * y2 / 9.8) >= 0 else -1
        if t2_val < 0: continue

        ans2, tmp2 = calculate_ans_for_phase3(t2_val, t_val2 - t2_val, 20, alpha_deg, V)
        if tmp2 < 0: tmp2 = tmp

        x3_l, x3_r = x2, 2000
        while x3_r - x3_l > 0.1:
            mid = (x3_l + x3_r) / 2
            y_mid = function1(mid, V, alpha_rad)
            if y_mid < 0:
                x3_r = mid
                continue
            t_mid_val = mid / V
            t1_mid_val = t_mid_val - math.sqrt(2 * y_mid / 9.8) if (2 * y_mid / 9.8) >= 0 else -1
            if t1_mid_val < 0:
                x3_r = mid
                continue
            if t1_mid_val < t2_val + 1 or tmp2 - (function3(mid, alpha_rad) - y_mid) / 3 > t_mid_val or check(mid,
                                                                                                              y_mid,
                                                                                                              tmp2, V,
                                                                                                              alpha_rad,
                                                                                                              P_FY1_start,
                                                                                                              P_M1_start,
                                                                                                              DIR_M1,
                                                                                                              V_M1,
                                                                                                              cylinder_surface_points):
                x3_l = mid
            else:
                x3_r = mid
        x3, y3 = x3_r + 5, function1(x3_r + 5, V, alpha_rad)
        if y3 < 0: y3 = 0

        t_val3 = x3 / V
        t3_val = t_val3 - math.sqrt(2 * y3 / 9.8) if (2 * y3 / 9.8) >= 0 else -1
        if t3_val < 0: continue

        ans3, _ = calculate_ans_for_phase3(t3_val, t_val3 - t3_val, 20, alpha_deg, V)
        if ans3 < 0: ans3 = 0.0

        current_ans = ans1 + ans2 + ans3
        if current_ans > mx:
            mx = current_ans
            best_params_local = [[x1, y, x2, y2, x3, y3, ans1, ans2, ans3]]

    return (mx, best_params_local, alpha_deg, V, i, j)


# --- 主程序 ---
if __name__ == "__main__":
    ANGLE_INTERVAL = 0.1 if mode == 0 else 0.005

    alpha_range_deg = np.arange(0.0, 0.8, ANGLE_INTERVAL)
    v_range = np.arange(v_min, v_max + 1, V_INTERVAL)
    results_grid = np.full((len(alpha_range_deg), len(v_range)), -1.0)

    tasks = []
    for i, alpha in enumerate(alpha_range_deg):
        for j, V in enumerate(v_range):
            tasks.append((alpha, V, i, j))

    print("--- 开始并行计算 (使用CPU + Numba JIT) ---")
    start_time = time.time()
    num_processes = cpu_count()
    print(f"将使用 {num_processes} 个CPU核心进行计算...")

    with Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap_unordered(worker_function, tasks), total=len(tasks), desc="Total Progress"))

    global_max_ans = -1
    global_best_params = []
    global_best_alpha = -1
    global_best_V = -1

    for mx, best_local, alpha, V, i, j in results:
        if mx > -1:
            results_grid[i, j] = mx
        if mx > global_max_ans:
            global_max_ans = mx
            global_best_params = best_local
            global_best_alpha = alpha
            global_best_V = V

    end_time = time.time()
    print(f"--- 并行计算完成，总耗时: {end_time - start_time:.2f} 秒 ---")

    print(f"\nGLOBAL BEST: Angle={global_best_alpha:.3f}°, V_FY1={global_best_V:.1f}, ANS={global_max_ans:.3f}")
    print(f"PARAMS: {global_best_params}")

    plot_heatmap(alpha_range_deg, v_range, results_grid)