import glob

import numpy as np
import matplotlib.pyplot as plt
import math
import time
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import numba
import os
import argparse  # <-- 1. 导入用于处理命令行参数的库


# --- (您的所有辅助函数和全局常量定义保持不变) ---
# ... (此处省略 point_segment_distance, generate_cylinder_points, 等函数的代码)
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


@numba.jit(nopython=True, fastmath=True)
def calculate_ans_for_phase3(pos_g_start, T_start, P3_INTERVAL, V_G_Z_p3, P_M_starts, DIR_M_array, V_M1,
                             cylinder_surface_points):
    ans = 0.0
    t_phase3_start = T_start
    t_phase3_end = t_phase3_start + 20.0
    t_range = np.arange(t_phase3_start, t_phase3_end + P3_INTERVAL / 2, P3_INTERVAL)
    for t in t_range:
        t_since_p3_start = t - t_phase3_start
        pos_g_current = pos_g_start.copy()
        pos_g_current[2] -= V_G_Z_p3 * t_since_p3_start
        for i in range(P_M_starts.shape[0]):
            pos_m_current = P_M_starts[i] + DIR_M_array[i] * V_M1 * t
            all_ok = True
            for j in range(cylinder_surface_points.shape[0]):
                cyl_point = cylinder_surface_points[j]
                dist = point_segment_distance(pos_g_current, pos_m_current, cyl_point)
                if dist >= 10.0:
                    all_ok = False
                    break
            if all_ok:
                ans += P3_INTERVAL
    return ans


FIXED_V_FY2 = 140.0
MAX_TIME_HORIZON_S = 25
DIST_STEP = 1.0
Z_STEP = 1.0
P3_INTERVAL = 0.2
NUM_CYLINDER_POINTS = 100
V_M1 = 300.0
V_G_Z_p3 = -3.0  # Corrected to negative for sinking
A_G_Z = -9.8
TARGET_M = np.array([0.0, 0.0, 0.0])
CYLINDER_CENTER = np.array([0., 200., 0.])
P_M_starts = np.array([
    [20000.0, 0.0, 2000.0], [19000.0, 600.0, 2100.0], [18000.0, -600.0, 1900.0]
])
norms = np.linalg.norm(TARGET_M - P_M_starts, axis=1, keepdims=True)
DIR_M_array = (TARGET_M - P_M_starts) / norms
F1 = np.array([17800.0, 0.0, 1800.0])
F3 = np.array([6000.0, -3000.0, 700.0])
F2 = np.array([12000.0, 1400.0, 1400.0])
F4 = np.array([11000.0, 2000.0, 1800.0])
F5 = np.array([13000.0, -2000.0, 1300.0])
START_POINT = F1
cylinder_surface_points = generate_cylinder_points(
    CYLINDER_CENTER, 7.0, 10.0, NUM_CYLINDER_POINTS
)


def worker_function(params):
    i, j, z, d, current_dir_vector_xy = params
    T = d / FIXED_V_FY2
    delta_z = z - START_POINT[2]
    if delta_z > 0: return i, j, -1.0
    t2_squared = 2 * delta_z / A_G_Z
    if t2_squared < 0: return i, j, -1.0
    t2 = np.sqrt(t2_squared)
    if t2 > T: return i, j, -1.0
    pos_g_xy = START_POINT[:2] + current_dir_vector_xy * d
    pos_g_start = np.array([pos_g_xy[0], pos_g_xy[1], z])
    ans = calculate_ans_for_phase3(pos_g_start, T, P3_INTERVAL, V_G_Z_p3, P_M_starts, DIR_M_array, V_M1,
                                   cylinder_surface_points)
    return i, j, ans


# --- 新增的、独立的计算函数 ---
def run_computation(angle_deg, output_dir):
    print(f"\n{'=' * 25}\n--- 开始计算角度: {angle_deg}° ---\n{'=' * 25}")

    FIXED_ANGLE_RAD = math.radians(angle_deg)
    dir_vector_xy = np.array([math.cos(FIXED_ANGLE_RAD), math.sin(FIXED_ANGLE_RAD)])

    max_dist = FIXED_V_FY2 * MAX_TIME_HORIZON_S
    dist_axis = np.arange(0, max_dist + DIST_STEP / 2, DIST_STEP)
    z_axis = np.arange(0, START_POINT[2] + Z_STEP / 2, Z_STEP)
    ans_grid = np.full((len(z_axis), len(dist_axis)), -1.0)

    tasks = [(i, j, z, d, dir_vector_xy) for i, z in enumerate(z_axis) for j, d in enumerate(dist_axis)]

    print(f"--- 角度 {angle_deg}°: 开始并行计算 ---")
    start_time = time.time()
    num_processes = cpu_count()
    print(f"将使用 {num_processes} 个CPU核心进行计算...")

    with Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap_unordered(worker_function, tasks), total=len(tasks), desc=f"Angle {angle_deg}°"))

    for i, j, ans in results:
        ans_grid[i, j] = ans

    end_time = time.time()
    print(f"--- 角度 {angle_deg}°: 并行计算完成，耗时: {end_time - start_time:.2f} 秒 ---")

    # --- 核心修改：保存结果到文件，而不是绘图 ---
    filename = f"results_angle_{angle_deg:.1f}.npz"
    filepath = os.path.join(output_dir, filename)

    np.savez(filepath,
             ans_grid=ans_grid,
             dist_axis=dist_axis,
             z_axis=z_axis,
             angle_deg=angle_deg)

    print(f"计算结果已保存至: {filepath}")


# --- 新增的、独立的绘图函数 ---
def run_plotting(input_filepath):
    print(f"\n--- 正在从 {input_filepath} 加载数据并绘图 ---")

    try:
        data = np.load(input_filepath)
        ans_grid = data['ans_grid']
        dist_axis = data['dist_axis']
        z_axis = data['z_axis']
        angle_deg = data['angle_deg']
    except Exception as e:
        print(f"错误：无法加载文件 {input_filepath}。错误信息: {e}")
        return

    output_dir = os.path.dirname(input_filepath)  # Get directory from filepath

    ans_grid_masked = np.ma.masked_where(ans_grid < 0, ans_grid)
    fig, ax = plt.subplots(figsize=(12, 9))
    im = ax.imshow(ans_grid_masked, origin='lower', aspect='auto',
                   extent=[dist_axis.min(), dist_axis.max(), z_axis.min(), z_axis.max()],
                   cmap='viridis')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Sum of ANS for M1,M2,M3 (s)')
    ax.set_xlabel(f"沿方向前进的距离 (对应T范围 0s ~ {MAX_TIME_HORIZON_S:.1f}s)")
    ax.set_ylabel("G点第三阶段初始Z坐标")
    ax.set_title(f"ANS(M1+M2+M3)分布热力图\n(V_FY2={FIXED_V_FY2:.1f}, Angle={angle_deg:.2f}°)")

    filename = f"heatmap_angle_{angle_deg:.1f}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"图片已保存至: {filepath}")


# --- 主程序 ---
if __name__ == "__main__":

    # --- 2. 使用 argparse 定义命令行接口 ---
    # parser = argparse.ArgumentParser(description="运行计算或绘图任务。")
    # parser.add_argument('--compute', nargs='+', type=float, metavar='ANGLE',
    #                     help='运行一个或多个角度的计算任务。例如: --compute 90 110 130')
    # parser.add_argument('--plot', nargs='+', type=str, metavar='FILE',
    #                     help='为一个或多个 .npz 结果文件绘图。例如: --plot results_angle_90.0.npz')
    #
    # args = parser.parse_args()
    #
    # output_dir = "heatmaps_output"
    # os.makedirs(output_dir, exist_ok=True)
    #
    # # --- 3. 根据命令行参数执行相应操作 ---
    # if args.compute:
    #     print(f"已接收到计算任务，角度为: {args.compute}")
    #     print(f"计算结果将保存到 '{output_dir}' 文件夹中。")
    #     for angle in args.compute:
    #         run_computation(angle, output_dir)
    #     print("\n--- 所有计算任务处理完毕！ ---")
    #
    # elif args.plot:
    #     print(f"已接收到绘图任务，文件为: {args.plot}")
    #     for filepath in args.plot:
    #         run_plotting(filepath)
    #     print("\n--- 所有绘图任务处理完毕！ ---")
    #
    # else:
    #     print("错误：请提供操作参数。")
    #     print("用法示例:")
    #     print("  1. 运行计算: python your_script_name.py --compute 90 110 130")
    #     print("  2. 运行绘图: python your_script_name.py --plot heatmaps_output/results_angle_90.0.npz")
    #     print("  3. 绘图 (通配符): python your_script_name.py --plot heatmaps_output/*.npz  (在某些shell中可用)")
    #
    #     # --- 第3步：执行所有绘图任务 ---
    #     # 查找输出文件夹中所有已生成的.npz数据文件，并为它们绘图
    # print(f"\n--- 开始在 '{output_dir}' 文件夹中查找数据文件并生成图像 ---")

    # 使用glob找到所有符合命名规则的.npz文件


    #angle_list = np.linspace(15,165,20)
    angle_list = [ 175, 185,0, 180, 5, 360, ]
    output_dir = "heatmaps_outputF1"
    os.makedirs(output_dir, exist_ok=True)

    # --- 3. 根据命令行参数执行相应操作 ---

    print(f"计算结果将保存到 '{output_dir}' 文件夹中。")
    for angle in angle_list:
        run_computation(angle, output_dir)
    print("\n--- 所有计算任务处理完毕！ ---")

    search_pattern = os.path.join(output_dir, "results_angle_*.npz")
    data_files = glob.glob(search_pattern)

    if not data_files:
        print("未找到任何数据文件来进行绘图。")
    else:
        # 遍历文件列表，为每个文件调用绘图函数
        for data_filepath in tqdm(data_files, desc="正在生成热力图"):
            run_plotting(data_filepath)
        print("\n--- 所有绘图任务处理完毕！ ---")