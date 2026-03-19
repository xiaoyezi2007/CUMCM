import numpy as np
import matplotlib.pyplot as plt
import math

# --- 辅助函数定义 (无需改变) ---
def point_segment_distance(p, a, b):
    ab = b - a
    if np.dot(ab, ab) == 0: return np.linalg.norm(p - a)
    ap = p - a
    t = np.dot(ap, ab) / np.dot(ab, ab)
    if t < 0.0: return np.linalg.norm(p - a)
    elif t > 1.0: return np.linalg.norm(p - b)
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

# --- 第五问 核心修改: 重写calculate_ans_for_phase3 ---
def calculate_ans_for_phase3(pos_g_start, T_start):
    ans = 0.0
    t_phase3_start = T_start
    t_phase3_end = t_phase3_start + 20.0

    for t in np.arange(t_phase3_start, t_phase3_end + P3_INTERVAL/2, P3_INTERVAL):
        t_since_p3_start = t - t_phase3_start

        # 计算当前G点的位置
        pos_g_current = np.copy(pos_g_start)
        pos_g_current[2] -= V_G_Z_p3 * t_since_p3_start

        # 计算当前 M1, M2, M3 的位置
        # (3, 3) 数组，每行是一个M点的坐标
        pos_m_currents = P_M_starts + DIR_M_array * V_M1 * t

        # 分别对三个M点进行判定，并累加ANS
        for i in range(3): # 遍历 M1, M2, M3
            pos_m_current = pos_m_currents[i]
            all_ok = True
            for cyl_point in cylinder_surface_points:
                dist = point_segment_distance(pos_g_current, pos_m_current, cyl_point)
                if dist >= 10.0:
                    all_ok = False
                    break
            if all_ok:
                ans += P3_INTERVAL

    return ans

# --- 预处理参数 (与您提供的代码相同) ---
FIXED_V_FY2 = 120.0
dx = 0
dy = -1
FIXED_ANGLE_RAD = math.atan2(dy, dx)
MAX_TIME_HORIZON_S = 25
DIST_STEP = 1.0
Z_STEP = 1.0
P3_INTERVAL = 0.2
NUM_CYLINDER_POINTS = 100

# --- 全局常量 ---
V_M1 = 300.0
V_G_Z_p3 = 3.0
A_G_Z = -9.8
TARGET_M = np.array([0.0, 0.0, 0.0])
CYLINDER_CENTER = np.array([0., 200., 0.])

# --- 第五问 修改: 定义三个M点 ---
P_M_starts = np.array([
    [20000.0, 0.0, 2000.0],    # M1
    [19000.0, 600.0, 2100.0],  # M2
    [18000.0, -600.0, 1900.0]  # M3
])
# 向量化计算所有M点的方向向量
norms = np.linalg.norm(TARGET_M - P_M_starts, axis=1, keepdims=True)
DIR_M_array = (TARGET_M - P_M_starts) / norms

# 初始位置仍以FY2为基准
START_POINT = np.array([12000.0, 1400.0, 1400.0])
Z_MAX = START_POINT[2]

# --- 预生成圆柱体点 (只需一次) ---
cylinder_surface_points = generate_cylinder_points(
    CYLINDER_CENTER, 7.0, 10.0, NUM_CYLINDER_POINTS
)

# --- 预处理主程序 (结构不变) ---
print("--- 开始为 FY2 & 三个M点 进行预处理计算 ---")
print(f"固定速度: {FIXED_V_FY2:.1f}, 固定角度: {math.degrees(FIXED_ANGLE_RAD):.2f} 度")

max_dist = FIXED_V_FY2 * MAX_TIME_HORIZON_S
dist_axis = np.arange(0, max_dist + DIST_STEP/2, DIST_STEP)
z_axis = np.arange(0, Z_MAX + Z_STEP/2, Z_STEP)

ans_grid = np.full((len(z_axis), len(dist_axis)), -1.0)
dir_vector_xy = np.array([math.cos(FIXED_ANGLE_RAD), math.sin(FIXED_ANGLE_RAD)])

total_points = len(z_axis) * len(dist_axis)
count = 0
for i, z in enumerate(z_axis):
    for j, d in enumerate(dist_axis):
        count += 1
        if count % 5000 == 0:
            print(f"  计算进度: {count}/{total_points} ({count/total_points*100:.1f}%)")

        T = d / FIXED_V_FY2
        delta_z = z - START_POINT[2]

        if delta_z > 0: continue
        t2_squared = 2 * delta_z / A_G_Z
        if t2_squared < 0: continue
        t2 = np.sqrt(t2_squared)
        if t2 > T: continue

        pos_g_xy = START_POINT[:2] + dir_vector_xy * d
        pos_g_start = np.array([pos_g_xy[0], pos_g_xy[1], z])

        # 调用的是新版的、计算三M点总和的ans函数
        ans = calculate_ans_for_phase3(pos_g_start, T)

        ans_grid[i, j] = ans

print("--- 预处理计算完成 ---")

# --- 可视化结果 (标签已更新) ---
print("正在生成 FY2 & 三个M点 的预处理结果热力图...")

ans_grid_masked = np.ma.masked_where(ans_grid < 0, ans_grid)

plt.figure(figsize=(12, 9))
im = plt.imshow(ans_grid_masked, origin='lower', aspect='auto',
                extent=[dist_axis.min(), dist_axis.max(), z_axis.min(), z_axis.max()],
                cmap='viridis')

cbar = plt.colorbar(im)
cbar.set_label('Sum of ANS for M1,M2,M3 (s)')

plt.xlabel(f"沿方向前进的距离 (对应T范围 0s ~ {MAX_TIME_HORIZON_S:.1f}s)")
plt.ylabel("G点第三阶段初始Z坐标")
plt.title(f"ANS(M1+M2+M3)分布热力图\n(V_FY2={FIXED_V_FY2:.1f}, Angle={math.degrees(FIXED_ANGLE_RAD):.2f}°)")

plt.show()