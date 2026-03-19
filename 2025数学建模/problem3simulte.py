from typing import Any

import numpy as np
from shapely.geometry import Polygon, LineString, Point
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys # <--- 添加的导入
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
Z_STEP = 0.5  # Z坐标步长
P3_INTERVAL = 0.005
NUM_CYLINDER_POINTS = 100

# --- 全局常量 (无需改变) ---
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

# --- 主搜索循环 ---
global_max_ans = -1
global_best_params = {}

v_min = 70
v_max = 140


def function1(
        x: int,      # x 是自变量
        V: int,      # V 是无人机的速度
        alpha: float # alpha 是一个角度值，单位为弧度
):
    return (3 * x) / V + 20 + (math.cos(alpha) / P_FY1_start[0]) * (P_FY1_start[2] - 20) * x + \
        (3 / (math.cos(math.atan2(P_M1_start[2], P_M1_start[0])) * V_M1)) * \
        (CYLINDER_CENTER[0] + (CYLINDER_CENTER[1] / (CYLINDER_CENTER[1] - x * math.sin(alpha))) * (P_FY1_start[0] - x * math.cos(alpha) - CYLINDER_CENTER[0]) - P_M1_start[0])


def function2(
        x: int,      # x 是自变量
        V: int,      # V 是无人机的速度
):
    return (9.8 / (2 * V * V)) * x * x

def simulate():
    alpha = 0.1
    alpha = math.radians(alpha)
    V = 120

    # 1. 创建用于存储绘图数据的列表
    x_values = []
    ans_values = []

    print("正在进行模拟计算，请稍候...")

    # 循环计算每一个x点对应的ans
    for x in tqdm(np.arange(0, 1500, 1)):
        y1 = function1(x, V, alpha)
        y2 = function2(x, V)
        y = y1 if y1 < y2 else y2
        p_g = np.array([P_FY1_start[0] - x * math.cos(alpha), x * math.sin(alpha), P_FY1_start[2] - y])

        # 避免除以零
        if V == 0:
            t = 0
        else:
            t = x / V

        ans = calculate_ans_for_phase3(p_g, t)

        # 2. 将每一次计算的结果存入列表
        x_values.append(x)
        ans_values.append(ans)
        print(f"x: {x:.1f}, ANS: {ans:.4f}")

    print("计算完成，正在生成图形...")

    # 3. 在循环结束后，使用matplotlib绘制图形
    # --- 开始: 专业级绘图模块 ---

    # 0. 数据准备: 将列表转换为Numpy数组，方便计算
    x_values = np.array(x_values)
    ans_values = np.array(ans_values)

    # 1. 设置一个更美观的绘图风格
    plt.style.use('seaborn-v0_8-whitegrid')

    # 2. 创建图形和坐标轴，并设置更大的尺寸
    fig, ax = plt.subplots(figsize=(14, 8), dpi=100)

    # 3. 绘制带有渐变填充的面积图，增强视觉冲击力
    # 定义渐变颜色
    color_start = '#3498db'  # 清新的蓝色
    color_end = '#9b59b6'  # 温和的紫色

    # 绘制平滑的线条
    ax.plot(x_values, ans_values, color=color_start, linewidth=2.5, zorder=3)

    # 绘制渐变填充区域
    # (这是一个高级技巧，通过在图像下方绘制一个渐变色块并用曲线进行裁剪)
    # gradient = np.linspace(0, 1, 256)
    # gradient = np.vstack((gradient, gradient))
    # im = ax.imshow(gradient, aspect='auto', extent=[x_values.min(), x_values.max(), 0, ans_values.max()],
    #                origin='lower', cmap=plt.get_cmap('cool'), zorder=1)
    #
    # from matplotlib.patches import Polygon
    # verts = list(zip(x_values, ans_values))
    # poly = Polygon(verts, facecolor='none', edgecolor='none')
    # ax.add_patch(poly)
    # im.set_clip_path(poly)

    # 4. 突出显示最重要的信息：最大值点
    # 找到最大值及其对应的x坐标
    max_ans_index = np.argmax(ans_values)
    max_x = x_values[max_ans_index]
    max_ans = ans_values[max_ans_index]

    # 在最大值点绘制一个醒目的标记
    ax.scatter(max_x, max_ans, color='#e74c3c', s=150, zorder=5,
               edgecolors='white', linewidth=2, label=f'peak: {max_ans:.2f}')

    # 绘制从最大值点到坐标轴的引导虚线
    ax.plot([max_x, max_x], [0, max_ans], color='gray', linestyle='--', linewidth=1, dashes=(5, 5))
    ax.plot([0, max_x], [max_ans, max_ans], color='gray', linestyle='--', linewidth=1, dashes=(5, 5))

    # 5. 添加优雅的注解，清晰地标出最大值
    # annotation_text = f"  best X: {max_x:.1f}\n  best ANS: {max_ans:.4f}"
    # ax.annotate(annotation_text,
    #             xy=(max_x, max_ans),
    #             xytext=(max_x + 100, max_ans * 0.8),
    #             fontsize=12,
    #             color='#34495e',
    #             bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", lw=1, alpha=0.8),
    #             arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2",
    #                             color="gray", lw=1.5))

    # 6. 美化标题、坐标轴标签和图例
    ax.set_title('V = 125 alpha = 174.81', fontsize=20, fontweight='bold', pad=20,
                 color='#2c3e50')
    ax.set_xlabel('flying distance of drone (X)', fontsize=14, labelpad=15)
    ax.set_ylabel('valid blocking time (ANS)', fontsize=14, labelpad=15)
    ax.legend(fontsize=12, frameon=True, facecolor='white', framealpha=0.9, shadow=True)

    # 7. 调整坐标轴和背景，使其更简洁
    ax.set_facecolor('#f7f9fa')  # 设置一个淡淡的背景色
    fig.patch.set_facecolor('white')

    # 移除顶部和右侧的边框，使图形更“开放”
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('gray')
    ax.spines['bottom'].set_color('gray')

    # 设置坐标轴刻度的样式
    ax.tick_params(axis='both', which='major', labelsize=12, colors='gray')

    # 设置坐标轴范围，留出一些空白，避免图形贴边
    ax.set_xlim(0, x_values.max() * 1.05)
    ax.set_ylim(0, ans_values.max() * 1.15)

    # 8. 自动调整布局并显示图形
    plt.tight_layout()
    plt.show()

    # --- 结束: 专业级绘图模块 ---

def search():
    ANGLE_INTERVAL= 1  # 角度步长
    buffer = 0.1
    for alpha in np.arange(0, 90, ANGLE_INTERVAL):
        alpha_rad = np.radians(alpha)
        for V in range(v_min, v_max + 1, V_INTERVAL):
            for x1 in np.arange(0.1, 100, 0.1):
                y1 = function1(x1, V, alpha_rad)
                y1 = max(y1, 0)  # 确保 y1 不为负值
                y2 = function2(x1, V)
                if y1 > y2 + buffer:
                    break
                y = min(y1, y2)
                p_g = np.array([P_FY1_start[0] - x1 * math.cos(alpha_rad), x1 * math.sin(alpha_rad), P_FY1_start[2] - y])

                # 避免除以零
                if V == 0:
                    t = 0
                else:
                    t = x1 / V

                ans1 = calculate_ans_for_phase3(p_g, t)
                t1 = x1 / V - math.sqrt(2 * y / 9.8)

                print(f"alpha={alpha:.1f}°, V_FY1={V:.1f}, ANS1={ans1:.3f}, x1={x1:.2f}")


if __name__ == "__main__":
    simulate()