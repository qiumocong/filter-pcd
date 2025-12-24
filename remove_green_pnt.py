import open3d as o3d
import numpy as np
import matplotlib.colors as mcolors


def remove_green_leaves(pcd):
    colors_rgb = np.asarray(pcd.colors)

    if len(colors_rgb) == 0:
        print("错误：点云不包含颜色信息！")
        return pcd

    colors_hsv = mcolors.rgb_to_hsv(colors_rgb)

    # H (Hue): 色调。绿色通常在 60° 到 180° 之间。
    # 归一化到 [0, 1] 后：
    # 绿色大致范围: 0.2 (72°) - 0.5 (180°)
    # S (Saturation): 饱和度。太低就是灰色/白色，保留饱和度较高的绿色。
    # V (Value): 亮度。太暗可能是阴影，视情况过滤。

    # 调整这些阈值以适应你的具体场景
    min_green_h = 0.1  # 约 65度 (黄绿色边界)
    max_green_h = 0.8  # 约 160度 (青绿色边界)
    min_saturation = 0.2  # 排除看起来像绿色的灰色物体
    min_value = 0.1  # 排除极暗的噪点

    # 找出所有符合“绿色”定义的点
    green_mask = (
            (colors_hsv[:, 0] >= min_green_h) &
            (colors_hsv[:, 0] <= max_green_h) &
            (colors_hsv[:, 1] >= min_saturation) &
            (colors_hsv[:, 2] >= min_value)
    )

    non_green_mask = ~green_mask

    # np.where 返回索引数组
    inlier_indices = np.where(non_green_mask)[0]
    pcd_filtered = pcd.select_by_index(inlier_indices)

    # 统计信息
    removed_count = len(pcd.points) - len(pcd_filtered.points)
    print(f"原始点数: {len(pcd.points)}")
    print(f"移除绿色点数: {removed_count}")
    print(f"剩余点数: {len(pcd_filtered.points)}")

    return pcd_filtered


# --- 测试代码 ---
if __name__ == "__main__":
    bush = o3d.io.read_point_cloud("point_clouds/bush.ply")
    processed_bush = remove_green_leaves(bush)
    o3d.visualization.draw_geometries([bush])
    o3d.visualization.draw_geometries([processed_bush])
