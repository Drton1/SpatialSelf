import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import json

def plot_visium_overlay(data, img_path, scalefactors_path,
                        use_hires=False, point_size=5, alpha=0.6, color="red"):
    """
    在组织切片图像上绘制 Visium spot overlay (自动缩放).

    参数:
    - data: PyG Data 对象 (必须有 data.pos)
    - img_path: HE 切片图像路径 (tissue_lowres_image.png 或 tissue_hires_image.png)
    - scalefactors_path: spatial/scalefactors_json.json 路径
    - use_hires: 是否用 hires 图像 (True=hires, False=lowres)
    - point_size: spot 点大小
    - alpha: spot 透明度
    - color: spot 颜色
    """
    # 读取切片图
    img = mpimg.imread(img_path)
    coords = data.pos.numpy()

    # 读取 scale factor
    with open(scalefactors_path, "r") as f:
        scales = json.load(f)

    scale = scales["tissue_hires_scalef"] if use_hires else scales["tissue_lowres_scalef"]

    # 缩放坐标
    coords_scaled = coords * scale

    # 绘制
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.scatter(coords_scaled[:, 0], coords_scaled[:, 1], s=point_size, c=color, alpha=alpha)
    plt.gca().invert_yaxis()   # 反转 y 轴，使坐标和图像方向一致
    plt.axis("off")
    plt.title("Visium Spots Overlay")
    plt.show()
