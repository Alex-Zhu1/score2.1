from PIL import Image
import numpy as np
import os

def analyze_image(path, save_alpha=True):
    img = Image.open(path)
    arr = np.array(img)

    print(f"Path: {path}")
    print(f"Mode: {img.mode}")
    print(f"Shape: {arr.shape}")
    print(f"Min/Max: {arr.min()} / {arr.max()}")
    print(f"Mean: {arr.mean():.4f}")
    print(f"Has Alpha: {'A' in img.mode}")
    print("-" * 40)

    # ---------- 提取 Alpha 通道 ----------
    if 'A' in img.mode:
        alpha = img.split()[-1]   # 获取最后一维，即 A 通道
        alpha_arr = np.array(alpha)
        print(f"Alpha 统计：min={alpha_arr.min()}, max={alpha_arr.max()}, mean={alpha_arr.mean():.2f}")

        # 可选：保存 alpha 通道可视化结果
        if save_alpha:
            save_path = os.path.splitext(path)[0] + "_alpha.png"
            alpha.save(save_path)
            print(f"Alpha 通道已保存到: {save_path}")
        # 可选：直接显示
        # alpha.show()
    else:
        print("❌ 此图像没有 Alpha 通道")

# 示例调用
# analyze_image('/home/haiming.zhu/hoi/InvScore/demos/demo.png')
# analyze_image('/home/haiming.zhu/hoi/Hunyuan3D-2.1/hy3dshape/submodules/hamer/examples/325_cropped_hoi_1.png')
# analyze_image('/home/haiming.zhu/hoi/InvScore/data/325_cropped_hoi_1_hoi.png')
# analyze_image("/home/haiming.zhu/hoi/Hunyuan3D-2.1/hy3dshape/output_meshes/2025-10-23_16-29-44_670487/325_cropped_hoi_1.png")
# analyze_image("/home/haiming.zhu/hoi/InvScore/data/325_cropped_hoi_1nwe.png")
hand_image_path = '/home/haiming.zhu/hoi/InvScore/data/325_cropped_hoi_1h.png'
hand_image_path1 = "/home/haiming.zhu/hoi/Hunyuan3D-2.1/hy3dshape/output_meshes_之前/2025-10-22_14-23-51_670487/rembg_1.png"  # hand必须是这个不行的rgba
analyze_image(hand_image_path)
analyze_image(hand_image_path1)
print("-------score2.1 infer output ---------")
analyze_image('/home/haiming.zhu/hoi/InvScore/data/325_cropped_hoi_1h.png')
analyze_image("/mnt/data/users/haiming.zhu/HOI2/score2.1/score2.1/infer_output/2025-10-29_10-50-08_92a92908/325_cropped_hoi_1h.png")
