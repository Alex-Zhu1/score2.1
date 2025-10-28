# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.

import os
import random
import datetime
import uuid
import numpy as np
import torch
from PIL import Image
from hy3dshape.rembg import BackgroundRemover
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

# -------------------- Settings --------------------
model_path = 'tencent/Hunyuan3D-2.1'
print(f"[INFO] Loading model from {model_path} ...")
pipeline_shapegen = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model_path)

check_box_rembg = True  # 是否强制去背景

# -------------------- Input Paths --------------------



image_path = '/mnt/data/users/haiming.zhu/hoi/InvScore/data/325_cropped_hoi_1.png'
hand_image_path = '/mnt/data/users/haiming.zhu/hoi/InvScore/data/325_cropped_hoi_1h.png'
# hand_image_path = "/mnt/data/users/haiming.zhu/hoi/Hunyuan3D-2.1/hy3dshape/output_meshes/2025-10-22_14-23-51_670487/rembg_1.png"
object_path = '/mnt/data/users/haiming.zhu/hoi/InvScore/data/325_cropped_hoi_1o.png'


# for registration and inversion
mesh_path = "/mnt/data/users/haiming.zhu/hoi/Hunyuan3D-2.1/hy3dshape/submodules/hamer/demo_out_1/325_cropped_hoi_1_0.obj"
# mesh_path = "/mnt/data/users/haiming.zhu/hoi/Hunyuan3D-2.1/hy3dshape/HOI_data/hand_shape/325_cropped_hoi_1_0_watertight.obj"
moge_path = "/mnt/data/users/haiming.zhu/hoi/Hunyuan3D-2.1/hy3dshape/outputs_depth/325_cropped_hoi_1/pointcloud.ply"
moge_hand_path = "/mnt/data/users/haiming.zhu/hoi/Hunyuan3D-2.1/hy3dshape/outputs_hand_depth/325_cropped_hoi_1/pointcloud.ply"

# -------------------- Reproducibility --------------------
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# -------------------- Output Directory --------------------
date_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
unique_id = str(uuid.uuid4())[:8]
actual_output_dir = os.path.join("infer_output", f"{date_str}_{unique_id}")
os.makedirs(actual_output_dir, exist_ok=True)
print(f"[INFO] Output directory: {actual_output_dir}")

# -------------------- Background Removal & Prepare --------------------
rembg = BackgroundRemover()

def prepare_for_hunyuan3d(img: Image.Image, image_path: str, output_dir: str):
    """统一为 RGBA 并确保背景透明，同时保存处理后的图片"""
    # 转为 RGBA
    if img.mode != "RGBA":
        img = img.convert("RGBA")

    # 背景去除
    if check_box_rembg or img.mode == "RGB":
        img = rembg(img)

    # 去掉 ICC profile
    img.info.pop('icc_profile', None)

    # 保存处理后的图片
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        basename = os.path.basename(image_path)
        name, _ = os.path.splitext(basename)
        save_name = f"{name}_prepared.png"
        save_path = os.path.join(output_dir, save_name)
        img.save(save_path)
        print(f"[INFO] Saved processed image → {save_path}")

    return img

# 处理图片
image = prepare_for_hunyuan3d(Image.open(image_path), image_path=image_path, output_dir=actual_output_dir)
hand_image = prepare_for_hunyuan3d(Image.open(hand_image_path), image_path=hand_image_path, output_dir=actual_output_dir)
object_image = prepare_for_hunyuan3d(Image.open(object_path), image_path=object_path, output_dir=actual_output_dir)

# -------------------- Mesh Generation --------------------
print("[INFO] Running Hunyuan3D pipeline ...")
mesh = pipeline_shapegen(
    image=image,
    hand_image=hand_image,
    object_image=object_image,
    num_inference_steps=20,
    guidance_scale=5.0,
    mesh_path=mesh_path,
    moge_path=moge_path,
    moge_hand_path=moge_hand_path,
    do_inversion_stage=True
)

# 确保输出是 list
if not isinstance(mesh, list):
    mesh = [mesh]

# -------------------- Save Mesh Results --------------------
mesh_input_paths = [image_path, hand_image_path, object_path]
for idx, m in enumerate(mesh):
    basename = os.path.basename(mesh_input_paths[idx]) if idx < len(mesh_input_paths) else f"mesh_{idx}"
    name, _ = os.path.splitext(basename)
    save_path = os.path.join(actual_output_dir, f"{name}.glb")
    m.export(save_path)
    print(f"[INFO] Saved mesh {idx+1}/{len(mesh)} to {save_path}")

print("[INFO] All done successfully!")
