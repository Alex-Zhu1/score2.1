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

# 大概率是错误的图片，可以进行inversion

ref_path = '/mnt/data/users/haiming.zhu/hoi/InvScore/data/325_cropped_hoi_1.png'
image_path = '/mnt/data/users/haiming.zhu/hoi/Hunyuan3D-2.1/hy3dshape/submodules/hamer/examples/325_cropped_hoi_1.png'  # 奇怪了，为啥这个要搭配那个不行rgba hand才行。我感觉这个dino encode有问题。需要金一步测试
# image_path = '/mnt/data/users/haiming.zhu/hoi/InvScore/data/325_cropped_hoi_1.png'
# hand_image_path = '/mnt/data/users/haiming.zhu/hoi/InvScore/data/325_cropped_hoi_1h.png'
# hand_image_path = "/mnt/data/users/haiming.zhu/hoi/Hunyuan3D-2.1/hy3dshape/output_meshes_之前/2025-10-22_14-23-51_670487/rembg_1.png"  # hand必须是这个不行的rgba
hand_image_path = '/mnt/data/users/haiming.zhu/hoi/Hunyuan3D-2.1/hy3dshape/HOI_data/input_img/rembg_1.png'  # 这个hand是补全的
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

# -------------------- Helper Functions --------------------
rembg = BackgroundRemover()

def prepare_for_hunyuan3d(image: Image.Image, image_path: str, output_dir: str, force_rembg=True):
    """去背景并保存到输出目录"""
    if force_rembg:
        image = rembg(image)

    basename = os.path.basename(image_path)
    name, _ = os.path.splitext(basename)
    out_path = os.path.join(output_dir, f"{name}.png")
    image.save(out_path)
    print(f"[INFO] Saved processed image: {out_path}")
    return image

# -------------------- Image Preprocessing --------------------
image = prepare_for_hunyuan3d(Image.open(image_path).convert("RGBA"), image_path=image_path, output_dir=actual_output_dir)
hand_image = prepare_for_hunyuan3d(Image.open(hand_image_path).convert("RGBA"), image_path=hand_image_path, output_dir=actual_output_dir)
object_image = prepare_for_hunyuan3d(Image.open(object_path).convert("RGBA"), image_path=object_path, output_dir=actual_output_dir)
ref_img = prepare_for_hunyuan3d(Image.open(ref_path).convert("RGBA"), image_path=ref_path, output_dir=actual_output_dir)

# -------------------- Mesh Generation --------------------
print("[INFO] Running Hunyuan3D pipeline ...")
mesh = pipeline_shapegen(
    ref=ref_img,
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

# -------------------- Save Mesh Results --------------------
if not isinstance(mesh, list):
    mesh = [mesh]

mesh_input_paths = [image_path, hand_image_path, object_path]
for idx, m in enumerate(mesh):
    basename = os.path.basename(mesh_input_paths[idx]) if idx < len(mesh_input_paths) else f"mesh_{idx}"
    name, _ = os.path.splitext(basename)
    save_path = os.path.join(actual_output_dir, f"{name}.glb")
    m.export(save_path)
    print(f"[INFO] Saved mesh {idx+1}/{len(mesh)} to {save_path}")

print("[INFO] ✅ All done successfully!")