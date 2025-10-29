# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.

import os
import random
import datetime
import uuid
import argparse
import numpy as np
import torch
from PIL import Image
from hy3dshape.rembg import BackgroundRemover
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline


def prepare_for_hunyuan3d(image: Image.Image, image_path: str, output_dir: str, rembg, force_rembg=True):
    """去背景并保存到输出目录"""
    if force_rembg:
        image = rembg(image)

    basename = os.path.basename(image_path)
    name, _ = os.path.splitext(basename)
    out_path = os.path.join(output_dir, f"{name}.png")
    image.save(out_path)
    print(f"[INFO] Saved processed image: {out_path}")
    return image


def main(args):
    # -------------------- Settings --------------------
    print(f"[INFO] Loading model from {args.model_path} ...")
    pipeline_shapegen = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(args.model_path)

    # -------------------- Reproducibility --------------------
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # -------------------- Generator --------------------
    generator = torch.Generator(
        device="cuda" if torch.cuda.is_available() else "cpu"
    ).manual_seed(args.seed)

    # -------------------- Output Directory --------------------
    date_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    unique_id = str(uuid.uuid4())[:8]
    actual_output_dir = os.path.join(args.output_dir, f"{date_str}_{unique_id}")
    os.makedirs(actual_output_dir, exist_ok=True)
    print(f"[INFO] Output directory: {actual_output_dir}")

    # -------------------- Image Preprocessing --------------------
    rembg = BackgroundRemover()

    image = prepare_for_hunyuan3d(
        Image.open(args.image_path).convert("RGBA"),
        args.image_path,
        actual_output_dir,
        rembg,
        force_rembg=args.force_rembg,
    )

    hand_image = prepare_for_hunyuan3d(
        Image.open(args.hand_image_path).convert("RGBA"),
        args.hand_image_path,
        actual_output_dir,
        rembg,
        force_rembg=args.force_rembg,
    )

    object_image = prepare_for_hunyuan3d(
        Image.open(args.object_path).convert("RGBA"),
        args.object_path,
        actual_output_dir,
        rembg,
        force_rembg=args.force_rembg,
    )

    ref_img = prepare_for_hunyuan3d(
        Image.open(args.ref_path).convert("RGBA"),
        args.ref_path,
        actual_output_dir,
        rembg,
        force_rembg=args.force_rembg,
    )

    # -------------------- Mesh Generation --------------------
    print("[INFO] Running Hunyuan3D pipeline ...")
    mesh = pipeline_shapegen(
        ref=ref_img,
        image=image,
        hand_image=hand_image,
        object_image=object_image,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        mesh_path=args.mesh_path,
        moge_path=args.moge_path,
        moge_hand_path=args.moge_hand_path,
        do_inversion_stage=args.do_inversion_stage,
        generator=generator,
    )

    # -------------------- Save Mesh Results --------------------
    if not isinstance(mesh, list):
        mesh = [mesh]

    mesh_input_paths = [args.image_path, args.hand_image_path, args.object_path]
    for idx, m in enumerate(mesh):
        basename = os.path.basename(mesh_input_paths[idx]) if idx < len(mesh_input_paths) else f"mesh_{idx}"
        name, _ = os.path.splitext(basename)
        save_path = os.path.join(actual_output_dir, f"{name}.glb")
        m.export(save_path)
        print(f"[INFO] Saved mesh {idx+1}/{len(mesh)} to {save_path}")

    print("[INFO] ✅ All done successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hunyuan3D Inference Script")

    # -------------------- Model and Paths --------------------
    parser.add_argument("--model_path", type=str, default="tencent/Hunyuan3D-2.1", help="模型路径")
    parser.add_argument("--ref_path", type=str, default='/mnt/data/users/haiming.zhu/hoi/InvScore/data/325_cropped_hoi_1.png')
    parser.add_argument("--image_path", type=str, default="/mnt/data/users/haiming.zhu/hoi/InvScore/data/325_cropped_hoi_1.png")
    parser.add_argument("--hand_image_path", type=str, default="/mnt/data/users/haiming.zhu/hoi/Hunyuan3D-2.1/hy3dshape/HOI_data/input_img/rembg_1.png")
    parser.add_argument("--object_path", type=str, default="/mnt/data/users/haiming.zhu/hoi/InvScore/data/325_cropped_hoi_1o.png")
    parser.add_argument("--mesh_path", type=str, default="/mnt/data/users/haiming.zhu/hoi/Hunyuan3D-2.1/hy3dshape/submodules/hamer/demo_out_1/325_cropped_hoi_1_0.obj")
    parser.add_argument("--moge_path", type=str, default="/mnt/data/users/haiming.zhu/hoi/Hunyuan3D-2.1/hy3dshape/outputs_depth/325_cropped_hoi_1/pointcloud.ply")
    parser.add_argument("--moge_hand_path", type=str, default="/mnt/data/users/haiming.zhu/hoi/Hunyuan3D-2.1/hy3dshape/outputs_hand_depth/325_cropped_hoi_1/pointcloud.ply")
    parser.add_argument("--output_dir", type=str, default="infer_output", help="输出目录")

    # -------------------- Inference Settings --------------------
    parser.add_argument("--num_inference_steps", type=int, default=20, help="推理步数")
    parser.add_argument("--guidance_scale", type=float, default=5.0, help="guidance scale 参数")
    parser.add_argument("--do_inversion_stage", action="store_true", help="是否执行 inversion 阶段")
    parser.add_argument("--force_rembg", action="store_true", help="是否强制去背景")

    # -------------------- Randomness & Generator --------------------
    parser.add_argument("--seed", type=int, default=42, help="全局随机种子")

    args = parser.parse_args()
    main(args)
