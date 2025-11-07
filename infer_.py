import argparse
from PIL import Image
from hy3dshape.rembg import BackgroundRemover
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline, export_to_trimesh
import torch
import os
import time
import random
from pathlib import Path
import shutil
import logging
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def gen_save_folder(save_dir: str, max_size: int = 200) -> str:
    """
    Create a unique folder for saving outputs, removing oldest folder if limit is reached.
    
    Args:
        save_dir: Base directory for saving outputs
        max_size: Maximum number of subdirectories to keep
        
    Returns:
        Path to the newly created folder
    """
    try:
        os.makedirs(save_dir, exist_ok=True)
        dirs = [f for f in Path(save_dir).iterdir() if f.is_dir()]
        
        if len(dirs) >= max_size:
            oldest_dir = min(dirs, key=lambda x: x.stat().st_ctime)
            shutil.rmtree(oldest_dir, ignore_errors=True)
            logger.info(f"Removed oldest folder: {oldest_dir}")
        
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S") + f"_{random.randint(0, 999999):06d}"
        new_folder = os.path.join(save_dir, timestamp)
        os.makedirs(new_folder, exist_ok=True)
        logger.info(f"Created new folder: {new_folder}")
        return new_folder
    except OSError as e:
        logger.error(f"Failed to create or manage save folder: {e}")
        raise


def load_and_process_image(image_path: str, rembg: BackgroundRemover, output_dir: str, prefix: str) -> Image.Image:
    """
    Load an image, remove background, and save the processed version.
    
    Args:
        image_path: Path to input image
        rembg: BackgroundRemover instance
        output_dir: Directory to save processed image
        prefix: Prefix for the saved filename
        
    Returns:
        Processed PIL Image
    """
    logger.info(f"Loading and processing image: {image_path}")
    image = Image.open(image_path).convert("RGBA")
    image = rembg(image)
    
    # Save background-removed image
    save_path = os.path.join(output_dir, f"{prefix}_nobg.png")
    image.save(save_path)
    # logger.info(f"Saved background-removed image to {save_path}")
    
    return image


def set_random_seeds(seed: int):
    """Set random seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seeds set to: {seed}")


def main(args):
    # Set seeds for reproducibility
    set_random_seeds(args.seed)

    # Create a unique subfolder with timestamp inside the output_dir
    actual_output_dir = gen_save_folder(args.output_dir)
    logger.info(f"Using output directory: {actual_output_dir}")

    # Initialize models
    logger.info("Loading Hunyuan3D model...")
    pipeline_shapegen = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        args.model_path,
        device=args.device
    )

    rembg = BackgroundRemover()

    # Load and process all images
    hoi_img = load_and_process_image(args.image_path, rembg, actual_output_dir, "hoi")
    hand_img = load_and_process_image(args.hand_path, rembg, actual_output_dir, "hand")
    object_img = load_and_process_image(args.object_path, rembg, actual_output_dir, "object")
    ref_img = load_and_process_image(args.ref_path, rembg, actual_output_dir, "ref")

    # Generate 3D meshes
    # logger.info("Generating 3D meshes...")
    outputs = pipeline_shapegen(
        image=hoi_img,
        ref=ref_img,
        hand_image=hand_img,
        object_image=object_img,
        mesh_path=args.mesh_path,
        moge_path=args.moge_path,
        moge_hand_path=args.moge_hand_path,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        # generator=torch.Generator(device=args.device).manual_seed(args.seed), # 跟sd一样，一旦确定generator, inversion结果就很差
        output_type="mesh",
        do_inversion_stage=True
    )
    # logger.info("Mesh generation completed")

    # Export and save meshes
    meshes = export_to_trimesh(outputs)
    basename = os.path.basename(args.image_path)
    name, _ = os.path.splitext(basename)
    
    # logger.info(f"Exporting {len(meshes)} mesh(es)...")
    for idx, mesh in enumerate(meshes):
        if len(meshes) > 1:
            save_path = os.path.join(actual_output_dir, f"{name}_mesh_{idx}.glb")
        else:
            save_path = os.path.join(actual_output_dir, f"{name}.glb")
        
        mesh.export(save_path)
        logger.info(f"Saved mesh to {save_path}")

    # logger.info("All processing completed successfully!")
    logger.info(f"All outputs saved to: {actual_output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate 3D meshes from 2D images using Hunyuan3D for hand-object interaction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model and device settings
    parser.add_argument(
        "--model_path",
        type=str,
        default="tencent/Hunyuan3D-2.1",
        help="Path to the pretrained Hunyuan3D model"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run the model on"
    )
    
    # Input/Output paths
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Path to the HOI (hand-object interaction) image"
    )
    parser.add_argument(
        "--hand_path",
        type=str,
        required=True,
        help="Path to the hand image"
    )
    parser.add_argument(
        "--object_path",
        type=str,
        required=True,
        help="Path to the object image"
    )
    parser.add_argument(
        "--ref_path",
        type=str,
        default='/home/haiming.zhu/hoi/InvScore/data/325_cropped_hoi_1.png',
        help="Path to the reference image"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output_meshes",
        help="Base directory for saving outputs"
    )
    
    # Generation parameters
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=20,
        help="Number of inference steps for mesh generation"
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=5.0,
        help="Guidance scale for the generation process"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    # Optional paths
    parser.add_argument(
        "--mesh_path",
        type=str,
        default=None,
        help="Optional custom output mesh path (.obj or .glb)"
    )
    parser.add_argument(
        "--moge_path",
        type=str,
        default=None,
        help="Optional MOGE pointmap path for registration"
    )
    parser.add_argument(
        "--moge_hand_path",
        type=str,
        default=None,
        help="Optional MOGE hand pointmap path for registration"
    )
    
    args = parser.parse_args()
    
    # Validate CUDA availability if specified
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available. Falling back to CPU.")
        args.device = "cpu"
    
    try:
        main(args)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        raise