
import copy
# import importlib
import inspect
import os
from typing import List, Optional, Union

import numpy as np
import torch
import trimesh
# import yaml
from PIL import Image
# from diffusers.utils.torch_utils import randn_tensor
# from diffusers.utils.import_utils import is_accelerate_version, is_accelerate_available
from tqdm import tqdm

# from .models.autoencoders import ShapeVAE
# from .models.autoencoders import SurfaceExtractors
from .utils import logger, synchronize_timer



from .module.mesh_align_fun import align_meshes
from .schedulers import UniInvEulerScheduler
from .surface_loaders import SharpEdgeSurfaceLoader

loader = SharpEdgeSurfaceLoader(
    num_sharp_points=0,
    num_uniform_points=81920,
)

def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


@synchronize_timer('Export to trimesh')
def export_to_trimesh(mesh_output, mc_algo='mc', requires_grad=False):
    if isinstance(mesh_output, list):
        outputs = []
        for mesh in mesh_output:
            if mesh is None:
                outputs.append(None)
            else:
                if requires_grad:
                    mesh_v, mesh_f = mesh # 这里是tensor
                    mesh_f = mesh_f.detach().cpu().numpy()
                    mesh_v = mesh_v.detach().cpu().numpy()
                    mesh_v = mesh_v.astype(np.float32)
                    mesh_f = np.ascontiguousarray(mesh_f)[:, ::-1]
                    mesh_output = trimesh.Trimesh(mesh_v, mesh_f)
                    outputs.append(mesh_output)
                else:
                    # if mc_algo == 'dmc':
                    #     mesh_v, mesh_f = mesh
                    #     mesh_f = mesh_f[:, ::-1]
                    #     mesh_output = trimesh.Trimesh(mesh_v, mesh_f)
                    #     outputs.append(mesh_output)
                    # else:
                    mesh.mesh_f = mesh.mesh_f[:, ::-1]
                    mesh_output = trimesh.Trimesh(mesh.mesh_v, mesh.mesh_f)
                    outputs.append(mesh_output)
        return outputs
    else:
        mesh_output.mesh_f = mesh_output.mesh_f[:, ::-1]
        mesh_output = trimesh.Trimesh(mesh_output.mesh_v, mesh_output.mesh_f)
        return mesh_output

class HunyuanInversion:
    """
    Stable Diffusion 风格的 inversion helper。
    不继承 pipeline，不加载模型，只复用 pipeline 的 UNet / VAE / Processor。
    """

    def __init__(self, pipeline):
        # 共享 pipeline 的组件
        self.device = pipeline.device
        self.dtype = pipeline.dtype
        self.model = pipeline.model
        self.vae = pipeline.vae
        self.image_processor = pipeline.image_processor
        self.set_surface_extractor = pipeline.set_surface_extractor

        self.scheduler = pipeline.scheduler
        self._export = pipeline._export

        # inversion 专用 scheduler
        self.inv_scheduler = UniInvEulerScheduler(num_train_timesteps=1000)

    def __call__(self, latents, cond_ref, cond_hand,
                timesteps, 
                do_classifier_free_guidance, guidance_scale,  
                guidance,
                mesh_path,
                moge_path,
                moge_hand_path,
                num_inference_steps,
                sigmas,
                eta,
                generator,
                box_v,
                octree_resolution,
                mc_level,
                mc_algo,
                num_chunks,
                output_type,
                enable_pbar=True):
            
        phase1_scheduler = copy.deepcopy(self.scheduler)  # pipeline 的 scheduler
        timesteps_phase1 = timesteps
        
        with synchronize_timer('Phase 1: Partial Sampling + Inversion'):
            pbar = tqdm(timesteps_phase1, disable=not enable_pbar, desc="(Phase 1) Partial Sampling + Inversion:")
            for i, t in enumerate(pbar):
                if do_classifier_free_guidance:
                    latent_model_input = torch.cat([latents] * 2)
                else:
                    latent_model_input = latents

                timestep = t.expand(latent_model_input.shape[0]).to(latents.dtype)
                timestep = timestep / phase1_scheduler.config.num_train_timesteps
                
                noise_pred = self.model(latent_model_input, timestep, cond_ref, guidance=guidance)

                if do_classifier_free_guidance:
                    noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

                outputs = phase1_scheduler.step(noise_pred, t, latents)
                latents = outputs.prev_sample

                if i == 9 or i == num_inference_steps - 1:
                    pbar.close()
                    
                    # 导出中间 mesh
                    mesh_i = self._export(
                        outputs.pred_original_sample,
                        box_v=box_v,
                        mc_level=mc_level,
                        num_chunks=num_chunks,
                        octree_resolution=octree_resolution,
                        mc_algo='mc',   # NOTE: 使用mc + norm, 以配准到cube的hunyuna空间
                        enable_pbar=enable_pbar,
                    )

                    # 可视化。 这里的mesh 是在 hunyuan 生成空间下的mesh
                    if enable_pbar:
                        print(f"[Phase 1] Exporting intermediate mesh at step {i+1}")
                        dir = "vis_phase1_mid_mesh"
                        os.makedirs(dir, exist_ok=True)
                        import time
                        if isinstance(mesh_i, list):
                            for midx, m in enumerate(mesh_i):
                                m.export(f"{dir}/check_step10_{midx}_{time.time()}.glb")
                        else:
                            mesh_i.export(f"{dir}/check_step10_{time.time()}.glb")

                    # # Registration
                    logger.info(f"[Phase 1] Start registration + inversion...")
                    Th, To = self.registration(
                        hunyuan_mesh=mesh_i[0] if isinstance(mesh_i, list) else mesh_i,
                        hamer_mesh=mesh_path,
                        moge_pointmap=moge_path,
                        moge_hand_pointmap=moge_hand_path
                    )

                    # Inversion
                    inversion = True if mesh_path is not None else False
                    latents = self.inversion(
                        mesh_path=mesh_path,
                        Th=Th,
                        To=To,
                        device=self.device,
                        batch_size=1,
                        inversion=inversion,
                        box_v=box_v,
                        octree_resolution=octree_resolution,
                        mc_level=mc_level,
                        num_chunks=num_chunks,
                        mc_algo=mc_algo,   # 需要使用mc，查看 inverison 的hunyuan mesh在哪
                        enable_pbar=enable_pbar,
                        cond=cond_hand,
                        num_inference_steps=num_inference_steps,
                        timesteps=timesteps,
                        do_classifier_free_guidance=do_classifier_free_guidance,
                        guidance_scale=guidance_scale,
                        generator=generator,
                    )
                    
                    del outputs, mesh_i, phase1_scheduler, Th
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    break
                
        return latents, To

    # ===============================
    #        FULL INVERSION API
    # ===============================
    def inversion(
        self,
        mesh_path: Union[str, List[str]],
        Th: np.ndarray,
        To: np.ndarray,
        device: torch.device,
        batch_size: int = 1,
        inversion: bool = True,
        box_v=1.01,
        octree_resolution=256,
        mc_level=-1 / 512,
        num_chunks=8000,
        mc_algo=None,
        enable_pbar=True,
        cond: torch.FloatTensor = None,
        num_inference_steps: int = 20,
        timesteps: List[int] = None,
        do_classifier_free_guidance: bool = True,
        guidance_scale: float = 5.0,
        generator=None,
    ) -> torch.FloatTensor:

        assert Th.shape == (4, 4) and To.shape == (4, 4)

        if inversion:
            surface = loader(mesh_path, Th, To).to(self.device, dtype=self.dtype)   # 这里hand mesh得配准到hunyuan的norm空间下，不下vae是错误的
            latents = self.vae.encode(surface)
            latents = self.vae.scale_factor * latents

            # test decode
            vis_test_decoding = True
            if vis_test_decoding:
                import time
                mesh = self._export(
                        latents.detach(),
                        box_v=box_v,
                        mc_level=mc_level,
                        num_chunks=num_chunks,
                        octree_resolution=octree_resolution,
                        mc_algo='dmc',   # NOTE: 这里可以测试dmc的space
                        enable_pbar=enable_pbar,
                    )
                if isinstance(mesh, list):
                    for mid, m in enumerate(mesh):
                        m.export(f"check_hand_flexicube_{mid}_{time.time()}.glb")
                else:
                    mesh.export(f"check_hand_{time.time()}.glb")

                # del latents_rec, outputs, mesh
                # torch.cuda.empty_cache()

            latents = self.inversion_loop(
                latents,
                cond,
                device=device,
                inversion_steps=num_inference_steps,
                do_classifier_free_guidance=do_classifier_free_guidance,
                guidance_scale=1.0,
                timesteps=timesteps,
            )

        else:
            latents = self.pipeline.prepare_latents(batch_size, self.dtype, device, generator)

        return latents

    # ===============================
    #          INVERSION LOOP
    # ===============================
    def inversion_loop(
        self,
        latents: torch.FloatTensor,
        cond: torch.FloatTensor,
        device: torch.device,
        inversion_steps: int = 20,
        sigmas: List[float] = None,
        do_classifier_free_guidance: bool = False,
        guidance_scale: float = 1.0,
        timesteps: List[int] = None,
        guidance=None,
        enable_pbar=True,
        generator=None,
    ) -> torch.FloatTensor:

        cond_hand = copy.deepcopy(cond)

        if do_classifier_free_guidance:
            cond_hand = cond
        else:
            cond_hand = [cond["main"][[-1], :, :]]  # uncond

        sigmas = np.linspace(0, 1, inversion_steps) if sigmas is None else sigmas

        timesteps, inversion_steps = retrieve_timesteps(
            self.inv_scheduler, inversion_steps, device, sigmas=sigmas
        )

        with synchronize_timer("Diffusion Inversion"):
            for i, t in enumerate(
                tqdm(timesteps, disable=not enable_pbar, desc="Diffusion Inversion:")
            ):
                # prepare input
                if do_classifier_free_guidance:
                    latent_model_input = torch.cat([latents] * 2)
                else:
                    latent_model_input = latents

                timestep = (
                    t.expand(latent_model_input.shape[0]).to(latents.dtype)
                    / self.inv_scheduler.config.num_train_timesteps
                )

                noise_pred = self.model(latent_model_input, timestep, cond_hand, guidance=guidance)

                if do_classifier_free_guidance:
                    noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

                outputs = self.inv_scheduler.step(noise_pred, t, latents)
                latents = outputs.prev_sample

        return latents


    # ===============================
    #          REGISTRATION
    # ===============================
    def registration(self, hunyuan_mesh, hamer_mesh, moge_pointmap, moge_hand_pointmap):

        # Th, from hamer hand mesh to Moge pointmap
        Th = align_meshes(
            source_mesh_path=hamer_mesh,
            target_mesh_path=moge_hand_pointmap,
            skip_coarse=True,
            # transformed_mesh_path="hand_registered.glb"
        )

        # To, from hunyuan generated mesh to Moge pointmap
        To = align_meshes(
            source_mesh_path=None,
            source_mesh=hunyuan_mesh,
            target_mesh_path=moge_pointmap,
            skip_coarse=True,
            # transformed_mesh_path="hunyuan_registered.glb"
        )

        return Th, To

    # ===============================
    #      VISUALIZATION HELPER
    # ===============================
    def visualize_cond_inputs(self, cond_inputs, save_dir="cond_inputs_vis"):
        os.makedirs(save_dir, exist_ok=True)

        def tensor_to_numpy(tensor):
            tensor = tensor.cpu() if tensor.is_cuda else tensor
            if tensor.dim() == 4:
                arr = tensor.permute(0, 2, 3, 1).numpy()
            elif tensor.dim() == 3:
                arr = tensor.permute(1, 2, 0).numpy()
                arr = arr[np.newaxis, ...]
            else:
                raise ValueError(f"Unsupported shape {tensor.shape}")
            return arr

        image_np = tensor_to_numpy(cond_inputs["image"])
        mask_np = tensor_to_numpy(cond_inputs["mask"])

        for idx in range(image_np.shape[0]):
            img = image_np[idx]
            msk = mask_np[idx]

            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)

            msk = ((msk + 1.0) / 2.0 * 255).astype(np.uint8)
            if msk.ndim == 3 and msk.shape[-1] == 1:
                msk = msk[:, :, 0]

            Image.fromarray(img).save(os.path.join(save_dir, f"image_{idx}.png"))
            Image.fromarray(msk).save(os.path.join(save_dir, f"mask_{idx}.png"))

        print(f"[Inversion] Saved {image_np.shape[0]} images/masks to {save_dir}")
