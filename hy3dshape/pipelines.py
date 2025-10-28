# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

import copy
import importlib
import inspect
import os
from typing import List, Optional, Union

import numpy as np
import torch
import trimesh
import yaml
from PIL import Image
from diffusers.utils.torch_utils import randn_tensor
from diffusers.utils.import_utils import is_accelerate_version, is_accelerate_available
from tqdm import tqdm

from .models.autoencoders import ShapeVAE
from .models.autoencoders import SurfaceExtractors
from .utils import logger, synchronize_timer, smart_load_model

from moudle.mesh_aligh_fun import align_meshes  # 进行registration
from .surface_loaders import SharpEdgeSurfaceLoader
from hy3dshape.schedulers import UniInvEulerScheduler

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
def export_to_trimesh(mesh_output):
    if isinstance(mesh_output, list):
        outputs = []
        for mesh in mesh_output:
            if mesh is None:
                outputs.append(None)
            else:
                mesh.mesh_f = mesh.mesh_f[:, ::-1]
                mesh_output = trimesh.Trimesh(mesh.mesh_v, mesh.mesh_f)
                outputs.append(mesh_output)
        return outputs
    else:
        mesh_output.mesh_f = mesh_output.mesh_f[:, ::-1]
        mesh_output = trimesh.Trimesh(mesh_output.mesh_v, mesh_output.mesh_f)
        return mesh_output


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config, **kwargs):
    if "target" not in config:
        raise KeyError("Expected key `target` to instantiate.")
    cls = get_obj_from_str(config["target"])
    params = config.get("params", dict())
    kwargs.update(params)
    instance = cls(**kwargs)
    return instance


class Hunyuan3DDiTPipeline:
    model_cpu_offload_seq = "conditioner->model->vae"
    _exclude_from_cpu_offload = []

    @classmethod
    @synchronize_timer('Hunyuan3DDiTPipeline Model Loading')
    def from_single_file(
        cls,
        ckpt_path,
        config_path,
        device='cuda',
        dtype=torch.float16,
        use_safetensors=None,
        **kwargs,
    ):
        # load config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # load ckpt
        if use_safetensors:
            ckpt_path = ckpt_path.replace('.ckpt', '.safetensors')
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Model file {ckpt_path} not found")
        logger.info(f"Loading model from {ckpt_path}")

        if use_safetensors:
            # parse safetensors
            import safetensors.torch
            safetensors_ckpt = safetensors.torch.load_file(ckpt_path, device='cpu')
            ckpt = {}
            for key, value in safetensors_ckpt.items():
                model_name = key.split('.')[0]
                new_key = key[len(model_name) + 1:]
                if model_name not in ckpt:
                    ckpt[model_name] = {}
                ckpt[model_name][new_key] = value
        else:
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        # load model
        model = instantiate_from_config(config['model'])
        model.load_state_dict(ckpt['model'])
        vae = instantiate_from_config(config['vae'])
        vae.load_state_dict(ckpt['vae'], strict=False)
        conditioner = instantiate_from_config(config['conditioner'])
        if 'conditioner' in ckpt:
            conditioner.load_state_dict(ckpt['conditioner'])
        image_processor = instantiate_from_config(config['image_processor'])
        scheduler = instantiate_from_config(config['scheduler'])

        model_kwargs = dict(
            vae=vae,
            model=model,
            scheduler=scheduler,
            conditioner=conditioner,
            image_processor=image_processor,
            device=device,
            dtype=dtype,
        )
        model_kwargs.update(kwargs)

        return cls(
            **model_kwargs
        )

    @classmethod
    def from_pretrained(
        cls,
        model_path,
        device='cuda',
        dtype=torch.float16,
        use_safetensors=False,
        variant='fp16',
        subfolder='hunyuan3d-dit-v2-1',
        **kwargs,
    ):
        kwargs['from_pretrained_kwargs'] = dict(
            model_path=model_path,
            subfolder=subfolder,
            use_safetensors=use_safetensors,
            variant=variant,
            dtype=dtype,
            device=device,
        )
        config_path, ckpt_path = smart_load_model(
            model_path,
            subfolder=subfolder,
            use_safetensors=use_safetensors,
            variant=variant
        )
        return cls.from_single_file(
            ckpt_path,
            config_path,
            device=device,
            dtype=dtype,
            use_safetensors=use_safetensors,
            **kwargs
        )

    def __init__(
        self,
        vae,
        model,
        scheduler,
        conditioner,
        image_processor,
        device='cuda',
        dtype=torch.float16,
        **kwargs
    ):
        self.vae = vae
        self.model = model
        self.scheduler = scheduler
        self.conditioner = conditioner
        self.image_processor = image_processor
        self.kwargs = kwargs
        self.to(device, dtype)

    def compile(self):
        self.vae = torch.compile(self.vae)
        self.model = torch.compile(self.model)
        self.conditioner = torch.compile(self.conditioner)

    def enable_flashvdm(
        self,
        enabled: bool = True,
        adaptive_kv_selection=True,
        topk_mode='mean',
        mc_algo='mc',
        replace_vae=True,
    ):
        if enabled:
            model_path = self.kwargs['from_pretrained_kwargs']['model_path']
            turbo_vae_mapping = {
                'Hunyuan3D-2': ('tencent/Hunyuan3D-2', 'hunyuan3d-vae-v2-0-turbo'),
                'Hunyuan3D-2mv': ('tencent/Hunyuan3D-2', 'hunyuan3d-vae-v2-0-turbo'),
                'Hunyuan3D-2mini': ('tencent/Hunyuan3D-2mini', 'hunyuan3d-vae-v2-mini-turbo'),
            }
            model_name = model_path.split('/')[-1]
            if replace_vae and model_name in turbo_vae_mapping:
                model_path, subfolder = turbo_vae_mapping[model_name]
                self.vae = ShapeVAE.from_pretrained(
                    model_path, subfolder=subfolder,
                    use_safetensors=self.kwargs['from_pretrained_kwargs']['use_safetensors'],
                    device=self.device,
                )
            self.vae.enable_flashvdm_decoder(
                enabled=enabled,
                adaptive_kv_selection=adaptive_kv_selection,
                topk_mode=topk_mode,
                mc_algo=mc_algo
            )
        else:
            model_path = self.kwargs['from_pretrained_kwargs']['model_path']
            vae_mapping = {
                'Hunyuan3D-2': ('tencent/Hunyuan3D-2', 'hunyuan3d-vae-v2-0'),
                'Hunyuan3D-2mv': ('tencent/Hunyuan3D-2', 'hunyuan3d-vae-v2-0'),
                'Hunyuan3D-2mini': ('tencent/Hunyuan3D-2mini', 'hunyuan3d-vae-v2-mini'),
            }
            model_name = model_path.split('/')[-1]
            if model_name in vae_mapping:
                model_path, subfolder = vae_mapping[model_name]
                self.vae = ShapeVAE.from_pretrained(model_path, subfolder=subfolder)
            self.vae.enable_flashvdm_decoder(enabled=False)

    def to(self, device=None, dtype=None):
        if dtype is not None:
            self.dtype = dtype
            self.vae.to(dtype=dtype)
            self.model.to(dtype=dtype)
            self.conditioner.to(dtype=dtype)
        if device is not None:
            self.device = torch.device(device)
            self.vae.to(device)
            self.model.to(device)
            self.conditioner.to(device)

    @property
    def _execution_device(self):
        r"""
        Returns the device on which the pipeline's models will be executed. After calling
        [`~DiffusionPipeline.enable_sequential_cpu_offload`] the execution device can only be inferred from
        Accelerate's module hooks.
        """
        for name, model in self.components.items():
            if not isinstance(model, torch.nn.Module) or name in self._exclude_from_cpu_offload:
                continue

            if not hasattr(model, "_hf_hook"):
                return self.device
            for module in model.modules():
                if (
                    hasattr(module, "_hf_hook")
                    and hasattr(module._hf_hook, "execution_device")
                    and module._hf_hook.execution_device is not None
                ):
                    return torch.device(module._hf_hook.execution_device)
        return self.device

    def enable_model_cpu_offload(self, gpu_id: Optional[int] = None, device: Union[torch.device, str] = "cuda"):
        r"""
        Offloads all models to CPU using accelerate, reducing memory usage with a low impact on performance. Compared
        to `enable_sequential_cpu_offload`, this method moves one whole model at a time to the GPU when its `forward`
        method is called, and the model remains in GPU until the next model runs. Memory savings are lower than with
        `enable_sequential_cpu_offload`, but performance is much better due to the iterative execution of the `unet`.

        Arguments:
            gpu_id (`int`, *optional*):
                The ID of the accelerator that shall be used in inference. If not specified, it will default to 0.
            device (`torch.Device` or `str`, *optional*, defaults to "cuda"):
                The PyTorch device type of the accelerator that shall be used in inference. If not specified, it will
                default to "cuda".
        """
        if self.model_cpu_offload_seq is None:
            raise ValueError(
                "Model CPU offload cannot be enabled because no `model_cpu_offload_seq` class attribute is set."
            )

        if is_accelerate_available() and is_accelerate_version(">=", "0.17.0.dev0"):
            from accelerate import cpu_offload_with_hook
        else:
            raise ImportError("`enable_model_cpu_offload` requires `accelerate v0.17.0` or higher.")

        torch_device = torch.device(device)
        device_index = torch_device.index

        if gpu_id is not None and device_index is not None:
            raise ValueError(
                f"You have passed both `gpu_id`={gpu_id} and an index as part of the passed device `device`={device}"
                f"Cannot pass both. Please make sure to either not define `gpu_id` or not pass the index as part of "
                f"the device: `device`={torch_device.type}"
            )

        # _offload_gpu_id should be set to passed gpu_id (or id in passed `device`)
        # or default to previously set id or default to 0
        self._offload_gpu_id = gpu_id or torch_device.index or getattr(self, "_offload_gpu_id", 0)

        device_type = torch_device.type
        device = torch.device(f"{device_type}:{self._offload_gpu_id}")

        if self.device.type != "cpu":
            self.to("cpu")
            device_mod = getattr(torch, self.device.type, None)
            if hasattr(device_mod, "empty_cache") and device_mod.is_available():
                device_mod.empty_cache()  
                # otherwise we don't see the memory savings (but they probably exist)

        all_model_components = {k: v for k, v in self.components.items() if isinstance(v, torch.nn.Module)}

        self._all_hooks = []
        hook = None
        for model_str in self.model_cpu_offload_seq.split("->"):
            model = all_model_components.pop(model_str, None)
            if not isinstance(model, torch.nn.Module):
                continue

            _, hook = cpu_offload_with_hook(model, device, prev_module_hook=hook)
            self._all_hooks.append(hook)

        # CPU offload models that are not in the seq chain unless they are explicitly excluded
        # these models will stay on CPU until maybe_free_model_hooks is called
        # some models cannot be in the seq chain because they are iteratively called, 
        # such as controlnet
        for name, model in all_model_components.items():
            if not isinstance(model, torch.nn.Module):
                continue

            if name in self._exclude_from_cpu_offload:
                model.to(device)
            else:
                _, hook = cpu_offload_with_hook(model, device)
                self._all_hooks.append(hook)

    def maybe_free_model_hooks(self):
        r"""
        Function that offloads all components, removes all model hooks that were added when using
        `enable_model_cpu_offload` and then applies them again. In case the model has not been offloaded this function
        is a no-op. Make sure to add this function to the end of the `__call__` function of your pipeline so that it
        functions correctly when applying enable_model_cpu_offload.
        """
        if not hasattr(self, "_all_hooks") or len(self._all_hooks) == 0:
            # `enable_model_cpu_offload` has not be called, so silently do nothing
            return

        for hook in self._all_hooks:
            # offload model and remove hook from model
            hook.offload()
            hook.remove()

        # make sure the model is in the same state as before calling it
        self.enable_model_cpu_offload()

    @synchronize_timer('Encode cond')
    def encode_cond(self, image, additional_cond_inputs, do_classifier_free_guidance, dual_guidance):
        bsz = image.shape[0]
        cond = self.conditioner(image=image, **additional_cond_inputs)

        if do_classifier_free_guidance:
            un_cond = self.conditioner.unconditional_embedding(bsz, **additional_cond_inputs)

            if dual_guidance:
                un_cond_drop_main = copy.deepcopy(un_cond)
                un_cond_drop_main['additional'] = cond['additional']

                def cat_recursive(a, b, c):
                    if isinstance(a, torch.Tensor):
                        return torch.cat([a, b, c], dim=0).to(self.dtype)
                    out = {}
                    for k in a.keys():
                        out[k] = cat_recursive(a[k], b[k], c[k])
                    return out

                cond = cat_recursive(cond, un_cond_drop_main, un_cond)
            else:
                def cat_recursive(a, b):
                    if isinstance(a, torch.Tensor):
                        return torch.cat([a, b], dim=0).to(self.dtype)
                    out = {}
                    for k in a.keys():
                        out[k] = cat_recursive(a[k], b[k])
                    return out

                cond = cat_recursive(cond, un_cond)
        return cond

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def prepare_latents(self, batch_size, dtype, device, generator, latents=None):
        shape = (batch_size, *self.vae.latent_shape)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * getattr(self.scheduler, 'init_noise_sigma', 1.0)
        return latents

    def prepare_image(self, image, mask=None) -> dict:
        if isinstance(image, torch.Tensor) and isinstance(mask, torch.Tensor):
            outputs = {
                'image': image,
                'mask': mask
            }
            return outputs
            
        if isinstance(image, str) and not os.path.exists(image):
            raise FileNotFoundError(f"Couldn't find image at path {image}")

        if not isinstance(image, list):
            image = [image]

        outputs = []
        for img in image:
            output = self.image_processor(img)
            outputs.append(output)

        cond_input = {k: [] for k in outputs[0].keys()}
        for output in outputs:
            for key, value in output.items():
                cond_input[key].append(value)
        for key, value in cond_input.items():
            if isinstance(value[0], torch.Tensor):
                cond_input[key] = torch.cat(value, dim=0)

        return cond_input

    def get_guidance_scale_embedding(self, w, embedding_dim=512, dtype=torch.float32):
        """
        See https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298

        Args:
            timesteps (`torch.Tensor`):
                generate embedding vectors at these timesteps
            embedding_dim (`int`, *optional*, defaults to 512):
                dimension of the embeddings to generate
            dtype:
                data type of the generated embeddings

        Returns:
            `torch.FloatTensor`: Embedding vectors with shape `(len(timesteps), embedding_dim)`
        """
        assert len(w.shape) == 1
        w = w * 1000.0

        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
        emb = w.to(dtype)[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1))
        assert emb.shape == (w.shape[0], embedding_dim)
        return emb

    def set_surface_extractor(self, mc_algo):
        if mc_algo is None:
            return
        logger.info('The parameters `mc_algo` is deprecated, and will be removed in future versions.\n'
                    'Please use: \n'
                    'from hy3dshape.models.autoencoders import SurfaceExtractors\n'
                    'pipeline.vae.surface_extractor = SurfaceExtractors[mc_algo]() instead\n')
        if mc_algo not in SurfaceExtractors.keys():
            raise ValueError(f"Unknown mc_algo {mc_algo}")
        self.vae.surface_extractor = SurfaceExtractors[mc_algo]()

    @torch.no_grad()
    def __call__(
        self,
        image: Union[str, List[str], Image.Image] = None,
        num_inference_steps: int = 20,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        eta: float = 0.0,
        guidance_scale: float = 7.5,
        dual_guidance_scale: float = 10.5,
        dual_guidance: bool = True,
        generator=None,
        box_v=1.01,
        octree_resolution=256, #384
        mc_level=-1 / 512,
        num_chunks=8000,
        mc_algo=None,
        output_type: Optional[str] = "trimesh",
        enable_pbar=True,
        **kwargs,
    ) -> List[List[trimesh.Trimesh]]:
        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        self.set_surface_extractor(mc_algo)

        device = self.device
        dtype = self.dtype
        do_classifier_free_guidance = guidance_scale >= 0 and \
                                      getattr(self.model, 'guidance_cond_proj_dim', None) is None
        dual_guidance = dual_guidance_scale >= 0 and dual_guidance

        if isinstance(image, torch.Tensor):
            pass
        else:
            cond_inputs = self.prepare_image(image)
            image = cond_inputs.pop('image')
        
        cond = self.encode_cond(
            image=image,
            additional_cond_inputs=cond_inputs,
            do_classifier_free_guidance=do_classifier_free_guidance,
            dual_guidance=False,
        )
        batch_size = image.shape[0]

        t_dtype = torch.long
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas)

        latents = self.prepare_latents(batch_size, dtype, device, generator)
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        guidance_cond = None
        if getattr(self.model, 'guidance_cond_proj_dim', None) is not None:
            logger.info('Using lcm guidance scale')
            guidance_scale_tensor = torch.tensor(guidance_scale - 1).repeat(batch_size)
            guidance_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.model.guidance_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)
        with synchronize_timer('Diffusion Sampling'):
            for i, t in enumerate(tqdm(timesteps, disable=not enable_pbar, desc="Diffusion Sampling:", leave=False)):
                # expand the latents if we are doing classifier free guidance
                if do_classifier_free_guidance:
                    latent_model_input = torch.cat([latents] * (3 if dual_guidance else 2))
                else:
                    latent_model_input = latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                timestep_tensor = torch.tensor([t], dtype=t_dtype, device=device)
                timestep_tensor = timestep_tensor.expand(latent_model_input.shape[0])
                noise_pred = self.model(latent_model_input, timestep_tensor, cond, guidance_cond=guidance_cond)

                # no drop, drop clip, all drop
                if do_classifier_free_guidance:
                    if dual_guidance:
                        noise_pred_clip, noise_pred_dino, noise_pred_uncond = noise_pred.chunk(3)
                        noise_pred = (
                            noise_pred_uncond
                            + guidance_scale * (noise_pred_clip - noise_pred_dino)
                            + dual_guidance_scale * (noise_pred_dino - noise_pred_uncond)
                        )
                    else:
                        noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                outputs = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs)
                latents = outputs.prev_sample

                if callback is not None and i % callback_steps == 0:
                    step_idx = i // getattr(self.scheduler, "order", 1)
                    callback(step_idx, t, outputs)

        return self._export(
            latents,
            output_type,
            box_v, mc_level, num_chunks, octree_resolution, mc_algo,
        )

    def _export(
        self,
        latents,
        output_type='trimesh',
        box_v=1.01,
        mc_level=0.0,
        num_chunks=20000,
        octree_resolution=256,
        mc_algo='mc',
        enable_pbar=True
    ):
        if not output_type == "latent":
            latents = 1. / self.vae.scale_factor * latents
            latents = self.vae(latents)
            outputs = self.vae.latents2mesh(
                latents,
                bounds=box_v,
                mc_level=mc_level,
                num_chunks=num_chunks,
                octree_resolution=octree_resolution,
                mc_algo=mc_algo,
                enable_pbar=enable_pbar,
            )
        else:
            outputs = latents

        if output_type == 'trimesh':
            outputs = export_to_trimesh(outputs)

        return outputs


class Hunyuan3DDiTFlowMatchingPipeline_ori(Hunyuan3DDiTPipeline):

    @torch.inference_mode()
    def __call__(
        self,
        image: Union[str, List[str], Image.Image, dict, List[dict], torch.Tensor] = None,
        num_inference_steps: int = 20,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        eta: float = 0.0,
        guidance_scale: float = 5.0,
        generator=None,
        box_v=1.01,
        octree_resolution=256,
        mc_level=0.0,
        mc_algo=None,
        num_chunks=8000,
        output_type: Optional[str] = "trimesh",
        enable_pbar=True,
        mask = None,
        **kwargs,
    ) -> List[List[trimesh.Trimesh]]:
        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        self.set_surface_extractor(mc_algo)

        device = self.device
        dtype = self.dtype
        do_classifier_free_guidance = guidance_scale >= 0 and not (
            hasattr(self.model, 'guidance_embed') and
            self.model.guidance_embed is True
        )

        # print('image', type(image), 'mask', type(mask))
        cond_inputs = self.prepare_image(image, mask)
        image = cond_inputs.pop('image')
        cond = self.encode_cond(
            image=image,
            additional_cond_inputs=cond_inputs,
            do_classifier_free_guidance=do_classifier_free_guidance,
            dual_guidance=False,
        )

        batch_size = image.shape[0]

        # 5. Prepare timesteps
        # NOTE: this is slightly different from common usage, we start from 0.
        sigmas = np.linspace(0, 1, num_inference_steps) if sigmas is None else sigmas
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
        )
        latents = self.prepare_latents(batch_size, dtype, device, generator)

        guidance = None
        if hasattr(self.model, 'guidance_embed') and \
            self.model.guidance_embed is True:
            guidance = torch.tensor([guidance_scale] * batch_size, device=device, dtype=dtype)
            # logger.info(f'Using guidance embed with scale {guidance_scale}')

        with synchronize_timer('Diffusion Sampling'):
            for i, t in enumerate(tqdm(timesteps, disable=not enable_pbar, desc="Diffusion Sampling:")):
                # expand the latents if we are doing classifier free guidance
                if do_classifier_free_guidance:
                    latent_model_input = torch.cat([latents] * 2)
                else:
                    latent_model_input = latents

                # NOTE: we assume model get timesteps ranged from 0 to 1
                timestep = t.expand(latent_model_input.shape[0]).to(latents.dtype)
                timestep = timestep / self.scheduler.config.num_train_timesteps
                noise_pred = self.model(latent_model_input, timestep, cond, guidance=guidance)

                if do_classifier_free_guidance:
                    noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                outputs = self.scheduler.step(noise_pred, t, latents)
                latents = outputs.prev_sample

                if callback is not None and i % callback_steps == 0:
                    step_idx = i // getattr(self.scheduler, "order", 1)
                    callback(step_idx, t, outputs)

        return self._export(
            latents,
            output_type,
            box_v, mc_level, num_chunks, octree_resolution, mc_algo,
            enable_pbar=enable_pbar,
        )
    

class Hunyuan3DDiTFlowMatchingPipeline(Hunyuan3DDiTPipeline):
    @torch.inference_mode()
    def __call__(
        self,
        ref: Union[str, List[str], Image.Image, dict, List[dict], torch.Tensor] = None,
        image: Union[str, List[str], Image.Image, dict, List[dict], torch.Tensor] = None,
        hand_image: Union[str, List[str], Image.Image, dict, List[dict], torch.Tensor] = None,
        object_image: Union[str, List[str], Image.Image, dict, List[dict], torch.Tensor] = None,
        mesh_path: Optional[Union[str, List[str]]] = None,
        moge_path: Optional[Union[str, List[str]]] = None,
        moge_hand_path: Optional[Union[str, List[str]]] = None,
        num_inference_steps: int = 20,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        eta: float = 0.0,
        guidance_scale: float = 5.0,
        generator=None,
        box_v=1.01,
        octree_resolution=256,  # 384
        mc_level=0.0,
        mc_algo=None,
        num_chunks=8000,
        output_type: Optional[str] = "trimesh",
        enable_pbar=True,
        mask=None,
        do_inversion_stage: bool = False,   # <--- ✅ 新增开关
        **kwargs,
    ) -> List[List[trimesh.Trimesh]]:
        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        self.set_surface_extractor(mc_algo)

        device = self.device
        dtype = self.dtype

        do_classifier_free_guidance = guidance_scale >= 0 and not (
            hasattr(self.model, 'guidance_embed') and
            self.model.guidance_embed is True
        )

        ## 1. Prepare cond inputs
        images = [image]
        has_hand = hand_image is not None
        has_object = object_image is not None
        # has_hand = False
        # has_object = False
        if has_hand:
            images.append(hand_image)
        if has_object:
            images.append(object_image)

        cond_inputs = self.prepare_image(images, mask)  # 这里有个问题是，这个prepare会把img都中心化，那么位置对应不上
        cond_ref = self.prepare_image(ref, mask)

        ############ DEBUG 保存 cond_inputs ############
        self.visualize_cond_inputs(cond_inputs, save_dir="cond_inputs_vis")
        self.visualize_cond_inputs(cond_ref, save_dir="cond_ref_vis")
        ############ DEBUG end ############

        image = cond_inputs.pop('image')
        cond = self.encode_cond(
            image=image,
            additional_cond_inputs=cond_inputs,
            do_classifier_free_guidance=do_classifier_free_guidance,
            dual_guidance=False,
        )

        ref_image = cond_ref.pop('image')
        cond_ref = self.encode_cond(
            image=ref_image,
            additional_cond_inputs=cond_ref,
            do_classifier_free_guidance=do_classifier_free_guidance,
            dual_guidance=False,
        )

        cond_hoi = cond  # 默认只有hoi cond

        if has_hand:
            num_images = len(images)
            cfg_offset = num_images if do_classifier_free_guidance else 0
            cond_hoi = copy.deepcopy(cond)
            cond_hoi['main'] = cond_hoi['main'][[0, cfg_offset], ...]  # only keep image cond (corrected comment)
            cond_hand = copy.deepcopy(cond)
            cond_hand['main'] = cond_hand['main'][[1, 1 + cfg_offset], ...]  # only keep hand cond (corrected comment)
            if has_object:
                cond_object = copy.deepcopy(cond)
                cond_object['main'] = cond_object['main'][[2, 2 + cfg_offset], ...]  # only keep object cond
        ########

        batch_size = 1  # 固定成1个batch进行

        # 5. Prepare timesteps
        # NOTE: this is slightly different from common usage, we start from 0.
        sigmas = np.linspace(0, 1, num_inference_steps) if sigmas is None else sigmas

        phase1_scheduler = copy.deepcopy(self.scheduler)
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
        )

        timesteps_phase1, _ = retrieve_timesteps(
            phase1_scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
        )

        latents = self.prepare_latents(batch_size, dtype, device, generator)

        # guidance 一直是None
        guidance = None
        if hasattr(self.model, 'guidance_embed') and self.model.guidance_embed is True:
            guidance = torch.tensor([guidance_scale] * batch_size, device=device, dtype=dtype)

        with synchronize_timer('Diffusion Sampling'):
            if do_inversion_stage:   # <--- ✅ 控制是否执行第一阶段 + inversion
                # ---------- 第一次 sampling ----------
                pbar = tqdm(timesteps_phase1, disable=not enable_pbar, desc="(Phase 1) Partial Sampling + Inversion:")
                for i, t in enumerate(pbar):
                    if do_classifier_free_guidance:
                        latent_model_input = torch.cat([latents] * 2)
                    else:
                        latent_model_input = latents

                    timestep = t.expand(latent_model_input.shape[0]).to(latents.dtype)
                    timestep = timestep / phase1_scheduler.config.num_train_timesteps
                    noise_pred = self.model(latent_model_input, timestep, cond_ref, guidance=guidance)  # 我们需要registration, 所以用hoi cond. 如果对齐hand 位置，会发现生成+decoding出来的mesh, 被拉伸了

                    if do_classifier_free_guidance:
                        noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

                    outputs = phase1_scheduler.step(noise_pred, t, latents)
                    latents = outputs.prev_sample

                    if i == 8:
                        # ---------- 导出 mesh ----------
                        mesh_i = self._export(
                            outputs.pred_original_sample,
                            box_v=box_v,
                            mc_level=mc_level,
                            num_chunks=num_chunks,
                            octree_resolution=octree_resolution,
                            mc_algo=mc_algo,
                            enable_pbar=enable_pbar,
                        )  # 这里导出的是 x_0 的 mesh, 会进行一次的decoding

                        # ---------- 可视化中间 mesh ----------
                        vis_mid_mesh = True
                        if vis_mid_mesh:    
                            dir = "vis_phase1_mid_mesh"
                            if not os.path.exists(dir):
                                os.makedirs(dir, exist_ok=True)
                            import time
                            if isinstance(mesh_i, list):
                                for midx, m in enumerate(mesh_i):
                                    m.export(f"{dir}/check_step10_{midx}_{time.time()}.glb")
                            else:
                                mesh_i.export(f"{dir}/check_step10_{time.time()}.glb")

                        pbar.close()  # <--- 关闭 tqdm
                        print(f"Inversion Stage: Start registration + inversion...")

                        # ---------- registration ----------
                        Th, To = self.registration(
                            hunyuan_mesh=mesh_i[0] if isinstance(mesh_i, list) else mesh_i,
                            hamer_mesh=mesh_path,
                            moge_pointmap=moge_path,
                            moge_hand_pointmap=moge_hand_path
                        )

                        # ---------- inversion ----------
                        inversion = True if mesh_path is not None else False
                        latents = self.inversion(
                            mesh_path=mesh_path,
                            Th=Th,
                            To=To,
                            device=device,
                            batch_size=batch_size,
                            inversion=inversion,
                            box_v=box_v,
                            octree_resolution=octree_resolution,
                            mc_level=mc_level,
                            num_chunks=num_chunks,
                            mc_algo=mc_algo,
                            enable_pbar=enable_pbar,
                            cond=cond_hand,  # 是对hamer的hand mesh做inversion, 所以inversion过程只用cond hand
                            num_inference_steps=num_inference_steps,
                            timesteps=timesteps,
                            do_classifier_free_guidance=do_classifier_free_guidance,
                            guidance_scale=guidance_scale,
                            generator=generator,
                        )
                        break  # 结束第一次采样循环
            del outputs, mesh_i
            torch.cuda.empty_cache()
            # self.scheduler._step_index = None  # reset step index for Phase 2

            # ---------- 第二次 sampling ----------
            double_branch = False
            if double_branch:
                # latents = torch.cat([latents] * 2, dim=0)
                # cond = {
                #     'main': torch.cat([cond_hoi['main'], cond_hand['main']], dim=0)
                # }

                latents = torch.cat([latents] * 3, dim=0)
                cond = cond
            else:
                cond = cond_ref

            for i, t in enumerate(tqdm(timesteps, disable=not enable_pbar, desc="(Phase 2) Full Sampling:")):
                if do_classifier_free_guidance:
                    latent_model_input = torch.cat([latents] * 2)
                else:
                    latent_model_input = latents

                timestep = t.expand(latent_model_input.shape[0]).to(latents.dtype)
                timestep = timestep / self.scheduler.config.num_train_timesteps
                noise_pred = self.model(latent_model_input, timestep, cond, guidance=guidance)  # NOTE: 这里的cond，记得切换

                if do_classifier_free_guidance:
                    noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

                outputs = self.scheduler.step(noise_pred, t, latents)
                latents = outputs.prev_sample

        return self._export(
            latents,
            output_type,
            box_v, mc_level, num_chunks, octree_resolution, mc_algo,
            enable_pbar=enable_pbar,
        )

    def prepare_image(self, image, mask=None) -> dict:
        if isinstance(image, torch.Tensor) and isinstance(mask, torch.Tensor):
            outputs = {
                'image': image,
                'mask': mask
            }
            return outputs
            
        if isinstance(image, str) and not os.path.exists(image):
            raise FileNotFoundError(f"Couldn't find image at path {image}")

        if not isinstance(image, list):
            image = [image]

        # 这里对于同一张图片，进行分割后不同图片recenter结果不对齐
        outputs = []
        for img in image:
            output = self.image_processor(img)
            outputs.append(output)

        # 记录
        # outputs = []
        # ref_bbox, ref_scale = None, None
        # for i, img in enumerate(image):
        #     if i == 0:
        #         output, ref_bbox, ref_scale = self.image_processor(img)
        #     else:
        #         output, _, _ = self.image_processor(img, ref_bbox=ref_bbox, ref_scale=ref_scale)
        #     outputs.append(output)


        cond_input = {k: [] for k in outputs[0].keys()}
        for output in outputs:
            for key, value in output.items():
                cond_input[key].append(value)
        for key, value in cond_input.items():
            if isinstance(value[0], torch.Tensor):
                cond_input[key] = torch.cat(value, dim=0)

        return cond_input

    # For multi-step inversion loop (call in __call__)
    def inversion_loop(
        self,
        latents: torch.FloatTensor,  # Zs_1 from VAE encode
        cond: torch.FloatTensor,  # hand DINO cond
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
        """
        Full inversion loop: from x_0 to x_T over num_steps.
        """
        cond_hand = copy.deepcopy(cond)

        if do_classifier_free_guidance:
            # cond_hand = cond   # cond, uncond
            cond_hand['main'] = torch.cat([cond['main'][[-1], :, :], cond['main'][[-1], :, :]], dim=0)     # uncond, uncond
        else:
            cond_hand = [cond['main'][[-1], :, :]]  # uncond only
            # cond_hand = [cond['main'][[0], :, :]]  # cond only

        inv_scheduler = UniInvEulerScheduler(num_train_timesteps=1000)
        sigmas = np.linspace(0, 1, inversion_steps) if sigmas is None else sigmas  # Hunyuan-3D default
        timesteps, inversion_steps = retrieve_timesteps(
            inv_scheduler,
            inversion_steps,
            device,
            sigmas=sigmas,
        )
        
        with synchronize_timer('Diffusion Inversion'):
            for i, t in enumerate(tqdm(timesteps, disable=not enable_pbar, desc="Diffusion Inversion:")):
                # expand the latents if we are doing classifier free guidance
                if do_classifier_free_guidance:
                    latent_model_input = torch.cat([latents] * 2)
                else:
                    latent_model_input = latents

                # NOTE: we assume model get timesteps ranged from 0 to 1
                timestep = t.expand(latent_model_input.shape[0]).to(
                    latents.dtype) / inv_scheduler.config.num_train_timesteps
                noise_pred = self.model(latent_model_input, timestep, cond_hand, guidance=guidance)

                if do_classifier_free_guidance:
                    noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                outputs = inv_scheduler.step(noise_pred, t, latents)
                latents = outputs.prev_sample
        
        return latents  # noisy x_T

    def inversion(self,
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
        """
        Perform inversion using registered hand mesh. 
        and using the hand inversion latents as initialization for Phase 2 full sampling.
        """
        assert Th.shape == (4, 4) and To.shape == (4, 4), "Th/To should be 4x4 transformation matrices for registration."

        # inversion = False
        if inversion:
            surface = loader(mesh_path, Th, To).to(self.device, dtype=self.dtype)  # 一定不要读取时候中心化
            latents = self.vae.encode(surface)
            latents = self.vae.scale_factor * latents
            
            # ---------- test hamer encode and decode result ----------
            test_encode_decode = True
            if test_encode_decode:
                import time
                latents_rec = 1. / self.vae.scale_factor * latents.clone().detach()
                latents_rec = self.vae(latents_rec)
                outputs = self.vae.latents2mesh(
                    latents_rec,
                    bounds=box_v,
                    mc_level=mc_level,
                    num_chunks=num_chunks,
                    octree_resolution=octree_resolution,
                    mc_algo=mc_algo,
                    enable_pbar=enable_pbar,
                )
                mesh = export_to_trimesh(outputs)
                if isinstance(mesh, list):
                    for midx, m in enumerate(mesh):
                        m.export(f"check_hand_{midx}_{time.time()}.glb")
                else:
                    mesh.export(f"check_hand_{time.time()}.glb")
                
                del latents_rec, outputs, mesh
                torch.cuda.empty_cache()
            

            # latents = latents * 1.15 # latent_nudging_scalar, 能完美重建，但是会影响full 分支
            # latents = latents * 1.05
            # ---------- inversion loop ----------
            latents = self.inversion_loop(
                latents,
                cond,
                device=device,
                inversion_steps=num_inference_steps,
                do_classifier_free_guidance=do_classifier_free_guidance,
                guidance_scale=1.0,
                timesteps=timesteps,
                generator=generator,
            )
            # latents = 1. / 1.15 * latents

            if latents.shape[0] == 1:
                latentss = latents
            # else:
            #     if latents.shape[0] != batch_size:
            #         if latents.shape[0] == 1:
            #             latentss = latents.expand(batch_size, *latents.shape[1:])
            #         else:
            #             # Repeat if batch size is incompatible
            #             latentss = latents.repeat((batch_size // latents.shape[0] + 1), 1, 1)[:batch_size]
        else:
            if enable_pbar:
                print("[inversion] Skipping inversion; generating random latents.")
            latentss = self.prepare_latents(batch_size, self.dtype, device, generator)
        
        return latentss
    
    def registration(self, hunyuan_mesh, hamer_mesh, moge_pointmap, moge_hand_pointmap):

        # Th, from hamer hand mesh to Moge pointmap
        Th = align_meshes(
            source_mesh_path=hamer_mesh,
            target_mesh_path=moge_hand_pointmap,
            skip_coarse=True,
            transformed_mesh_path="hand_registered.glb"
        )

        # To, from hunyuan generated mesh to Moge pointmap
        To = align_meshes(
            source_mesh_path=None,
            source_mesh=hunyuan_mesh,
            target_mesh_path=moge_pointmap,
            skip_coarse=True,
            transformed_mesh_path="hunyuan_registered.glb"
        )

        return Th, To
    

    def visualize_cond_inputs(self, cond_inputs, save_dir="cond_inputs_vis"):
        """
        可视化 cond_inputs 中的所有 image 和 mask, 并保存为 PNG 文件。

        参数:
            cond_inputs (dict): 包含 'image' 和 'mask' 的 tensor
            save_dir (str): 保存图片的目录
        """
        
        # import os
        # import torch
        # import numpy as np
        # from PIL import Image
        os.makedirs(save_dir, exist_ok=True)

        def tensor_to_numpy(tensor):
            """
            Convert tensor [B, C, H, W] or [C, H, W] to numpy [B, H, W, C]
            """
            tensor = tensor.cpu() if tensor.is_cuda else tensor
            if tensor.dim() == 4:
                arr = tensor.permute(0, 2, 3, 1).numpy()  # [B, H, W, C]
            elif tensor.dim() == 3:
                arr = tensor.permute(1, 2, 0).numpy()
                arr = arr[np.newaxis, ...]  # add batch dim
            else:
                raise ValueError(f"Unsupported tensor shape {tensor.shape}")
            return arr

        # 提取 tensor 并转换为 numpy
        image_np = tensor_to_numpy(cond_inputs['image'])
        mask_np = tensor_to_numpy(cond_inputs['mask'])

        # 遍历 batch 保存每张图片
        for idx in range(image_np.shape[0]):
            img = image_np[idx]
            msk = mask_np[idx]

            # ===== 处理 image =====
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)

            # ===== 处理 mask (值域 [-1, 1]) =====
            msk = ((msk + 1.0) / 2.0 * 255).astype(np.uint8)
            if msk.ndim == 3 and msk.shape[-1] == 1:
                msk = msk[:, :, 0]

            # 保存图片
            Image.fromarray(img).save(os.path.join(save_dir, f"image_{idx}.png"))
            Image.fromarray(msk).save(os.path.join(save_dir, f"mask_{idx}.png"))

        print(f"Saved {image_np.shape[0]} images and masks to '{save_dir}'")