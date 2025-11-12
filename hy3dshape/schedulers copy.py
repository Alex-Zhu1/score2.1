# Copyright 2024 Stability AI, Katherine Crowson and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
import os
import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.utils import BaseOutput, logging

import imageio.v3 as iio
import trimesh
import cv2 as cv
try:
    import nvdiffrast.torch as dr
    NVDIFFRAST_AVAILABLE = True
except ImportError:
    NVDIFFRAST_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class FlowMatchEulerDiscreteSchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's `step` function output.

    Args:
        prev_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample `(x_{t-1})` of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
    """

    prev_sample: torch.FloatTensor

class FlowMatchEulerDiscreteScheduler(SchedulerMixin, ConfigMixin):
    """
    NOTE: this is very similar to diffusers.FlowMatchEulerDiscreteScheduler. Except our timesteps are reversed

    Euler scheduler with optional differentiable 2D guidance.

    This model inherits from [`SchedulerMixin`] and [`ConfigMixin`]. Check the superclass documentation for the generic
    methods the library implements for all schedulers such as loading and saving.

    Args:
        num_train_timesteps (`int`, defaults to 1000):
            The number of diffusion steps to train the model.
        timestep_spacing (`str`, defaults to `"linspace"`):
            The way the timesteps should be scaled. Refer to Table 2 of the [Common Diffusion Noise Schedules and
            Sample Steps are Flawed](https://huggingface.co/papers/2305.08891) for more information.
        shift (`float`, defaults to 1.0):
            The shift value for the timestep schedule.
    """

    _compatibles = []
    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 1.0,
        use_dynamic_shifting=False,
    ):
        timesteps = np.linspace(1, num_train_timesteps, num_train_timesteps, dtype=np.float32).copy()
        timesteps = torch.from_numpy(timesteps).to(dtype=torch.float32)

        sigmas = timesteps / num_train_timesteps
        if not use_dynamic_shifting:
            # when use_dynamic_shifting is True, we apply the timestep shifting on the fly based on the image resolution
            sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)

        self.timesteps = sigmas * num_train_timesteps

        self._step_index = None
        self._begin_index = None

        self.sigmas = sigmas.to("cpu")  # to avoid too much CPU/GPU communication
        self.sigma_min = self.sigmas[-1].item()
        self.sigma_max = self.sigmas[0].item()
        
        # Initialize GL context for 2D guidance (lazy initialization)
        self.glctx = None

    def _init_glctx(self):
        """Lazy initialization of GL context."""
        if self.glctx is None and NVDIFFRAST_AVAILABLE:
            self.glctx = dr.RasterizeGLContext()
        return self.glctx

    @property
    def step_index(self):
        """
        The index counter for current timestep. It will increase 1 after each scheduler step.
        """
        return self._step_index

    @property
    def begin_index(self):
        """
        The index for the first timestep. It should be set from pipeline with `set_begin_index` method.
        """
        return self._begin_index

    # Copied from diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler.set_begin_index
    def set_begin_index(self, begin_index: int = 0):
        """
        Sets the begin index for the scheduler. This function should be run from pipeline before the inference.

        Args:
            begin_index (`int`):
                The begin index for the scheduler.
        """
        self._begin_index = begin_index

    def scale_noise(
        self,
        sample: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        noise: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        """
        Forward process in flow-matching

        Args:
            sample (`torch.FloatTensor`):
                The input sample.
            timestep (`int`, *optional*):
                The current timestep in the diffusion chain.

        Returns:
            `torch.FloatTensor`:
                A scaled input sample.
        """
        # Make sure sigmas and timesteps have the same device and dtype as original_samples
        sigmas = self.sigmas.to(device=sample.device, dtype=sample.dtype)

        if sample.device.type == "mps" and torch.is_floating_point(timestep):
            # mps does not support float64
            schedule_timesteps = self.timesteps.to(sample.device, dtype=torch.float32)
            timestep = timestep.to(sample.device, dtype=torch.float32)
        else:
            schedule_timesteps = self.timesteps.to(sample.device)
            timestep = timestep.to(sample.device)

        # self.begin_index is None when scheduler is used for training, or pipeline does not implement set_begin_index
        if self.begin_index is None:
            step_indices = [self.index_for_timestep(t, schedule_timesteps) for t in timestep]
        elif self.step_index is not None:
            # add_noise is called after first denoising step (for inpainting)
            step_indices = [self.step_index] * timestep.shape[0]
        else:
            # add noise is called before first denoising step to create initial latent(img2img)
            step_indices = [self.begin_index] * timestep.shape[0]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < len(sample.shape):
            sigma = sigma.unsqueeze(-1)

        sample = sigma * noise + (1.0 - sigma) * sample

        return sample

    def _sigma_to_t(self, sigma):
        return sigma * self.config.num_train_timesteps

    def time_shift(self, mu: float, sigma: float, t: torch.Tensor):
        return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)

    def set_timesteps(
        self,
        num_inference_steps: int = None,
        device: Union[str, torch.device] = None,
        sigmas: Optional[List[float]] = None,
        mu: Optional[float] = None,
    ):
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model.
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        """

        if self.config.use_dynamic_shifting and mu is None:
            raise ValueError(" you have a pass a value for `mu` when `use_dynamic_shifting` is set to be `True`")

        if sigmas is None:
            self.num_inference_steps = num_inference_steps
            timesteps = np.linspace(
                self._sigma_to_t(self.sigma_max), self._sigma_to_t(self.sigma_min), num_inference_steps
            )

            sigmas = timesteps / self.config.num_train_timesteps

        if self.config.use_dynamic_shifting:
            sigmas = self.time_shift(mu, 1.0, sigmas)
        else:
            sigmas = self.config.shift * sigmas / (1 + (self.config.shift - 1) * sigmas)

        sigmas = torch.from_numpy(sigmas).to(dtype=torch.float32, device=device)
        timesteps = sigmas * self.config.num_train_timesteps

        self.timesteps = timesteps.to(device=device)
        self.sigmas = torch.cat([sigmas, torch.ones(1, device=sigmas.device)])

        self._step_index = None
        self._begin_index = None

    def index_for_timestep(self, timestep, schedule_timesteps=None):
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps

        indices = (schedule_timesteps == timestep).nonzero()

        # The sigma index that is taken for the **very** first `step`
        # is always the second index (or the last index if there is only 1)
        # This way we can ensure we don't accidentally skip a sigma in
        # case we start in the middle of the denoising schedule (e.g. for image-to-image)
        pos = 1 if len(indices) > 1 else 0

        return indices[pos].item()

    def _init_step_index(self, timestep):
        if self.begin_index is None:
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        sample: torch.FloatTensor,
        s_churn: float = 0.0,
        s_tmin: float = 0.0,
        s_tmax: float = float("inf"),
        s_noise: float = 1.0,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
        enable_guidance_2d: bool = False,
        To: Optional[torch.Tensor] = None,
        _export: Optional[Callable] = None,
        guidance_config: Optional[dict] = None,
    ) -> Union["ConsistencyFlowMatchEulerDiscreteSchedulerOutput", Tuple]:
        """
        Predict the sample from the previous timestep by reversing the SDE.
        
        Args:
            model_output: Direct output from learned diffusion model
            timestep: Current discrete timestep in the diffusion chain
            sample: Current instance of a sample created by the diffusion process
            To: Global transformation matrix (4x4)
            _export: Function to convert latent/depth to mesh
            enable_guidance_2d: Whether to enable 2D guidance optimization
            guidance_config: Dictionary with guidance parameters (lr, num_steps, weights)
            Other args: Standard scheduler parameters
        
        Returns:
            ConsistencyFlowMatchEulerDiscreteSchedulerOutput or tuple with prev_sample
        """
        
        # Validate timestep format
        if isinstance(timestep, (int, torch.IntTensor, torch.LongTensor)):
            raise ValueError(
                "Passing integer indices as timesteps to EulerDiscreteScheduler.step() "
                "is not supported. Pass one of scheduler.timesteps instead."
            )

        if self.step_index is None:
            self._init_step_index(timestep)

        # Get device and dtype info
        device = sample.device
        original_dtype = model_output.dtype
        
        # Upcast to avoid precision issues
        sample = sample.to(torch.float32)

        # Get current and next sigma values
        sigma = self.sigmas[self.step_index]
        sigma_next = self.sigmas[self.step_index + 1]

        # Apply 2D guidance if enabled
        if enable_guidance_2d:
            if not NVDIFFRAST_AVAILABLE:
                raise RuntimeError(
                    "nvdiffrast is required for 2D guidance but not available. "
                    "Install it with: pip install nvdiffrast"
                )
            
            model_output = self._apply_2d_guidance(
                model_output=model_output,
                sample=sample,
                sigma=sigma,
                To=To,
                _export=_export,
                device=device,
                original_dtype=original_dtype,
                config=guidance_config or {}
            )
        
        # Standard Euler update step
        prev_sample = sample + (sigma_next - sigma) * model_output
        prev_sample = prev_sample.to(original_dtype)

        # Predict original sample (x_0)
        pred_original_sample = sample + (1.0 - sigma) * model_output
        pred_original_sample = pred_original_sample.to(original_dtype)

        # Increment step index
        self._step_index += 1

        if not return_dict:
            return (prev_sample,)

        return ConsistencyFlowMatchEulerDiscreteSchedulerOutput(
            prev_sample=prev_sample,
            pred_original_sample=pred_original_sample
        )

    # ============== 2D GUIDANCE METHODS ==============

    def _skew_symmetric(self, v: torch.Tensor) -> torch.Tensor:
        """Create skew-symmetric matrix from 3D vector."""
        return torch.tensor([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ], device=v.device, dtype=v.dtype)

    def axis_angle_to_matrix(self, axis_angle: torch.Tensor) -> torch.Tensor:
        """
        Convert axis-angle representation to rotation matrix (differentiable).
        
        Args:
            axis_angle: (3,) tensor representing rotation as axis * angle
        
        Returns:
            rotation_matrix: (3, 3) orthogonal rotation matrix
        """
        angle = torch.norm(axis_angle)
        
        # Handle small angles with Taylor expansion to avoid division by zero
        if angle < 1e-6:
            # R ≈ I + [axis_angle]_×
            return torch.eye(3, device=axis_angle.device) + self._skew_symmetric(axis_angle)
        
        axis = axis_angle / angle
        K = self._skew_symmetric(axis)
        
        # Rodrigues' formula: R = I + sin(θ)K + (1-cos(θ))K²
        cos_angle = torch.cos(angle)
        sin_angle = torch.sin(angle)
        
        I = torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype)
        R = I + sin_angle * K + (1 - cos_angle) * (K @ K)
        
        return R
    
    def _build_transform_matrix(
        self, 
        scale: torch.Tensor, 
        axis_angle: torch.Tensor, 
        translation: torch.Tensor
    ) -> torch.Tensor:
        """
        Build 4x4 transformation matrix with proper rotation parameterization.
        
        Args:
            scale: (3,) scale factors
            axis_angle: (3,) rotation in axis-angle form
            translation: (3,) translation vector
        """
        device = scale.device
        transform = torch.eye(4, device=device, dtype=torch.float32)
        
        # Convert axis-angle to rotation matrix
        R = self.axis_angle_to_matrix(axis_angle)
        
        # Apply scale then rotation: SR
        S = torch.diag(scale)
        SR = S @ R
        
        transform[:3, :3] = SR
        transform[:3, 3] = translation
        
        return transform

    @torch.enable_grad()
    def _apply_2d_guidance(
        self,
        model_output: torch.FloatTensor,
        sample: torch.FloatTensor,
        sigma: float,
        To: torch.Tensor,
        _export: Callable,
        device: torch.device,
        original_dtype: torch.dtype,
        config: dict,
    ) -> torch.FloatTensor:
        """
        Apply differentiable 2D guidance to refine model output.
        """
        
        # === Freeze VAE & Decoder ===
        pipeline = getattr(_export, "__self__", None)
        if pipeline is not None:
            for m in [pipeline.vae, getattr(pipeline.vae, "geo_decoder", None)]:
                if m is not None:
                    for p in m.parameters():
                        p.requires_grad_(False)
        
        # === Hyperparameters ===
        num_steps = config.get("num_steps", 50)
        lr_velocity = config.get("lr_velocity", 1e-4)
        lr_scale = config.get("lr_scale", 0.01)
        lr_rotation = config.get("lr_rotation", 0.01)
        lr_translation = config.get("lr_translation", 0.01)
        
        weight_norm = config.get("weight_norm", 10.0)
        weight_sil = config.get("weight_sil", 10.0)
        weight_reg_scale = config.get("weight_reg_scale", 1e-3)
        weight_reg_trans = config.get("weight_reg_trans", 1e-3)
        weight_reg_rot = config.get("weight_reg_rot", 1e-4)
        
        fov_x = config.get("fov_x", 41.039776)
        fov_y = config.get("fov_y", 41.039776)
        
        # === Initialize Optimizable Parameters ===
        para_velocity = model_output.clone().detach().requires_grad_(True)
        axis_angle = torch.zeros(3, device=device, dtype=torch.float32, requires_grad=True)
        scale = torch.ones(3, device=device, dtype=torch.float32, requires_grad=True)
        translation = torch.zeros(3, device=device, dtype=torch.float32, requires_grad=True)
        
        optimizer = torch.optim.Adam([
            {'params': [para_velocity], 'lr': lr_velocity},
            {'params': [scale], 'lr': lr_scale},
            {'params': [axis_angle], 'lr': lr_rotation},
            {'params': [translation], 'lr': lr_translation},
        ])
        
        # === Load Reference Images ===
        # ref_dir = config.get("reference_dir", "./references")
        ref_dir = "/home/haiming.zhu/hoi/Hunyuan3D-2.1/hy3dshape/outputs_depth/325_cropped_hoi_1"
        
        ref_normal = cv.imread(f"{ref_dir}/rendered_normal.png", cv.IMREAD_COLOR)
        if ref_normal is None:
            raise FileNotFoundError(f"Could not load {ref_dir}/rendered_normal.png")
        ref_normal = cv.cvtColor(ref_normal, cv.COLOR_BGR2RGB).astype(np.float32) / 255.0
        ref_normal = torch.from_numpy(ref_normal).to(device)
        
        ref_silhouette = cv.imread(f"{ref_dir}/rendered_silhouette.png", cv.IMREAD_GRAYSCALE)
        if ref_silhouette is None:
            raise FileNotFoundError(f"Could not load {ref_dir}/rendered_silhouette.png")
        ref_silhouette = ref_silhouette.astype(np.float32) / 255.0
        ref_silhouette = torch.from_numpy(ref_silhouette).to(device)
        
        # === Optimization Loop ===
        for opt_step in range(num_steps):
            optimizer.zero_grad()
            
            transform = self._build_transform_matrix(scale, axis_angle, translation)
            x1 = sample + (1.0 - sigma) * para_velocity
            
            with torch.cuda.amp.autocast(enabled=False):
                outputs = _export(
                    x1.to(original_dtype),
                    output_type="mesh",
                    requires_grad=True,
                    use_checkpoint=True,
                    enable_pbar=False,
                    mc_algo="dmc"
                )  # 利用dmc导出可微的mesh

            vertices, faces = outputs[0]
            # ---- Sanity Check ----
            if faces is None or len(faces) == 0 or faces.numel() == 0:
                assert False, f"[Warning] Empty mesh at opt_step {opt_step}, skip rendering."

            if faces.dim() == 3 and faces.shape[0] == 1:
                faces = faces.squeeze(0)
            faces = faces.to(torch.int32).contiguous()

            # if True:
            #     with torch.no_grad():
            #         v_vis = vertices.detach().cpu().numpy()
            #         f_vis = faces.detach().cpu().numpy()
            #         # 如果你的渲染器是逆时针为正，且画面全黑，试试反转：
            #         f_vis = np.ascontiguousarray(f_vis)[:, ::-1]
            #         v_vis = v_vis.astype(np.float32)
            #         trimesh.Trimesh(v_vis, f_vis).export(f"./debug/mesh_{opt_step:03d}.glb")
            

            vertices = torch.as_tensor(vertices, device=device, dtype=torch.float32)
            faces = torch.as_tensor(faces, device=device, dtype=torch.int32)
            
            # Apply transformations, 将mesh变到Moge的pointmap相机坐标系下
            vertices_homo = torch.cat([
                vertices, 
                torch.ones(vertices.shape[0], 1, device=device)
            ], dim=1)
            vertices_homo = vertices_homo @ transform.T
            
            if not isinstance(To, torch.Tensor):
                To = torch.as_tensor(To, device=device, dtype=torch.float32)
            vertices_homo = vertices_homo @ To.T
            vertices_transformed = vertices_homo[:, :3] / vertices_homo[:, 3:4]
            
            # Render
            normal_map, alpha_map, silhouette = self.render_maps(
                verts=vertices_transformed,
                faces=faces,
                fov_x=fov_x,
                fov_y=fov_y
            )  # 可视化发现这里渲染得到的normal， silhouette是全黑，需要debug
            
            # === Compute Losses ===
            loss_norm = F.l1_loss(normal_map, ref_normal)
            loss_sil = F.binary_cross_entropy(alpha_map, ref_silhouette, reduction='mean')
            loss_scale = torch.sum((scale - 1.0) ** 2)
            loss_trans = torch.sum(translation ** 2)
            loss_rot = torch.sum(axis_angle ** 2)
            
            total_loss = (
                weight_norm * loss_norm +
                weight_sil * loss_sil +
                weight_reg_scale * loss_scale +
                weight_reg_trans * loss_trans +
                weight_reg_rot * loss_rot
            )
            
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_([para_velocity], max_norm=1.0)
            torch.nn.utils.clip_grad_norm_([scale, axis_angle, translation], max_norm=0.1)
            
            optimizer.step()
            
            if opt_step % 5 == 0:
                print(
                    f"[2D Guidance {opt_step:3d}] "
                    f"L_norm={loss_norm.item():.4f} "
                    f"L_sil={loss_sil.item():.4f} | "
                    f"Total={total_loss.item():.4f}"
                )
            
            if opt_step % 10 == 0 and config.get("save_debug", False):
                import os
                os.makedirs("./debug", exist_ok=True)
                
                normal_np = (normal_map.detach().cpu().numpy() * 255).astype(np.uint8)
                cv.imwrite(
                    f"./debug/opt_step_{opt_step:03d}_normal.png", 
                    cv.cvtColor(normal_np, cv.COLOR_RGB2BGR)
                )
                
                sil_np = (silhouette.detach().cpu().numpy() * 255).astype(np.uint8)
                cv.imwrite(f"./debug/opt_step_{opt_step:03d}_silhouette.png", sil_np)

        return para_velocity.detach()
    
    def render_maps(
        self,
        verts: torch.Tensor,
        faces: torch.Tensor,
        fov_x: float,
        fov_y: float,
        render_res: Tuple[int, int] = (224, 224),
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Render normal map, alpha map, and silhouette."""
        device = verts.device
        
        verts = verts.unsqueeze(0)
        faces = faces.unsqueeze(0)
        
        W, H = render_res
        fx = W / (2 * np.tan(np.deg2rad(fov_x) / 2))
        fy = H / (2 * np.tan(np.deg2rad(fov_y) / 2))
        cx, cy = W / 2, H / 2
        
        projection = torch.tensor([
            [fx, 0, cx, 0],
            [0, fy, cy, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], device=device, dtype=torch.float32)
        
        c2ws = torch.eye(4, device=device, dtype=torch.float32)
        
        normals = self._compute_vertex_normals(verts[0], faces[0]).unsqueeze(0)
        normal_colors = ((normals + 1.0) / 2.0).clamp(0, 1)
        alpha = torch.ones_like(normal_colors[..., :1])
        rgba = torch.cat([normal_colors, alpha], dim=-1)
        
        normal_map, alpha_map = self._renderer(
            verts, faces[0], rgba[0], projection, c2ws, (H, W)
        )   # 渲染
        
        silhouette = (alpha_map > 0.5).float()
        
        return normal_map.clamp(0, 1), alpha_map.clamp(0, 1), silhouette
    
    def _renderer(
        self, 
        verts: torch.Tensor,  # [1, V, 3]，在 MOGE 相机坐标系中
        tri: torch.Tensor,    # [F, 3]
        attr: torch.Tensor,   # [1, V, C]
        projection: torch.Tensor,  # K 矩阵扩展成 4x4
        c2ws: torch.Tensor,        # 相机外参（=I）
        resolution: Tuple[int, int]
    ):
        device = verts.device

        if self.glctx is None:
            self._init_glctx()

        tri = tri.to(torch.int32).contiguous()
        verts = verts.to(torch.float32).contiguous()
        attr = attr.to(torch.float32).contiguous()

        ones = torch.ones(verts.shape[0], verts.shape[1], 1, device=device)
        pos = torch.cat((verts, ones), dim=2)  # [1, V, 4]

        # 这里的 c2ws 是单位矩阵（因为 verts 已经在相机坐标系中）
        view_matrix = torch.linalg.inv(c2ws)
        mat = (projection @ view_matrix).unsqueeze(0)
        pos_clip = (pos @ mat.mT).contiguous()

        rast, _ = dr.rasterize(self.glctx, pos_clip, tri, resolution)
        out, _ = dr.interpolate(attr, rast, tri)
        out = dr.antialias(out, rast, pos_clip, tri)

        img = torch.flip(out[0, :, :, :3], dims=[0])
        alpha = torch.flip(out[0, :, :, 3], dims=[0])
        return img, alpha

    
    def _compute_vertex_normals(
        self, 
        verts: torch.Tensor, 
        faces: torch.Tensor
    ) -> torch.Tensor:
        """Compute smooth vertex normals from face normals."""
        v0 = verts[faces[:, 0]]
        v1 = verts[faces[:, 1]]
        v2 = verts[faces[:, 2]]
        
        face_normals = torch.cross(v1 - v0, v2 - v0, dim=1)
        face_normals = F.normalize(face_normals, dim=1)
        
        vertex_normals = torch.zeros_like(verts)
        vertex_normals.index_add_(0, faces[:, 0], face_normals)
        vertex_normals.index_add_(0, faces[:, 1], face_normals)
        vertex_normals.index_add_(0, faces[:, 2], face_normals)
        
        return F.normalize(vertex_normals, dim=1)

@dataclass
class UniInvEulerSchedulerOutput(BaseOutput):
    prev_sample: torch.FloatTensor

class UniInvEulerScheduler(FlowMatchEulerDiscreteScheduler):
    zero_initial=False
    alpha=1

    def set_hyperparameters(self, zero_initial=False, alpha=1):
        self.zero_initial = zero_initial
        self.alpha = alpha
    
    def set_timesteps(
        self,
        num_inference_steps: int = None,
        device: Union[str, torch.device] = None,
        sigmas: Optional[List[float]] = None,
        mu: Optional[float] = None,
    ):
        if self.config.use_dynamic_shifting and mu is None:
            raise ValueError(" you have a pass a value for `mu` when `use_dynamic_shifting` is set to be `True`")

        if sigmas is None:
            self.num_inference_steps = num_inference_steps
            timesteps = np.linspace(
                self._sigma_to_t(self.sigma_max), self._sigma_to_t(self.sigma_min), num_inference_steps
            )

            sigmas = timesteps / self.config.num_train_timesteps
        else:
            self.num_inference_steps = len(sigmas)

        if self.config.use_dynamic_shifting:
            sigmas = self.time_shift(mu, 1.0, sigmas)
        else:
            sigmas = self.config.shift * sigmas / (1 + (self.config.shift - 1) * sigmas)


        sigmas = torch.from_numpy(sigmas).to(dtype=torch.float32, device=device)
        
        # timesteps
        timesteps = sigmas * self.config.num_train_timesteps
        # timesteps = torch.cat([timesteps, torch.zeros(1).to(sigmas)])
        self.timesteps = timesteps.flip(dims=[0]).to(device=device)

        # sigmas
        sigmas = torch.cat([sigmas, torch.ones(1, device=sigmas.device)])
        self.sigmas = sigmas.flip(dims=[0]).to(device=device)
        
        # empty dt and derivative
        self.sample = None
        
        # zero_initial
        if self.zero_initial:
            self.timesteps = self.timesteps[1: ]
            self.sigmas = self.sigmas[1: ]
            self.sample = 'placeholder'
            self.first_sigma = 0
            
        # alpha, early stop
        if self.alpha < 1:
            inv_steps = math.floor(self.alpha * self.num_inference_steps)
            skip_steps = self.num_inference_steps - inv_steps
            self.timesteps = self.timesteps[: -skip_steps]
            self.sigmas = self.sigmas[: -skip_steps]

        self._step_index = 0
        self._begin_index = 0
        
        
    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        sample: torch.FloatTensor,
        s_churn: float = 0.0,
        s_tmin: float = 0.0,
        s_tmax: float = float("inf"),
        s_noise: float = 1.0,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> Union[UniInvEulerSchedulerOutput, Tuple]:
        
        if (
            isinstance(timestep, int)
            or isinstance(timestep, torch.IntTensor)
            or isinstance(timestep, torch.LongTensor)
        ):
            raise ValueError(
                (
                    "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
                    " `HeunDiscreteScheduler.step()` is not supported. Make sure to pass"
                    " one of the `scheduler.timesteps` as a timestep."
                ),
            )
            
        sample = sample.to(torch.float32)

        if self.sample is None:
            # just for the first step
            sigma = self.sigmas[self.step_index]
            sigma_next = self.sigmas[self.step_index + 1]
            
            derivative = model_output               # v_0 = f(t=0, x_0)
            dt = sigma_next - sigma                 # sigma_{t + \Delta t} - sigma_t

            # store for correction
            self.sample = sample                    # Z_0
            
            prev_sample = sample + derivative * dt
            prev_sample = prev_sample.to(model_output.dtype)
        else:
            sigma = self.sigmas[self.step_index - 1]
            sigma_next = self.sigmas[self.step_index]
            
            if isinstance(self.sample, str):
                # for zero_initial
                sigma = self.first_sigma
                self.sample = sample
                
            derivative = model_output
            dt = sigma_next - sigma

            sample = self.sample

            self.sample = sample + dt * derivative

            if (self.step_index + 1) < len(self.sigmas):
                sigma_next_next = self.sigmas[self.step_index + 1]
                dt_next = sigma_next_next - sigma_next
                
                prev_sample = self.sample + dt_next * derivative
            else:
                # end loop
                prev_sample = self.sample
            prev_sample = prev_sample.to(model_output.dtype)
        ##log
        # print(f"inverse_step: step_index={self.step_index}, timestep={timestep}, sigma={sigma}, sigma_prev={sigma_next}")
        # upon completion increase step index by one
        self._step_index += 1

        if not return_dict:
            return (prev_sample,)

        return UniInvEulerSchedulerOutput(prev_sample=prev_sample)
    
    def step1(
        self,
        model_output: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        sample: torch.FloatTensor,
        s_churn: float = 0.0,
        s_tmin: float = 0.0,
        s_tmax: float = float("inf"),
        s_noise: float = 1.0,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> Union[UniInvEulerSchedulerOutput, Tuple]:
        
        if (
            isinstance(timestep, int)
            or isinstance(timestep, torch.IntTensor)
            or isinstance(timestep, torch.LongTensor)
        ):
            raise ValueError(
                (
                    "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
                    " `HeunDiscreteScheduler.step()` is not supported. Make sure to pass"
                    " one of the `scheduler.timesteps` as a timestep."
                ),
            )
            
        sample = sample.to(torch.float32)

        sigma = self.sigmas[self.step_index]
        sigma_next = self.sigmas[self.step_index + 1]

        prev_sample = sample + (sigma_next - sigma) * model_output
        prev_sample = prev_sample.to(model_output.dtype)

        prev_sample = prev_sample.to(model_output.dtype)

        # upon completion increase step index by one
        self._step_index += 1

        if not return_dict:
            return (prev_sample,)

        return UniInvEulerSchedulerOutput(prev_sample=prev_sample)


@dataclass
class ConsistencyFlowMatchEulerDiscreteSchedulerOutput(BaseOutput):
    prev_sample: torch.FloatTensor
    pred_original_sample: torch.FloatTensor


class ConsistencyFlowMatchEulerDiscreteScheduler(SchedulerMixin, ConfigMixin):
    _compatibles = []
    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        pcm_timesteps: int = 50,
    ):
        sigmas = np.linspace(0, 1, num_train_timesteps)
        step_ratio = num_train_timesteps // pcm_timesteps

        euler_timesteps = (np.arange(1, pcm_timesteps) * step_ratio).round().astype(np.int64) - 1
        euler_timesteps = np.asarray([0] + euler_timesteps.tolist())

        self.euler_timesteps = euler_timesteps
        self.sigmas = sigmas[self.euler_timesteps]
        self.sigmas = torch.from_numpy((self.sigmas.copy())).to(dtype=torch.float32)
        self.timesteps = self.sigmas * num_train_timesteps
        self._step_index = None
        self._begin_index = None
        self.sigmas = self.sigmas.to("cpu")  # to avoid too much CPU/GPU communication

    @property
    def step_index(self):
        """
        The index counter for current timestep. It will increase 1 after each scheduler step.
        """
        return self._step_index

    @property
    def begin_index(self):
        """
        The index for the first timestep. It should be set from pipeline with `set_begin_index` method.
        """
        return self._begin_index

    # Copied from diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler.set_begin_index
    def set_begin_index(self, begin_index: int = 0):
        """
        Sets the begin index for the scheduler. This function should be run from pipeline before the inference.

        Args:
            begin_index (`int`):
                The begin index for the scheduler.
        """
        self._begin_index = begin_index

    def _sigma_to_t(self, sigma):
        return sigma * self.config.num_train_timesteps

    def set_timesteps(
        self,
        num_inference_steps: int = None,
        device: Union[str, torch.device] = None,
        sigmas: Optional[List[float]] = None,
    ):
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model.
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        """
        self.num_inference_steps = num_inference_steps if num_inference_steps is not None else len(sigmas)
        inference_indices = np.linspace(
            0, self.config.pcm_timesteps, num=self.num_inference_steps, endpoint=False
        )
        inference_indices = np.floor(inference_indices).astype(np.int64)
        inference_indices = torch.from_numpy(inference_indices).long()

        self.sigmas_ = self.sigmas[inference_indices]
        timesteps = self.sigmas_ * self.config.num_train_timesteps
        self.timesteps = timesteps.to(device=device)
        self.sigmas_ = torch.cat(
            [self.sigmas_, torch.ones(1, device=self.sigmas_.device)]
        )

        self._step_index = None
        self._begin_index = None

    def index_for_timestep(self, timestep, schedule_timesteps=None):
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps

        indices = (schedule_timesteps == timestep).nonzero()

        # The sigma index that is taken for the **very** first `step`
        # is always the second index (or the last index if there is only 1)
        # This way we can ensure we don't accidentally skip a sigma in
        # case we start in the middle of the denoising schedule (e.g. for image-to-image)
        pos = 1 if len(indices) > 1 else 0

        return indices[pos].item()

    def _init_step_index(self, timestep):
        if self.begin_index is None:
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        sample: torch.FloatTensor,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> Union[ConsistencyFlowMatchEulerDiscreteSchedulerOutput, Tuple]:
        if (
            isinstance(timestep, int)
            or isinstance(timestep, torch.IntTensor)
            or isinstance(timestep, torch.LongTensor)
        ):
            raise ValueError(
                (
                    "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
                    " `EulerDiscreteScheduler.step()` is not supported. Make sure to pass"
                    " one of the `scheduler.timesteps` as a timestep."
                ),
            )

        if self.step_index is None:
            self._init_step_index(timestep)

        sample = sample.to(torch.float32)

        sigma = self.sigmas_[self.step_index]
        sigma_next = self.sigmas_[self.step_index + 1]

        prev_sample = sample + (sigma_next - sigma) * model_output
        prev_sample = prev_sample.to(model_output.dtype)

        pred_original_sample = sample + (1.0 - sigma) * model_output
        pred_original_sample = pred_original_sample.to(model_output.dtype)

        self._step_index += 1

        if not return_dict:
            return (prev_sample,)

        return ConsistencyFlowMatchEulerDiscreteSchedulerOutput(prev_sample=prev_sample,
                                                                pred_original_sample=pred_original_sample)

    def __len__(self):
        return self.config.num_train_timesteps
