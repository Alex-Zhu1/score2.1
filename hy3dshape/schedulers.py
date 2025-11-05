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
import nvdiffrast.torch as dr

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

    Euler scheduler.

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
        mesh_ref: Optional[trimesh.Trimesh] = None,
        guidance_config: Optional[dict] = None,
    ) -> Union["FlowMatchEulerDiscreteSchedulerOutput", Tuple]:
        """
        Predict the sample from the previous timestep by reversing the SDE.
        
        Args:
            model_output: Direct output from learned diffusion model
            timestep: Current discrete timestep in the diffusion chain
            sample: Current instance of a sample created by the diffusion process
            To: Global transformation matrix (4x4)
            _export: Function to convert latent/depth to mesh
            reference_mesh_path: Path to reference mesh for 2D guidance (optional)
            enable_guidance_2d: Whether to enable 2D guidance optimization
            guidance_config: Dictionary with guidance parameters (lr, num_steps, weights)
            Other args: Standard scheduler parameters
        
        Returns:
            FlowMatchEulerDiscreteSchedulerOutput or tuple with prev_sample
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
        if enable_guidance_2d and mesh_ref is not None:
            model_output = self._apply_2d_guidance(
                model_output=model_output,
                sample=sample,
                sigma=sigma,
                To=To,
                _export=_export,
                mesh_ref=mesh_ref,
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

    @torch.enable_grad()
    def _apply_2d_guidance(
        self,
        model_output: torch.FloatTensor,
        sample: torch.FloatTensor,
        sigma: float,
        To: torch.Tensor,
        _export: Callable,
        mesh_ref: trimesh.Trimesh,
        device: torch.device,
        original_dtype: torch.dtype,
        config: dict,
    ) -> torch.FloatTensor:
        """
        Apply 2D guidance optimization to refine model output.
        
        Args:
            model_output: Current velocity prediction
            sample: Current sample x_t
            sigma: Current noise level
            To: Global transformation matrix
            _export: Function to export latent to mesh
            reference_mesh_path: Path to reference mesh
            device: Torch device
            config: Configuration dict with lr, num_steps, loss_weights, etc.
        
        Returns:
            Optimized velocity prediction
        """
        
        # Extract config with defaults
        num_steps = config.get('num_steps', 50)
        lr_scale = config.get('lr_scale', 0.01)
        lr_translation = config.get('lr_translation', 0.01)
        lr_rotation = config.get('lr_rotation', 0.5)
        lr_velocity = config.get('lr_velocity', 0.0001)
        
        weight_norm = config.get('weight_norm', 10.0)
        weight_disp = config.get('weight_disp', 10.0)
        weight_sil = config.get('weight_sil', 10.0)
        weight_reg = config.get('weight_reg', 0.001)
        
        fov_x = config.get('fov_x', 41.039776)
        fov_y = config.get('fov_y', 41.039776)
        
        # Initialize optimizable parameters
        para_velocity = model_output.clone().detach().requires_grad_(True)
        
        scale = torch.ones(3, device=device, dtype=torch.float32, requires_grad=True)
        rotation = torch.eye(3, device=device, dtype=torch.float32, requires_grad=True)
        translation = torch.zeros(3, device=device, dtype=torch.float32, requires_grad=True)
        
        # Setup optimizer
        optimizer = torch.optim.Adam([
            {'params': [scale], 'lr': lr_scale},
            {'params': [translation], 'lr': lr_translation},
            {'params': [rotation], 'lr': lr_rotation},
            {'params': [para_velocity], 'lr': lr_velocity},
        ])
        
        # Load and render reference mesh once (outside loop)
        try:
            # mesh_ref = mesh_ref
            # ref_normal, ref_disparity, ref_silhouette = self.render_maps(
            #     mesh_ref, fov_x=fov_x, fov_y=fov_y
            # )
            # # Move to correct device
            # ref_normal = self.to_device(ref_normal, device)
            # ref_disparity = self.to_device(ref_disparity, device)
            # ref_silhouette = self.to_device(ref_silhouette, device)

            # 上面是读取参考网格并渲染得到参考图
            # 下面直接利用处理数据时候的img
            base_dir = "/mnt/data/users/haiming.zhu/hoi/Hunyuan3D-2.1/hy3dshape/outputs_depth/325_cropped_hoi_1"

            # 1. 读取法线图（normal_map，RGB）
            ref_normal = cv.imread(f"{base_dir}/rendered_normal.png", cv.IMREAD_COLOR)
            ref_normal = cv.cvtColor(ref_normal, cv.COLOR_BGR2RGB) / 255.0  # 转 RGB 并归一化
            ref_normal = self.to_device(ref_normal, device)  # shape: (H, W, 3)

            # 2. 读取视差图（disparity_map，EXR 浮点）
            # ref_disparity = cv.imread(f"{base_dir}/rendered_disparity.exr", cv.IMREAD_UNCHANGED)
            # if ref_disparity is None:
            #     raise FileNotFoundError(f"Cannot read EXR file: {base_dir}/rendered_disparity.exr")
            # ref_disparity = self.to_device(ref_disparity.astype(np.float32), device)  # shape: (H, W)
            ref_disparity = iio.imread(f"{base_dir}/rendered_disparity.exr").astype(np.float32)
            ref_disparity = self.to_device(ref_disparity, device)  # shape: (H, W)


            # 3. 读取掩码（silhouette，灰度）
            ref_silhouette = cv.imread(f"{base_dir}/rendered_silhouette.png", cv.IMREAD_GRAYSCALE)
            ref_silhouette = (ref_silhouette / 255.0).astype(np.float32)
            ref_silhouette = self.to_device(ref_silhouette, device)  # shape: (H, W)

            # 
            self.glctx = dr.RasterizeGLContext()
        except Exception as e:
            print(f"Warning: Could not load reference mesh: {e}")
            return model_output  # Fall back to original output
        
        # Optimization loop
        for opt_step in range(num_steps):
            optimizer.zero_grad()
            
            # Build transformation matrix
            transform = self._build_transform_matrix(scale, rotation, translation, device)
            
            # Predict x_1 using current velocity
            x1 = sample + (1.0 - sigma) * para_velocity
            
            # Export to mesh and apply transforms
            x1 = x1.to(original_dtype)
            mesh_x1 = _export(x1, output_type='mesh')   # vae的dtype 和 x1不一样
            mesh_x1.apply_transform(transform.detach().cpu().numpy())
            mesh_x1.apply_transform(To)
            
            # Render current mesh
            normal_map, disparity_map, silhouette = self.render_maps(
                mesh_x1, fov_x=fov_x, fov_y=fov_y
            )
            # 可视化下normal_map, disparity_map, silhouette
            cv.imwrite(f"./debug/opt_step_{opt_step:03d}_normal.png", (normal_map.cpu().numpy() * 255).astype(np.uint8))
            cv.imwrite(f"./debug/opt_step_{opt_step:03d}_disparity.exr", disparity_map.cpu().numpy().astype(np.float32))
            cv.imwrite(f"./debug/opt_step_{opt_step:03d}_silhouette.png", (silhouette.cpu().numpy() * 255).astype(np.uint8))
            
            # Move to device
            normal_map = self.to_device(normal_map, device)
            disparity_map = self.to_device(disparity_map, device)
            silhouette = self.to_device(silhouette, device)
            
            # Compute losses
            loss_norm = F.l1_loss(normal_map, ref_normal)
            loss_disp = F.l1_loss(disparity_map, ref_disparity)
            loss_sil = F.binary_cross_entropy_with_logits(silhouette, ref_silhouette)
            
            # Regularization losses
            loss_translation = torch.norm(translation)
            loss_scale = torch.abs(torch.det(transform[:3, :3]) - 1.0)
            loss_reg = loss_translation + loss_scale
            
            # Total loss
            total_loss = (
                weight_norm * loss_norm +
                weight_disp * loss_disp +
                weight_sil * loss_sil +
                weight_reg * loss_reg
            )
            
            # Backward and optimize
            total_loss.backward()
            optimizer.step()
            
            # Logging
            if opt_step % 10 == 0:
                print(
                    f"[Step {opt_step}] "
                    f"L_norm={loss_norm.item():.4f}, "
                    f"L_disp={loss_disp.item():.4f}, "
                    f"L_sil={loss_sil.item():.4f}, "
                    f"L_reg={loss_reg.item():.6f}, "
                    f"Total={total_loss.item():.4f}"
                )
        
        # Return optimized velocity
        return para_velocity.detach()
    
    def to_device(self, x, device=None, dtype=torch.float32):
        """
        将输入数据转换为 torch.Tensor，并移动到指定设备和数据类型。
        支持 torch.Tensor 或 numpy.ndarray。
        """
        if isinstance(x, torch.Tensor):
            return x.to(device=device, dtype=dtype)
        elif isinstance(x, np.ndarray):
            return torch.as_tensor(x, dtype=dtype, device=device)
        else:
            raise TypeError(f"Unsupported type {type(x)}")


    def _build_transform_matrix(
        self,
        scale: torch.Tensor,
        rotation: torch.Tensor,
        translation: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Build 4x4 transformation matrix from scale, rotation, and translation.
        
        Args:
            scale: (3,) scaling factors [sx, sy, sz]
            rotation: (3, 3) rotation matrix
            translation: (3,) translation vector [tx, ty, tz]
            device: Target device
        
        Returns:
            (4, 4) transformation matrix
        """
        transform = torch.eye(4, device=device, dtype=torch.float32)
        
        # Apply scale to rotation matrix
        SR = torch.diag(scale) @ rotation
        
        # Fill in transformation matrix
        transform[:3, :3] = SR
        transform[:3, 3] = translation
        
        return transform


    def __len__(self):
        return self.config.num_train_timesteps



    def render_maps(
            self,
            mesh,
            fov_x: float,
            fov_y: float,
            render_res: tuple[int, int] = (224, 224),
            scene_bg_color: tuple[int, int, int] = (0, 0, 0),
            device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        ):
        """
        使用 FOV 渲染法线图、视差图和轮廓图。
        
        Args:
            mesh: pyrender.Mesh 对象或包含 vertices, faces 属性的对象
            fov_x, fov_y: 水平和垂直视场角（单位：度）
            render_res: (width, height)
            scene_bg_color: 背景颜色
        
        Returns:
            normal_map: (H, W, 3) 世界空间法线图，范围 [0, 1]
            disparity_map: (H, W) 视差图 (1/depth)
            silhouette: (H, W) 二值轮廓图
        """
        # 提取顶点和面
        if hasattr(mesh, 'primitives'):  # pyrender.Mesh
            verts = torch.tensor(mesh.primitives[0].positions, device=device, dtype=torch.float32)
            faces = torch.tensor(mesh.primitives[0].indices, device=device, dtype=torch.int32)
        else:  # 直接访问属性
            verts = torch.tensor(mesh.vertices, device=device, dtype=torch.float32)
            faces = torch.tensor(mesh.faces, device=device, dtype=torch.int32)
        
        verts = verts.unsqueeze(0)  # (1, V, 3)
        faces = faces.unsqueeze(0)  # (1, F, 3)

        W, H = render_res

        # === 1. 由 FOV 计算焦距 ===
        fx = W / (2 * np.tan(np.deg2rad(fov_x) / 2))
        fy = H / (2 * np.tan(np.deg2rad(fov_y) / 2))
        cx, cy = W / 2, H / 2

        # === 2. 初始化投影矩阵与相机 ===
        projection = torch.tensor([
            [fx, 0, cx, 0],
            [0, fy, cy, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], device=device, dtype=torch.float32)

        c2ws = torch.eye(4, device=device, dtype=torch.float32)

        # === 3. 计算顶点法线 ===
        normals = self.compute_vertex_normals(verts[0], faces[0])  # (V, 3)
        normals = normals.unsqueeze(0)  # (1, V, 3)

        # === 4. 渲染法线图 ===
        # 将法线从 [-1, 1] 映射到 [0, 1] 用于渲染
        normal_colors = (normals + 1.0) / 2.0
        normal_map, alpha_map = self.renderer(
            verts, faces, normal_colors[0], projection, c2ws, (H, W)
        )

        # === 5. 渲染深度图 ===
        depth_map = self.render_depth(verts, faces, projection, c2ws, (H, W))

        # === 6. 计算视差图 ===
        # 视差 = 1 / depth，避免除零
        epsilon = 1e-6
        disparity_map = 1.0 / (depth_map + epsilon)
        # 背景设为0
        disparity_map = disparity_map * alpha_map

        # === 7. 提取轮廓 ===
        silhouette = (alpha_map > 0.5).float()

        return normal_map, disparity_map, silhouette


    def renderer(self, verts, tri, color, projection, c2ws, resolution):
        """
        基础渲染器，返回颜色图和 alpha 通道
        
        Returns:
            img: (H, W, 3) RGB 图像
            alpha: (H, W) alpha 通道
        """
        device = projection.device
        
        # 齐次坐标
        ones = torch.ones(1, verts.shape[1], 1, device=device, dtype=torch.float32)
        pos = torch.cat((verts, ones), dim=2)  # (1, V, 4)
        
        # 计算 MVP 矩阵
        try:
            view_matrix = torch.inverse(c2ws)
        except:
            view_matrix = torch.linalg.pinv(c2ws)
        
        mat = (projection @ view_matrix).unsqueeze(0)  # (1, 4, 4)
        pos_clip = pos @ mat.mT  # (1, V, 4)
        
        # 光栅化
        rast, _ = dr.rasterize(self.glctx, pos_clip, tri, resolution)
        
        # 插值颜色
        color = color.unsqueeze(0) if color.dim() == 2 else color
        out, _ = dr.interpolate(color, rast, tri)
        out = dr.antialias(out, rast, pos_clip, tri)
        
        # 提取 RGB 和 alpha
        img = torch.flip(out[0, :, :, :3], dims=[0])  # (H, W, 3)
        alpha = torch.flip(out[0, :, :, 3], dims=[0])  # (H, W)
        
        return img, alpha


    def render_depth(self, verts, tri, projection, c2ws, resolution):
        """
        渲染深度图
        
        Returns:
            depth: (H, W) 深度值
        """
        device = projection.device
        
        # 齐次坐标
        ones = torch.ones(1, verts.shape[1], 1, device=device, dtype=torch.float32)
        pos = torch.cat((verts, ones), dim=2)  # (1, V, 4)
        
        # MVP 变换
        try:
            view_matrix = torch.inverse(c2ws)
        except:
            view_matrix = torch.linalg.pinv(c2ws)
        
        mat = (projection @ view_matrix).unsqueeze(0)
        pos_clip = pos @ mat.mT
        
        # 光栅化
        rast, _ = dr.rasterize(self.glctx, pos_clip, tri, resolution)
        
        # 计算相机空间深度（Z 值）
        pos_camera = (pos @ view_matrix.unsqueeze(0).mT)[:, :, 2:3]  # (1, V, 1)
        
        # 插值深度
        depth, _ = dr.interpolate(pos_camera, rast, tri)
        depth = torch.flip(depth[0, :, :, 0], dims=[0])  # (H, W)
        
        # 取绝对值（深度为正）
        depth = torch.abs(depth)
        
        return depth


    def compute_vertex_normals(self, verts, faces):
        """
        计算顶点法线
        
        Args:
            verts: (V, 3) 顶点坐标
            faces: (F, 3) 面索引
        
        Returns:
            normals: (V, 3) 单位法线向量
        """
        # 获取三角形的三个顶点
        v0 = verts[faces[:, 0]]  # (F, 3)
        v1 = verts[faces[:, 1]]
        v2 = verts[faces[:, 2]]
        
        # 计算面法线
        face_normals = torch.cross(v1 - v0, v2 - v0, dim=1)  # (F, 3)
        
        # 归一化
        face_normals = torch.nn.functional.normalize(face_normals, dim=1)
        
        # 累加到顶点
        vertex_normals = torch.zeros_like(verts)
        vertex_normals.index_add_(0, faces[:, 0], face_normals)
        vertex_normals.index_add_(0, faces[:, 1], face_normals)
        vertex_normals.index_add_(0, faces[:, 2], face_normals)
        
        # 归一化
        vertex_normals = torch.nn.functional.normalize(vertex_normals, dim=1)
        
        return vertex_normals

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
