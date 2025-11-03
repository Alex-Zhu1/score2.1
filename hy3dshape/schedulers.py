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

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.utils import BaseOutput, logging

import pyrender

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
        _export: Callable,
        To: torch.Tensor,
        s_churn: float = 0.0,
        s_tmin: float = 0.0,
        s_tmax: float = float("inf"),
        s_noise: float = 1.0,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> Union[FlowMatchEulerDiscreteSchedulerOutput, Tuple]:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                A current instance of a sample created by the diffusion process.
            s_churn (`float`):
            s_tmin  (`float`):
            s_tmax  (`float`):
            s_noise (`float`, defaults to 1.0):
                Scaling factor for noise added to the sample.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            return_dict (`bool`):
                Whether or not to return a [`~schedulers.scheduling_euler_discrete.EulerDiscreteSchedulerOutput`] or
                tuple.

        Returns:
            [`~schedulers.scheduling_euler_discrete.EulerDiscreteSchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_euler_discrete.EulerDiscreteSchedulerOutput`] is
                returned, otherwise a tuple is returned where the first element is the sample tensor.
        """

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

        # Upcast to avoid precision issues when computing prev_sample
        sample = sample.to(torch.float32)

        sigma = self.sigmas[self.step_index]
        sigma_next = self.sigmas[self.step_index + 1]

        prev_sample = sample + (sigma_next - sigma) * model_output

        # Cast sample back to model compatible dtype
        prev_sample = prev_sample.to(model_output.dtype)

        # predict the original sample (x_0) from the model's output
        pred_original_sample = sample + (1.0 - sigma) * model_output
        pred_original_sample = pred_original_sample.to(model_output.dtype)

        guidance_2D = False
        if guidance_2D:
            # 2D signal for guidance
            # 1 set optimization params
            para_velocity = model_output.clone().detach().requires_grad_(True)

            scale = torch.ones(3, requires_grad=True)            # (sx, sy, sz)
            rotation = torch.eye(3, requires_grad=True)          # 初始单位旋转
            translation = torch.zeros(3, requires_grad=True)     # 初始平移

            def build_mesh_transform(scale, rotation, translation):
                # Apply scale to rotation (缩放 + 旋转)
                SR = torch.diag(scale) @ rotation  # 3×3
                # Build 4×4 transform
                transform = torch.eye(4)
                transform[:3, :3] = SR
                transform[:3, 3] = translation
                return transform

            para_Tp = build_mesh_transform(scale, rotation, translation)

            optimizer = torch.optim.Adam([
                    {'params': [scale], 'lr': 0.01},
                    {'params': [translation], 'lr': 0.01},
                    {'params': [rotation], 'lr': 0.5},
                    {'params': [para_velocity], 'lr': 0.0001},
                ])

            for opt_step in range(50):
                optimizer.zero_grad()
                
                x1 = sample + (1.0 - sigma) * para_velocity
                # 解码
                mesh_x1 =   _export(x1)
                mesh_x1.apply_transform(para_Tp)
                mesh_x1.apply_transform(To)
                # 渲染
                rend_x1 = self.render_mesh(mesh_x1, image_size=224)



        # upon completion increase step index by one
        self._step_index += 1

        if not return_dict:
            return (prev_sample,)

        return ConsistencyFlowMatchEulerDiscreteSchedulerOutput(prev_sample=prev_sample,
                                                     pred_original_sample=pred_original_sample)

    def __len__(self):
        return self.config.num_train_timesteps

    
    def render_maps(
            self,
            mesh,
            focal_length: float,
            render_res: Tuple[int, int] = (224, 224),
            scene_bg_color: Tuple[int, int, int] = (0, 0, 0),
        ):
        """
        渲染白模的 normal map, disparity map 和 silhouette
        
        Returns:
            normal_map: (H, W, 3) 世界空间的法线图，范围 [0, 1]
            disparity_map: (H, W) 视差图
            silhouette: (H, W) 二值轮廓图
        """
        renderer = pyrender.OffscreenRenderer(
            viewport_width=render_res[0],
            viewport_height=render_res[1],
            point_size=1.0
        )
        
        # 设置相机
        camera_pose = np.eye(4)
        camera_center = [render_res[0] / 2., render_res[1] / 2.]
        camera = pyrender.IntrinsicsCamera(
            fx=focal_length, 
            fy=focal_length,
            cx=camera_center[0], 
            cy=camera_center[1], 
            zfar=1e12
        )
        
        # ===== 1. 渲染 Normal Map =====
        scene_normal = pyrender.Scene(bg_color=[*scene_bg_color, 0.0])
        scene_normal.add(mesh, 'mesh')
        camera_node = pyrender.Node(camera=camera, matrix=camera_pose)
        scene_normal.add_node(camera_node)
        
        # 使用 FLAT 或 SKIP_CULL_FACES 标志来获取法线
        flags = pyrender.RenderFlags.FLAT | pyrender.RenderFlags.RGBA
        color_normal, depth = renderer.render(scene_normal, flags=flags)
        
        # 从深度图计算法线（更可靠的方法）
        normal_map = self.depth_to_normal(depth, focal_length, camera_center)
        
        # ===== 2. 渲染 Depth/Disparity Map =====
        scene_depth = pyrender.Scene(bg_color=[*scene_bg_color, 0.0])
        scene_depth.add(mesh, 'mesh')
        camera_node_depth = pyrender.Node(camera=camera, matrix=camera_pose)
        scene_depth.add_node(camera_node_depth)
        
        _, depth_map = renderer.render(scene_depth)
        
        # 将深度转换为视差（disparity = 1/depth）
        disparity_map = np.zeros_like(depth_map)
        valid_depth = depth_map > 0
        disparity_map[valid_depth] = 1.0 / depth_map[valid_depth]
        
        # ===== 3. 生成 Silhouette =====
        silhouette = (depth_map > 0).astype(np.float32)
        
        renderer.delete()
        
        return normal_map, disparity_map, silhouette


    def depth_to_normal(self, depth, focal_length, camera_center):
        """
        从深度图计算法线图
        
        Args:
            depth: (H, W) 深度图
            focal_length: 焦距
            camera_center: [cx, cy] 相机中心
            
        Returns:
            normal_map: (H, W, 3) 法线图，范围 [0, 1]
        """
        H, W = depth.shape
        
        # 创建像素坐标网格
        y, x = np.mgrid[0:H, 0:W]
        
        # 将像素坐标转换为相机空间的 3D 点
        z = depth
        x_cam = (x - camera_center[0]) * z / focal_length
        y_cam = (y - camera_center[1]) * z / focal_length
        
        # 计算梯度
        dz_dx = np.zeros_like(depth)
        dz_dy = np.zeros_like(depth)
        
        # 使用中心差分
        dz_dx[:, 1:-1] = (z[:, 2:] - z[:, :-2]) / 2.0
        dz_dy[1:-1, :] = (z[2:, :] - z[:-2, :]) / 2.0
        
        # 计算法线向量 (梯度叉乘)
        normal = np.zeros((H, W, 3))
        normal[:, :, 0] = -dz_dx * focal_length / (W / 2.0)
        normal[:, :, 1] = -dz_dy * focal_length / (H / 2.0)
        normal[:, :, 2] = 1.0
        
        # 归一化
        norm = np.linalg.norm(normal, axis=2, keepdims=True)
        norm[norm == 0] = 1.0  # 避免除零
        normal = normal / norm
        
        # 将法线从 [-1, 1] 转换到 [0, 1] 用于可视化
        normal_map = (normal + 1.0) / 2.0
        
        # 背景设为黑色
        mask = depth > 0
        normal_map[~mask] = 0
        
        return normal_map

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
