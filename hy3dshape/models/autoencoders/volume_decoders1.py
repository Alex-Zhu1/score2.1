# Differentiable FlashVDM Volume Decoding
# Based on Hunyuan 3D (TENCENT HUNYUAN NON-COMMERCIAL LICENSE)
# Modified to support gradient backpropagation

from typing import Union, Tuple, List, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from tqdm import tqdm


def extract_near_surface_volume_fn(input_tensor: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Extract near-surface voxels (non-differentiable, used for sampling position selection).
    Returns a binary mask indicating which voxels are near the surface.
    """
    device = input_tensor.device
    
    val = input_tensor + alpha
    valid_mask = val > -9000

    def get_neighbor(t: torch.Tensor, shift: int, axis: int) -> torch.Tensor:
        if shift == 0:
            return t.clone()

        pad_dims = [0, 0, 0, 0, 0, 0]

        if axis == 0:
            pad_idx = 0 if shift > 0 else 1
            pad_dims[pad_idx] = abs(shift)
        elif axis == 1:
            pad_idx = 2 if shift > 0 else 3
            pad_dims[pad_idx] = abs(shift)
        elif axis == 2:
            pad_idx = 4 if shift > 0 else 5
            pad_dims[pad_idx] = abs(shift)

        padded = F.pad(t.unsqueeze(0).unsqueeze(0), pad_dims[::-1], mode='replicate')

        slice_dims = [slice(None)] * 3
        if axis == 0:
            slice_dims[0] = slice(shift, None) if shift > 0 else slice(None, shift)
        elif axis == 1:
            slice_dims[1] = slice(shift, None) if shift > 0 else slice(None, shift)
        elif axis == 2:
            slice_dims[2] = slice(shift, None) if shift > 0 else slice(None, shift)

        padded = padded.squeeze(0).squeeze(0)
        return padded[slice_dims]

    left = get_neighbor(val, 1, axis=0)
    right = get_neighbor(val, -1, axis=0)
    back = get_neighbor(val, 1, axis=1)
    front = get_neighbor(val, -1, axis=1)
    down = get_neighbor(val, 1, axis=2)
    up = get_neighbor(val, -1, axis=2)

    def safe_where(neighbor):
        return torch.where(neighbor > -9000, neighbor, val)

    left, right = safe_where(left), safe_where(right)
    back, front = safe_where(back), safe_where(front)
    down, up = safe_where(down), safe_where(up)

    sign = torch.sign(val.to(torch.float32))
    neighbors_sign = torch.stack([
        torch.sign(left.to(torch.float32)),
        torch.sign(right.to(torch.float32)),
        torch.sign(back.to(torch.float32)),
        torch.sign(front.to(torch.float32)),
        torch.sign(down.to(torch.float32)),
        torch.sign(up.to(torch.float32))
    ], dim=0)

    same_sign = torch.all(neighbors_sign == sign, dim=0)
    mask = (~same_sign).to(torch.int32)
    return mask * valid_mask.to(torch.int32)


def generate_dense_grid_points(
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    octree_resolution: int,
    indexing: str = "ij",
) -> Tuple[np.ndarray, List[int], np.ndarray]:
    """Generate dense grid points for volume sampling."""
    length = bbox_max - bbox_min
    num_cells = octree_resolution

    x = np.linspace(bbox_min[0], bbox_max[0], int(num_cells) + 1, dtype=np.float32)
    y = np.linspace(bbox_min[1], bbox_max[1], int(num_cells) + 1, dtype=np.float32)
    z = np.linspace(bbox_min[2], bbox_max[2], int(num_cells) + 1, dtype=np.float32)
    [xs, ys, zs] = np.meshgrid(x, y, z, indexing=indexing)
    xyz = np.stack((xs, ys, zs), axis=-1)
    grid_size = [int(num_cells) + 1, int(num_cells) + 1, int(num_cells) + 1]

    return xyz, grid_size, length


class DifferentiableFlashVDMVolumeDecoding:
    """
    Differentiable version of FlashVDM Volume Decoding.
    
    Key changes from original:
    1. Removed @torch.no_grad() to allow gradient flow
    2. Separated position selection (non-differentiable) from value computation (differentiable)
    3. Uses straight-through estimator pattern for sparse sampling
    
    Usage:
        decoder = DifferentiableFlashVDMVolumeDecoding(topk_mode='mean')
        
        # For inference (no gradients needed)
        with torch.no_grad():
            logits = decoder(latents, geo_decoder, ...)
        
        # For training (gradients flow through decoder)
        logits = decoder(latents, geo_decoder, ...)
        loss = some_loss_fn(logits, target)
        loss.backward()  # Gradients flow to latents and geo_decoder
    """
    
    def __init__(self, topk_mode: str = 'mean'):
        if topk_mode not in ['mean', 'merge']:
            raise ValueError(f'Unsupported topk_mode {topk_mode}, available: {["mean", "merge"]}')
        
        self.topk_mode = topk_mode
        self._processor = None
    
    def _get_processor(self):
        """Lazy initialization of processor to avoid import issues."""
        if self._processor is None:
            # Import here to match original code structure
            # Users need to provide their own processor implementations
            if self.topk_mode == 'mean':
                from .attention_processors import FlashVDMCrossAttentionProcessor
                self._processor = FlashVDMCrossAttentionProcessor()
            else:
                from .attention_processors import FlashVDMTopMCrossAttentionProcessor
                self._processor = FlashVDMTopMCrossAttentionProcessor()
        return self._processor
    
    def _compute_resolutions(
        self, 
        octree_resolution: int, 
        min_resolution: int, 
        mini_grid_num: int
    ) -> List[int]:
        """Compute hierarchical resolution levels."""
        resolutions = []
        res = octree_resolution
        if res < min_resolution:
            resolutions.append(res)
        while res >= min_resolution:
            resolutions.append(res)
            res = res // 2
        resolutions.reverse()
        resolutions[0] = round(resolutions[0] / mini_grid_num) * mini_grid_num - 1
        for i, resolution in enumerate(resolutions[1:]):
            resolutions[i + 1] = resolutions[0] * 2 ** (i + 1)
        return resolutions
    
    def _select_sparse_positions(
        self,
        grid_logits: torch.Tensor,
        mc_level: float,
        bbox_min: np.ndarray,
        resolution: np.ndarray,
        target_grid_size: np.ndarray,
        dilate: nn.Conv3d,
        expand_num: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...], torch.Tensor]:
        """
        Select sparse sampling positions based on current grid logits.
        This is the NON-DIFFERENTIABLE part - we detach here.
        
        Returns:
            next_points: Coordinates of points to sample [1, N, 3]
            nidx: Indices in the target grid
            sort_indices: Indices to restore original order
        """
        with torch.no_grad():
            next_index = torch.zeros(tuple(target_grid_size), dtype=dtype, device=device)
            
            # Extract surface voxels
            curr_points = extract_near_surface_volume_fn(grid_logits.squeeze(0).detach(), mc_level)
            curr_points += grid_logits.squeeze(0).detach().abs() < 0.95
            
            # Dilate to expand region
            for _ in range(expand_num):
                curr_points = dilate(curr_points.unsqueeze(0).to(dtype)).squeeze(0)
            
            (cidx_x, cidx_y, cidx_z) = torch.where(curr_points > 0)
            
            # Map to next resolution
            next_index[cidx_x * 2, cidx_y * 2, cidx_z * 2] = 1
            for _ in range(2 - expand_num):
                next_index = dilate(next_index.unsqueeze(0)).squeeze(0)
            
            nidx = torch.where(next_index > 0)
            
            # Convert indices to coordinates
            next_points = torch.stack(nidx, dim=1).float()
            next_points = (next_points * torch.tensor(resolution, dtype=torch.float32, device=device) +
                          torch.tensor(bbox_min, dtype=torch.float32, device=device))
            
            # Spatial sorting for efficient processing
            query_grid_num = 6
            min_val = next_points.min(dim=0).values
            max_val = next_points.max(dim=0).values
            
            # Avoid division by zero
            range_val = max_val - min_val
            range_val = torch.where(range_val > 0, range_val, torch.ones_like(range_val))
            
            vol_queries_index = (next_points - min_val) / range_val * (query_grid_num - 0.001)
            index = torch.floor(vol_queries_index).long()
            index = index[..., 0] * (query_grid_num ** 2) + index[..., 1] * query_grid_num + index[..., 2]
            sort_result = index.sort()
            
            next_points = next_points[sort_result.indices].unsqueeze(0).contiguous()
            
            return next_points, nidx, sort_result
    
    def __call__(
        self,
        latents: torch.FloatTensor,
        geo_decoder: nn.Module,
        bounds: Union[Tuple[float], List[float], float] = 1.01,
        num_chunks: int = 10000,
        mc_level: float = 0.0,
        octree_resolution: int = None,
        min_resolution: int = 63,
        mini_grid_num: int = 4,
        enable_pbar: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """
        Differentiable volume decoding.
        
        Gradient flow:
        - Position selection is detached (discrete decisions)
        - Decoder evaluation at selected positions maintains gradients
        - Final logits can be used for loss computation and backprop
        
        Args:
            latents: Input latent codes [B, P, C]
            geo_decoder: Geometry decoder network
            bounds: Bounding box bounds
            num_chunks: Batch size for decoder queries
            mc_level: Marching cubes level
            octree_resolution: Target resolution
            min_resolution: Minimum resolution for hierarchical sampling
            mini_grid_num: Grid subdivision factor
            enable_pbar: Enable progress bar
            
        Returns:
            grid_logits: Volume logits [B, D, D, D], differentiable w.r.t. latents
        """
        processor = self._get_processor()
        geo_decoder.set_cross_attention_processor(processor)
        
        device = latents.device
        dtype = latents.dtype
        batch_size = latents.shape[0]
        
        # Compute resolution hierarchy
        resolutions = self._compute_resolutions(octree_resolution, min_resolution, mini_grid_num)
        
        # Setup bounding box
        if isinstance(bounds, float):
            bounds = [-bounds, -bounds, -bounds, bounds, bounds, bounds]
        bbox_min = np.array(bounds[0:3])
        bbox_max = np.array(bounds[3:6])
        bbox_size = bbox_max - bbox_min
        
        # Generate initial grid
        xyz_samples, grid_size, length = generate_dense_grid_points(
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            octree_resolution=resolutions[0],
            indexing="ij"
        )
        
        # Dilation kernel for region expansion
        dilate = nn.Conv3d(1, 1, 3, padding=1, bias=False, device=device, dtype=dtype)
        dilate.weight = nn.Parameter(
            torch.ones(dilate.weight.shape, dtype=dtype, device=device),
            requires_grad=False  # This is not a learnable parameter
        )
        
        grid_size = np.array(grid_size)
        xyz_samples = torch.from_numpy(xyz_samples).to(device, dtype=dtype)
        
        # Reshape for mini-grid processing
        mini_grid_size = xyz_samples.shape[0] // mini_grid_num
        xyz_samples = xyz_samples.view(
            mini_grid_num, mini_grid_size,
            mini_grid_num, mini_grid_size,
            mini_grid_num, mini_grid_size, 3
        ).permute(
            0, 2, 4, 1, 3, 5, 6
        ).reshape(
            -1, mini_grid_size * mini_grid_size * mini_grid_size, 3
        )
        
        # ========== STAGE 1: Initial coarse sampling (DIFFERENTIABLE) ==========
        batch_logits = []
        num_batches = max(num_chunks // xyz_samples.shape[1], 1)
        
        for start in tqdm(range(0, xyz_samples.shape[0], num_batches),
                         desc=f"FlashVDM Coarse Sampling", disable=not enable_pbar):
            queries = xyz_samples[start: start + num_batches, :]
            batch = queries.shape[0]
            batch_latents = repeat(latents.squeeze(0), "p c -> b p c", b=batch)
            processor.topk = True
            
            # This call IS differentiable
            logits = geo_decoder(queries=queries, latents=batch_latents)
            batch_logits.append(logits)
        
        grid_logits = torch.cat(batch_logits, dim=0).reshape(
            mini_grid_num, mini_grid_num, mini_grid_num,
            mini_grid_size, mini_grid_size, mini_grid_size
        ).permute(0, 3, 1, 4, 2, 5).contiguous().view(
            (batch_size, grid_size[0], grid_size[1], grid_size[2])
        )
        
        # ========== STAGE 2: Hierarchical refinement ==========
        for level_idx, octree_depth_now in enumerate(resolutions[1:]):
            target_grid_size = np.array([octree_depth_now + 1] * 3)
            resolution = bbox_size / octree_depth_now
            
            # Determine expansion for this level
            is_final_level = (octree_depth_now == resolutions[-1])
            expand_num = 0 if is_final_level else 1
            
            # NON-DIFFERENTIABLE: Select which positions to sample
            next_points, nidx, sort_result = self._select_sparse_positions(
                grid_logits=grid_logits,
                mc_level=mc_level,
                bbox_min=bbox_min,
                resolution=resolution,
                target_grid_size=target_grid_size,
                dilate=dilate,
                expand_num=expand_num,
                dtype=dtype,
                device=device,
            )
            
            # DIFFERENTIABLE: Compute values at selected positions
            unique_values = torch.unique(sort_result.values, return_counts=True)
            
            # Prepare output tensor
            next_logits = torch.full(
                tuple(target_grid_size), 
                -10000., 
                dtype=dtype, 
                device=device
            )
            sparse_logits = torch.zeros(
                next_points.shape[1], 
                dtype=dtype, 
                device=device
            )
            
            # Batched decoder evaluation (DIFFERENTIABLE)
            input_grid = [[], []]
            logits_list = []
            start_num = 0
            sum_num = 0
            
            for grid_index, count in zip(
                unique_values[0].cpu().tolist(), 
                unique_values[1].cpu().tolist()
            ):
                if sum_num + count < num_chunks or sum_num == 0:
                    sum_num += count
                    input_grid[0].append(grid_index)
                    input_grid[1].append(count)
                else:
                    processor.topk = input_grid
                    # DIFFERENTIABLE decoder call
                    logits_batch = geo_decoder(
                        queries=next_points[:, start_num:start_num + sum_num], 
                        latents=latents
                    )
                    start_num = start_num + sum_num
                    logits_list.append(logits_batch)
                    input_grid = [[grid_index], [count]]
                    sum_num = count
            
            # Process remaining
            if sum_num > 0:
                processor.topk = input_grid
                logits_batch = geo_decoder(
                    queries=next_points[:, start_num:start_num + sum_num], 
                    latents=latents
                )
                logits_list.append(logits_batch)
            
            # Concatenate all batches
            all_logits = torch.cat(logits_list, dim=1)
            
            # Restore original order using scatter (DIFFERENTIABLE)
            sparse_logits[sort_result.indices] = all_logits.squeeze(0).squeeze(-1)
            
            # Scatter into full grid
            # Note: This scatter operation is differentiable
            next_logits[nidx] = sparse_logits
            grid_logits = next_logits.unsqueeze(0)
        
        # Mark invalid regions
        grid_logits = torch.where(
            grid_logits == -10000.,
            torch.tensor(float('nan'), dtype=dtype, device=device),
            grid_logits
        )
        
        return grid_logits


class FlashVDMVolumeDecoding(DifferentiableFlashVDMVolumeDecoding):
    """
    Alternative version using Straight-Through Estimator (STE) for position selection.
    
    This version allows gradients to flow through the position selection process
    using a soft-to-hard approximation during forward pass, while using soft
    values during backward pass.
    
    Useful when you want to train the system end-to-end including the
    surface detection criteria.
    """
    
    def __init__(self, topk_mode: str = 'mean', temperature: float = 1.0):
        super().__init__(topk_mode)
        self.temperature = temperature
    
    def _soft_surface_mask(
        self, 
        logits: torch.Tensor, 
        mc_level: float
    ) -> torch.Tensor:
        """
        Compute a soft (differentiable) approximation of near-surface mask.
        
        Uses gradient of SDF-like values to detect surfaces.
        """
        # Compute soft absolute distance to surface
        soft_dist = torch.abs(logits + mc_level)
        
        # Compute spatial gradients (approximation of surface normal magnitude)
        grad_x = F.pad(logits[..., 1:] - logits[..., :-1], (0, 1))
        grad_y = F.pad(logits[..., 1:, :] - logits[..., :-1, :], (0, 0, 0, 1))
        grad_z = F.pad(logits[:, 1:, :, :] - logits[:, :-1, :, :], (0, 0, 0, 0, 0, 1))
        
        grad_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + grad_z**2 + 1e-8)
        
        # Combine distance and gradient for surface likelihood
        surface_score = grad_magnitude / (soft_dist + 0.1)
        
        # Soft thresholding
        soft_mask = torch.sigmoid(surface_score / self.temperature)
        
        return soft_mask
    
    def _ste_threshold(self, soft_mask: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Straight-Through Estimator: hard threshold in forward, soft in backward.
        """
        hard_mask = (soft_mask > threshold).float()
        # STE: use hard in forward, but gradient flows through soft
        return hard_mask.detach() + soft_mask - soft_mask.detach()


# # Convenience function for common use case
# def create_differentiable_vdm(
#     topk_mode: str = 'mean',
#     use_ste: bool = False,
#     temperature: float = 1.0
# ) -> Union[DifferentiableFlashVDMVolumeDecoding, DifferentiableFlashVDMWithSTE]:
#     """
#     Factory function to create differentiable VDM decoder.
    
#     Args:
#         topk_mode: 'mean' or 'merge' for attention processing
#         use_ste: If True, use STE version for end-to-end training
#         temperature: Temperature for soft thresholding (only used with STE)
    
#     Returns:
#         Differentiable VDM decoder instance
#     """
#     if use_ste:
#         return DifferentiableFlashVDMWithSTE(topk_mode=topk_mode, temperature=temperature)
#     return DifferentiableFlashVDMVolumeDecoding(topk_mode=topk_mode)


class VanillaVolumeDecoder:
    """整体 checkpoint 版本 - 更节省显存但重计算更多"""
    
    def _full_decode(
        self,
        latents: torch.Tensor,
        xyz_samples: torch.Tensor,
        geo_decoder: Callable,
        batch_size: int,
        num_chunks: int,
    ) -> torch.Tensor:
        """完整解码过程"""
        batch_logits = []
        for start in range(0, xyz_samples.shape[0], num_chunks):
            chunk_queries = xyz_samples[start: start + num_chunks, :]
            chunk_queries = repeat(chunk_queries, "p c -> b p c", b=batch_size)
            logits = geo_decoder(queries=chunk_queries, latents=latents)
            batch_logits.append(logits)
        
        return torch.cat(batch_logits, dim=1)
    
    def __call__(
        self,
        latents: torch.FloatTensor,
        geo_decoder: Callable,
        bounds: Union[Tuple[float], List[float], float] = 1.01,
        num_chunks: int = 10000,
        octree_resolution: int = None,
        enable_pbar: bool = True,
        use_checkpoint: bool = True,
        **kwargs,
    ):
        device = latents.device
        dtype = latents.dtype
        batch_size = latents.shape[0]

        if isinstance(bounds, float):
            bounds = [-bounds, -bounds, -bounds, bounds, bounds, bounds]

        bbox_min = torch.tensor(bounds[0:3], device=device, dtype=dtype)
        bbox_max = torch.tensor(bounds[3:6], device=device, dtype=dtype)
        
        xyz_samples, grid_size = generate_dense_grid_points(
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            octree_resolution=octree_resolution,
            indexing="ij"
        )
        
        xyz_samples = xyz_samples.contiguous().reshape(-1, 3)

        # 对整个解码过程做 checkpoint
        if use_checkpoint and latents.requires_grad:
            grid_logits = checkpoint.checkpoint(
                self._full_decode,
                latents,
                xyz_samples,
                geo_decoder,
                batch_size,
                num_chunks,
                use_reentrant=False,
            )
        else:
            grid_logits = self._full_decode(
                latents, xyz_samples, geo_decoder, batch_size, num_chunks
            )

        grid_logits = grid_logits.view((batch_size, *grid_size)).float()

        return grid_logits

class HierarchicalVolumeDecoding:
    @torch.no_grad()
    def __call__(
        self,
        latents: torch.FloatTensor,
        geo_decoder: Callable,
        bounds: Union[Tuple[float], List[float], float] = 1.01,
        num_chunks: int = 10000,
        mc_level: float = 0.0,
        octree_resolution: int = None,
        min_resolution: int = 63,
        enable_pbar: bool = True,
        **kwargs,
    ):
        device = latents.device
        dtype = latents.dtype

        resolutions = []
        if octree_resolution < min_resolution:
            resolutions.append(octree_resolution)
        while octree_resolution >= min_resolution:
            resolutions.append(octree_resolution)
            octree_resolution = octree_resolution // 2
        resolutions.reverse()

        # 1. generate query points
        if isinstance(bounds, float):
            bounds = [-bounds, -bounds, -bounds, bounds, bounds, bounds]
        bbox_min = np.array(bounds[0:3])
        bbox_max = np.array(bounds[3:6])
        bbox_size = bbox_max - bbox_min

        xyz_samples, grid_size, length = generate_dense_grid_points(
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            octree_resolution=resolutions[0],
            indexing="ij"
        )

        dilate = nn.Conv3d(1, 1, 3, padding=1, bias=False, device=device, dtype=dtype)
        dilate.weight = torch.nn.Parameter(torch.ones(dilate.weight.shape, dtype=dtype, device=device))

        grid_size = np.array(grid_size)
        xyz_samples = torch.from_numpy(xyz_samples).to(device, dtype=dtype).contiguous().reshape(-1, 3)

        # 2. latents to 3d volume
        batch_logits = []
        batch_size = latents.shape[0]
        for start in tqdm(range(0, xyz_samples.shape[0], num_chunks),
                          desc=f"Hierarchical Volume Decoding [r{resolutions[0] + 1}]"):
            queries = xyz_samples[start: start + num_chunks, :]
            batch_queries = repeat(queries, "p c -> b p c", b=batch_size)
            logits = geo_decoder(queries=batch_queries, latents=latents)
            batch_logits.append(logits)

        grid_logits = torch.cat(batch_logits, dim=1).view((batch_size, grid_size[0], grid_size[1], grid_size[2]))

        for octree_depth_now in resolutions[1:]:
            grid_size = np.array([octree_depth_now + 1] * 3)
            resolution = bbox_size / octree_depth_now
            next_index = torch.zeros(tuple(grid_size), dtype=dtype, device=device)
            next_logits = torch.full(next_index.shape, -10000., dtype=dtype, device=device)
            curr_points = extract_near_surface_volume_fn(grid_logits.squeeze(0), mc_level)
            curr_points += grid_logits.squeeze(0).abs() < 0.95

            if octree_depth_now == resolutions[-1]:
                expand_num = 0
            else:
                expand_num = 1
            for i in range(expand_num):
                curr_points = dilate(curr_points.unsqueeze(0).to(dtype)).squeeze(0)
            (cidx_x, cidx_y, cidx_z) = torch.where(curr_points > 0)
            next_index[cidx_x * 2, cidx_y * 2, cidx_z * 2] = 1
            for i in range(2 - expand_num):
                next_index = dilate(next_index.unsqueeze(0)).squeeze(0)
            nidx = torch.where(next_index > 0)

            next_points = torch.stack(nidx, dim=1)
            next_points = (next_points * torch.tensor(resolution, dtype=next_points.dtype, device=device) +
                           torch.tensor(bbox_min, dtype=next_points.dtype, device=device))
            batch_logits = []
            for start in tqdm(range(0, next_points.shape[0], num_chunks),
                              desc=f"Hierarchical Volume Decoding [r{octree_depth_now + 1}]"):
                queries = next_points[start: start + num_chunks, :]
                batch_queries = repeat(queries, "p c -> b p c", b=batch_size)
                logits = geo_decoder(queries=batch_queries.to(latents.dtype), latents=latents)
                batch_logits.append(logits)
            grid_logits = torch.cat(batch_logits, dim=1)
            next_logits[nidx] = grid_logits[0, ..., 0]
            grid_logits = next_logits.unsqueeze(0)
        grid_logits[grid_logits == -10000.] = float('nan')

        return grid_logits
    

