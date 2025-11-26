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

from typing import Union, Tuple, List

import numpy as np
import torch
from skimage import measure


class Latent2MeshOutput:
    def __init__(self, mesh_v=None, mesh_f=None):
        self.mesh_v = mesh_v
        self.mesh_f = mesh_f


def center_vertices(vertices):
    """Translate the vertices so that bounding box is centered at zero."""
    vert_min = vertices.min(dim=0)[0]
    vert_max = vertices.max(dim=0)[0]
    vert_center = 0.5 * (vert_min + vert_max)
    return vertices - vert_center


class SurfaceExtractor:
    def _compute_box_stat(self, bounds: Union[Tuple[float], List[float], float], octree_resolution: int):
        """
        Compute grid size, bounding box minimum coordinates, and bounding box size based on input 
        bounds and resolution.

        Args:
            bounds (Union[Tuple[float], List[float], float]): Bounding box coordinates or a single 
            float representing half side length.
                If float, bounds are assumed symmetric around zero in all axes.
                Expected format if list/tuple: [xmin, ymin, zmin, xmax, ymax, zmax].
            octree_resolution (int): Resolution of the octree grid.

        Returns:
            grid_size (List[int]): Grid size along each axis (x, y, z), each equal to octree_resolution + 1.
            bbox_min (np.ndarray): Minimum coordinates of the bounding box (xmin, ymin, zmin).
            bbox_size (np.ndarray): Size of the bounding box along each axis (xmax - xmin, etc.).
        """
        if isinstance(bounds, float):
            bounds = [-bounds, -bounds, -bounds, bounds, bounds, bounds]

        bbox_min, bbox_max = np.array(bounds[0:3]), np.array(bounds[3:6])
        bbox_size = bbox_max - bbox_min
        grid_size = [int(octree_resolution) + 1, int(octree_resolution) + 1, int(octree_resolution) + 1]
        return grid_size, bbox_min, bbox_size

    def run(self, *args, **kwargs):
        """
        Abstract method to extract surface mesh from grid logits.

        This method should be implemented by subclasses.

        Raises:
            NotImplementedError: Always, since this is an abstract method.
        """
        return NotImplementedError

    def __call__(self, grid_logits, requires_grad: bool = False, **kwargs):
        outputs = []
        for i in range(grid_logits.shape[0]):
            try:
                result = self.run(grid_logits[i], requires_grad=requires_grad, **kwargs)
                if requires_grad:
                    verts, faces = result
                    outputs.append((verts, faces))
                else:
                    # 🚫 非可微版本，numpy + 封装成 Latent2MeshOutput
                    vertices, faces = result
                    vertices = vertices.astype(np.float32)
                    faces = np.ascontiguousarray(faces)
                    outputs.append(Latent2MeshOutput(mesh_v=vertices, mesh_f=faces))

            except Exception:
                import traceback
                traceback.print_exc()
                outputs.append(None)

        return outputs


class MCSurfaceExtractor(SurfaceExtractor):
    def run(self, grid_logit, *, mc_level, bounds, octree_resolution, **kwargs):
        """
        Extract surface mesh using the Marching Cubes algorithm.

        Args:
            grid_logit (torch.Tensor): 3D grid logits tensor representing the scalar field.
            mc_level (float): The level (iso-value) at which to extract the surface.
            bounds (Union[Tuple[float], List[float], float]): Bounding box coordinates or half side length.
            octree_resolution (int): Resolution of the octree grid.
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple containing:
                - vertices (np.ndarray): Extracted mesh vertices, scaled and translated to bounding 
                  box coordinates.
                - faces (np.ndarray): Extracted mesh faces (triangles).
        """
        vertices, faces, normals, _ = measure.marching_cubes(grid_logit.cpu().numpy(),
                                                             mc_level,
                                                             method="lewiner")
        grid_size, bbox_min, bbox_size = self._compute_box_stat(bounds, octree_resolution)
        vertices = vertices / grid_size * bbox_size + bbox_min
        return vertices, faces


# class DMCSurfaceExtractor(SurfaceExtractor):
#     def run(self, grid_logit, *, octree_resolution, requires_grad, **kwargs):
#         device = grid_logit.device
#         if not hasattr(self, 'dmc'):
#             try:
#                 from diso import DiffDMC
#                 self.dmc = DiffDMC(dtype=torch.float32).to(device)
#             except:
#                 raise ImportError("Please install diso via `pip install diso`, or set mc_algo to 'mc'")
#         sdf = -grid_logit / octree_resolution
#         sdf = sdf.to(torch.float32).contiguous()
#         verts, faces = self.dmc(sdf, deform=None, return_quads=False, normalize=False)  # norm也要关掉

#         # verts = center_vertices(verts)  # 这里改变了 verts的值，也就是缩放了w 

#         if requires_grad:
#             # 保持可微（Torch Tensor）
#             idx = torch.arange(faces.shape[1]-1, -1, -1, device=faces.device)
#             faces = faces[:, idx]  # 通用反转，不假设面顶点数
#             return verts, faces
#         else:
#             vertices = verts.detach().cpu().numpy()
#             faces = faces.detach().cpu().numpy()[:, ::-1]
#         return vertices, faces

class DMCSurfaceExtractor_(SurfaceExtractor):
    def run(self, grid_logit, *, octree_resolution, requires_grad, **kwargs):
        device = grid_logit.device
        if not hasattr(self, 'dmc'):
            try:
                from diso import DiffDMC
                self.dmc = DiffDMC(dtype=torch.float32).to(device)
            except:
                raise ImportError("Please install diso via `pip install diso`, or set mc_algo to 'mc'")

        sdf = (-grid_logit / octree_resolution).to(torch.float32).contiguous()

        verts, faces = self.dmc(sdf, deform=None, return_quads=False, normalize=False)

        # Force contiguous always (safe for backward)
        verts = verts.contiguous()
        faces = faces.contiguous()

        if requires_grad:
            idx = torch.arange(faces.shape[1]-1, -1, -1, device=faces.device)
            faces = faces[:, idx].contiguous()
            return verts, faces
        else:
            vertices = verts.detach().cpu().numpy()
            faces = faces.detach().cpu().numpy()[:, ::-1]
            return vertices, faces


class DMCSurfaceExtractor(SurfaceExtractor):
    def run(self, grid_logit, *, octree_resolution, requires_grad, **kwargs):
        device = grid_logit.device
        
        # 初始化 FlexiCubes
        if not hasattr(self, 'flexicubes'):
            try:
                from kaolin.ops.conversions import FlexiCubes
                self.flexicubes = FlexiCubes(device=device)
            except:
                raise ImportError("Please install kaolin with FlexiCubes support")
        
        # 获取分辨率
        # grid_logit shape: (253, 253, 253) 表示 253 个顶点，对应 252 个体素
        assert grid_logit.shape[0] == grid_logit.shape[1] == grid_logit.shape[2], \
            "grid_logit must be cubic"
        
        grid_res = grid_logit.shape[0]  # 253
        voxel_res = grid_res - 1  # 252，实际的体素数量
        
        # 与 DMC 保持一致的 SDF 归一化
        # 这里的 octree_resolution 应该是指体素分辨率（如 252 或 256）
        sdf = (-grid_logit / octree_resolution).to(torch.float32).contiguous()  # 这里scale, 体素规格设定为 octree_resolution=256
        
        # 构建体素网格（首次调用时）
        if not hasattr(self, '_voxelgrid_cache') or self._voxelgrid_cache[0] != voxel_res:
            voxelgrid_vertices, cube_idx = self.flexicubes.construct_voxel_grid(voxel_res)
            # construct_voxel_grid 返回的顶点在 [-1, 1] 范围内
            # 顶点数量应该是 (voxel_res+1)^3 = 253^3
            self._voxelgrid_cache = (voxel_res, voxelgrid_vertices.to(device), cube_idx.to(device))
        
        _, voxelgrid_vertices, cube_idx = self._voxelgrid_cache
        
        # 将 3D 网格的 SDF 展平为 1D，与 voxelgrid_vertices 对应
        # voxelgrid_vertices 的顺序是：x 变化最快，然后 y，最后 z
        # 对应 grid_logit 需要按 (z, y, x) 顺序展平
        # scalar_field = sdf.permute(2, 1, 0).reshape(-1)  # (253*253*253,)
        
        # 或者，如果 grid_logit 已经是 (x, y, z) 顺序：
        scalar_field = sdf.reshape(-1)  # (253*253*253,)  # 这个顺序才对
        
        assert scalar_field.shape[0] == voxelgrid_vertices.shape[0], \
            f"SDF vertices {scalar_field.shape[0]} != grid vertices {voxelgrid_vertices.shape[0]}"
        
        # 提取网格
        if requires_grad:
            verts, faces, l_dev = self.flexicubes(
                voxelgrid_vertices=voxelgrid_vertices,
                scalar_field=scalar_field,
                cube_idx=cube_idx,
                resolution=voxel_res,
                training=True,
                output_tetmesh=False,
                weight_scale=0.99,
            )
            
            # 保持与 DMC 一致的面方向，flexicube 默认逆时针，法线朝外
            idx = torch.arange(faces.shape[1]-1, -1, -1, device=faces.device)
            faces = faces[:, idx]
            
            verts = verts.contiguous()
            faces = faces.contiguous()
            
            return verts, faces
        else:
            with torch.no_grad():
                verts, faces, l_dev = self.flexicubes(
                    voxelgrid_vertices=voxelgrid_vertices,
                    scalar_field=scalar_field,
                    cube_idx=cube_idx,
                    resolution=voxel_res,
                    training=False,
                    output_tetmesh=False,
                )
            
            # verts = center_vertices(verts).contiguous()  # 这里改变了 verts的值，也就是缩放了w 

            vertices = verts.detach().cpu().numpy()
            faces = faces.detach().cpu().numpy()[:, ::-1]
            
            return vertices, faces

import cv2 as cv
def save_debug_image(tensor, path, normalize=False):
        arr = tensor.detach().cpu().numpy()
        if normalize:
            arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)

        if arr.ndim == 2:
            arr = (arr * 255).astype(np.uint8)
            cv.imwrite(path, arr)
        else:
            arr = (arr * 255).astype(np.uint8)
            cv.imwrite(path, cv.cvtColor(arr, cv.COLOR_RGB2BGR))


SurfaceExtractors = {
    'mc': MCSurfaceExtractor,
    'dmc': DMCSurfaceExtractor,
}
