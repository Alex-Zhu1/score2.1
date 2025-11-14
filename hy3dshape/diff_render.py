import torch
import torch.nn.functional as F
import math
import nvdiffrast.torch as dr


class DifferentiableRenderer:
    """
    A fully differentiable renderer with OpenGL-style projection.
    Provides silhouette, normal, depth, and disparity maps.
    """

    def __init__(self, device="cuda"):
        self.device = device
        self.ctx = dr.RasterizeGLContext(device=device)

    # ===========================================================
    # 可微法线计算
    # ===========================================================
    def compute_vertex_normals(self, verts, faces_int32):
        faces = faces_int32.long()
        v0 = verts[faces[:, 0]]
        v1 = verts[faces[:, 1]]
        v2 = verts[faces[:, 2]]
        face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
        vert_normals = torch.zeros_like(verts)
        for i in range(3):
            vert_normals.index_add_(0, faces[:, i], face_normals)
        vert_normals = F.normalize(vert_normals, dim=-1, eps=1e-8)
        return vert_normals

    # ===========================================================
    # 可微相机投影 (OpenGL)
    # ===========================================================
    def camera_to_clip(self, verts_cam, Kp, near, far):
        fxp, fyp = Kp[0, 0], Kp[1, 1]
        cxp, cyp = Kp[0, 2], Kp[1, 2]
        X, Y = verts_cam[:, 0], verts_cam[:, 1]
        front_z = (-verts_cam[:, 2]).clamp(min=1e-6)
        w = front_z
        x_c = 2.0 * fxp * X + (2.0 * cxp - 1.0) * w
        y_c = -2.0 * fyp * Y + (1.0 - 2.0 * cyp) * w
        A = (far + near) / (far - near)
        B = (-2.0 * near * far) / (far - near)
        z_c = A * w + B
        return torch.stack([x_c, y_c, z_c, w], dim=-1)

    # ===========================================================
    # FOV -> 归一化内参
    # ===========================================================
    def fov_to_K_normalized(self, fov_x_deg, width, height):
        fov_x = math.radians(fov_x_deg)
        fxp = 1.0 / (2.0 * math.tan(fov_x / 2.0))
        fyp = fxp * (width / height)
        cxp, cyp = 0.5, 0.5
        Kp = torch.tensor([[fxp, 0, cxp], [0, fyp, cyp], [0, 0, 1]], dtype=torch.float32)
        return Kp.to(self.device)

    # ===========================================================
    # 主渲染函数
    # ===========================================================
    def render_normals_disparity_silhouette(
        self,
        verts_cam,
        faces,
        Kp,
        H,
        W,
        near=None,
        far=None,
        smooth_mask=True,
    ):
        device = self.device
        verts_cam = verts_cam.to(device)
        faces = faces.to(device)
        Kp = Kp.to(device)

        # 深度裁剪
        Z = verts_cam[:, 2]
        front_z = (-Z).clamp(min=1e-6)
        if near is None:
            near = front_z.min() * 0.5
        if far is None:
            far = front_z.max() * 1.5
        near, far = float(near), float(far)

        # 顶点投影到裁剪空间
        verts_clip = self.camera_to_clip(verts_cam, Kp, near, far).unsqueeze(0)
        rast, _ = dr.rasterize(self.ctx, verts_clip, faces, (H, W))

        # silhouette
        if smooth_mask:
            sil = torch.sigmoid(500.0 * rast[..., 3:])
        else:
            sil = (rast[..., 3:] > 0).float()

        # 顶点属性：法线 + 逆深度
        vnorm = self.compute_vertex_normals(verts_cam, faces)
        inv_front_z = (1.0 / front_z).unsqueeze(-1)
        attr = torch.cat([vnorm, inv_front_z], dim=-1)[None, ...]

        # 插值到像素
        attr_img, _ = dr.interpolate(attr, rast, faces)
        n_img = F.normalize(attr_img[..., :3], dim=-1, eps=1e-8)
        inv_depth = torch.maximum(attr_img[..., 3:4], torch.tensor(1e-8, device=device))
        depth = 1.0 / inv_depth
        disparity = inv_depth

        n_vis = (n_img * 0.5 + 0.5) * sil
        depth = depth * sil
        disparity = disparity * sil

        return {
            "silhouette": sil[0],
            "depth": depth[0],
            "disparity": disparity[0],
            "normal_rgb": n_vis[0],
            "normal_cam": n_img[0],
            "near_far": (near, far),
        }
