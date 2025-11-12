import os
import math
import torch
import torch.nn.functional as F
import nvdiffrast.torch as dr
import trimesh
import numpy as np
import cv2 as cv

# ===========================================================
# 可微法线计算
# ===========================================================
def compute_vertex_normals(verts, faces_int32):
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
def camera_to_clip(verts_cam, Kp, near, far):
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
# 可微渲染主函数
# ===========================================================
def render_normals_disparity_silhouette_differentiable(
    verts_cam, faces, Kp, H, W, near=None, far=None, smooth_mask=True, device="cuda"
):
    verts_cam = verts_cam.to(device)
    faces = faces.to(device)
    Kp = Kp.to(device)

    Z = verts_cam[:, 2]
    front_z = (-Z).clamp(min=1e-6)
    if near is None:
        near = front_z.min() * 0.5
    if far is None:
        far = front_z.max() * 1.5
    near, far = float(near), float(far)

    verts_clip = camera_to_clip(verts_cam, Kp, near, far).unsqueeze(0)  # verts是点集合，而face是点索引，所以不需要对face进行transform
    ctx = dr.RasterizeGLContext(device=device)
    rast, _ = dr.rasterize(ctx, verts_clip, faces, (H, W))

    if smooth_mask:  # silhouette， 类似与二值mask，即这个像素是否被物体占据
        sil = torch.sigmoid(500.0 * rast[..., 3:])
    else:
        sil = (rast[..., 3:] > 0).float()

    vnorm = compute_vertex_normals(verts_cam, faces)  # 计算每个顶点的法线向量
    inv_front_z = (1.0 / front_z).unsqueeze(-1)  # 计算每个顶点的逆深度值
    attr = torch.cat([vnorm, inv_front_z], dim=-1)[None, ...]

    '''nvdiffrast 会：
        对每个像素找到它属于哪个三角形；
        根据 rast 提供的重心坐标；
        对该三角形三个顶点的属性加权求和（透视校正插值）；
        输出该像素的属性 attr_img[y, x, :]。

        于是：
        attr_img[..., :3] → 每像素的法线；
        attr_img[..., 3] → 每像素的逆深度。像素的逆深度是线性的。
        '''
    attr_img, _ = dr.interpolate(attr, rast, faces)


    n_img = F.normalize(attr_img[..., :3], dim=-1, eps=1e-8)  # 由于插值（这里是因为顶点的法线，转化到像素中心的发现，这个插值是位于球面的，但重心插值又是线性的），像素的法线不是单位向量，需要归一化，数值范围是 ∈[−1,1]^3
    inv_depth = torch.maximum(attr_img[..., 3:4], torch.tensor(1e-8, device=device))  # 逆深度过小，说明是背景，所以用一个很小的值代替，避免深度计算除零
    depth = 1.0 / inv_depth
    disparity = inv_depth

    n_vis = (n_img * 0.5 + 0.5) * sil  # 法线可视化，从[-1, 1]映射到[0,1]范围内，
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

# ===========================================================
# FOV -> 归一化内参
# ===========================================================
def fov_to_K_normalized(fov_x_deg, width, height):
    fov_x = math.radians(fov_x_deg)
    fxp = 1.0 / (2.0 * math.tan(fov_x / 2.0))
    fyp = fxp * (width / height)
    cxp, cyp = 0.5, 0.5
    Kp = torch.tensor([[fxp, 0, cxp], [0, fyp, cyp], [0, 0, 1]], dtype=torch.float32)
    return Kp

# ===========================================================
# 可视化辅助函数 (不破坏计算图)
# ===========================================================
def visualize_outputs(outputs, step, save_dir="./debug"):
    os.makedirs(save_dir, exist_ok=True)
    with torch.no_grad():
        depth = outputs["depth"].detach().cpu().numpy()
        disp = outputs["disparity"].detach().cpu().numpy()
        normal_rgb = outputs["normal_rgb"].detach().cpu().numpy()
        sil = outputs["silhouette"].detach().cpu().numpy()
        near, far = outputs["near_far"]

        depth_vis = np.clip((depth - near) / (far - near), 0.0, 1.0)
        depth_vis_u8 = (depth_vis[..., 0] * 255.0).astype(np.uint8)
        depth_color = cv.applyColorMap(depth_vis_u8, cv.COLORMAP_VIRIDIS)
        cv.imwrite(f"{save_dir}/depth_{step:04d}.png", depth_color)

        if disp.max() > 0:
            disp_vis = (disp / (disp.max() + 1e-8))[..., 0]
            cv.imwrite(f"{save_dir}/disp_{step:04d}.png", (disp_vis * 255).astype(np.uint8))

        cv.imwrite(f"{save_dir}/normal_{step:04d}.png", (normal_rgb * 255).astype(np.uint8))
        cv.imwrite(f"{save_dir}/sil_{step:04d}.png", (sil[..., 0] * 255).astype(np.uint8))

# ===========================================================
# 示例：可微优化 + 可视化
# ===========================================================
if __name__ == "__main__":
    mesh = trimesh.load("/home/haiming.zhu/hoi/InvScore/hunyuan_registered.glb", process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(mesh.dump())
    verts = torch.from_numpy(mesh.vertices).float().cuda()
    faces = torch.from_numpy(mesh.faces).int().cuda()
    verts.requires_grad_(True)

    H, W = 224, 224
    fovx = 41.0
    Kp = fov_to_K_normalized(fovx, W, H).cuda()
    optimizer = torch.optim.Adam([verts], lr=1e-3)

    for step in range(0, 100):
        optimizer.zero_grad()

        outputs = render_normals_disparity_silhouette_differentiable(
            verts, faces, Kp, H, W, smooth_mask=True, device="cuda"
        )

        # example: 深度约束损失（鼓励平滑）
        loss_depth = outputs["depth"].mean()
        loss_normal = (1 - outputs["normal_rgb"].mean())
        loss = loss_depth + 0.1 * loss_normal
        loss.backward()
        optimizer.step()

        print(f"[Step {step}] Loss={loss.item():.6f}, grad_norm={verts.grad.norm().item():.6f}")

        # 每隔几步保存可视化
        if step % 10 == 0:
            visualize_outputs(outputs, step)
