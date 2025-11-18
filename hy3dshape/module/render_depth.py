import os
import math
import torch
import torch.nn.functional as F
import nvdiffrast.torch as dr
import trimesh
import numpy as np
import cv2 as cv

# ===========================================================
# 计算顶点法线（camera/view space）
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
# 相机 (K') + near/far -> clip-space
# 约定：输入 verts_cam 是 camera/view space（相机朝 -Z，OpenGL 约定）
# front_z = -Z_view 为正深度 (前方)
# ===========================================================
def camera_to_clip(verts_cam, Kp, near, far):
    fxp, fyp = Kp[0, 0].item(), Kp[1, 1].item()
    cxp, cyp = Kp[0, 2].item(), Kp[1, 2].item()

    X = verts_cam[:, 0]
    Y = verts_cam[:, 1]
    front_z = (-verts_cam[:, 2]).clamp(min=1e-6)   # 正深度, >0 for visible verts

    # Keep w proportional to front_z (standard)
    w = front_z
    x_c = 2.0 * fxp * X + (2.0 * cxp - 1.0) * w
    y_c = -2.0 * fyp * Y + (1.0 - 2.0 * cyp) * w

    A = (far + near) / (far - near)
    B = (-2.0 * near * far) / (far - near)
    z_c = A * w + B

    return torch.stack([x_c, y_c, z_c, w], dim=-1)

# ===========================================================
# 渲染法线 / depth / disparity / silhouette
# 重要：attr 的第4通道传入 inv_front_z = 1 / front_z
# 插值后 inv_depth = attr_img[...,3]；depth = 1.0 / inv_depth；disparity = inv_depth
# ===========================================================
def render_normals_disparity_silhouette(
    verts_cam, faces, Kp, H, W, near=None, far=None, antialias=True, device="cuda"
):
    device = torch.device(device)
    verts_cam = verts_cam.to(device=device, dtype=torch.float32).contiguous()
    faces = faces.to(device=device, dtype=torch.int32).contiguous()
    Kp = Kp.to(device=device, dtype=torch.float32)

    # front_z 判定（-Z_view）
    Z_view = verts_cam[:, 2]
    front_mask = (-Z_view) > 1e-6
    front_z_vals = (-Z_view)[front_mask]
    if front_z_vals.numel() == 0:
        raise ValueError("All vertices are behind the camera (no -Z > 0).")

    # near/far 基于 front_z
    if near is None:
        near = max(1e-3, float(front_z_vals.min().item() * 0.5))
    if far is None:
        far = float(front_z_vals.max().item() * 1.5)
    near = float(near)
    far = float(far)
    if not (near > 0 and far > near):
        raise ValueError(f"Invalid near/far computed: near={near}, far={far}")
    print(f"[Debug] near={near:.6f}, far={far:.6f} (based on front_z)")

    # clip coords & rasterize
    verts_clip = camera_to_clip(verts_cam, Kp, near, far).contiguous().unsqueeze(0)
    ctx = dr.RasterizeGLContext(device=device)
    rast, _ = dr.rasterize(ctx, verts_clip, faces, (H, W))

    # silhouette (antialias 可选)
    sil = (rast[..., 3:] > 0).float()
    if antialias:
        sil = dr.antialias(sil, rast, verts_clip, faces)

    # attributes: normals + inv_front_z（与 w 成反比）
    vnorm = compute_vertex_normals(verts_cam, faces).contiguous()  # [V,3]
    front_z = (-verts_cam[:, 2]).clamp(min=1e-6)                 # [V]
    inv_front_z = (1.0 / front_z).unsqueeze(-1)                 # [V,1]
    attr = torch.cat([vnorm, inv_front_z], dim=-1)[None, ...].contiguous()  # [1,V,4]

    # interpolate attributes (透视校正插值)
    attr_img, _ = dr.interpolate(attr, rast, faces)  # [1,H,W,4]
    n_img = F.normalize(attr_img[..., :3], dim=-1, eps=1e-8)

    # 插值后得到 inv_depth（即 disparity），再反求 depth
    inv_depth = attr_img[..., 3:4].clamp(min=1e-8)  # avoid div0
    depth = (1.0 / inv_depth)
    disparity = inv_depth  # 已经是 1/depth 的量

    # apply silhouette mask
    n_vis = (n_img * 0.5 + 0.5).clamp(0, 1) * sil
    depth = depth * sil
    disparity = disparity * sil

    # debug print ranges
    dmin = float(depth.min().item()) if depth.numel() else 0.0
    dmax = float(depth.max().item()) if depth.numel() else 0.0
    imin = float(disparity.min().item()) if disparity.numel() else 0.0
    imax = float(disparity.max().item()) if disparity.numel() else 0.0
    print(f"[Debug] depth min/max = {dmin:.6e} / {dmax:.6e}")
    print(f"[Debug] disparity min/max = {imin:.6e} / {imax:.6e}")

    outputs = {
        "silhouette": sil[0],        # [H,W,1]
        "disparity": disparity[0],   # [H,W,1]
        "depth": depth[0],           # [H,W,1]
        "normal_rgb": n_vis[0],      # [H,W,3]
        "normal_cam": n_img[0],      # [H,W,3]
        "near_far": (near, far),
    }
    print(f"[Debug] Silhouette sum: {outputs['silhouette'].sum().item():.2f}")
    return outputs

# ===========================================================
# FOV -> 归一化内参（保持你的原实现）
# ===========================================================
def fov_to_K_normalized(fov_x_deg, width, height):
    fov_x = math.radians(fov_x_deg)
    fxp = 1.0 / (2.0 * math.tan(fov_x / 2.0))
    fyp = fxp * (width / height)
    cxp, cyp = 0.5, 0.5
    Kp = torch.tensor([[fxp, 0, cxp], [0, fyp, cyp], [0, 0, 1]], dtype=torch.float32)
    return Kp

# ===========================================================
# 主入口：加载 mesh，渲染并保存 depth/disparity 可视化
# ===========================================================
if __name__ == "__main__":
    mesh_path = "/home/haiming.zhu/hoi/InvScore/hunyuan_registered.glb"
    scene_or_mesh = trimesh.load(mesh_path, process=False)

    if isinstance(scene_or_mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(scene_or_mesh.dump())
        print(f"[Info] Loaded Scene with {len(scene_or_mesh.geometry)} submeshes.")
    else:
        mesh = scene_or_mesh
        print("[Info] Loaded single mesh.")

    verts = torch.from_numpy(mesh.vertices).float()
    faces = torch.from_numpy(mesh.faces).int()

    print("[Debug] Verts range:", verts.min(0).values, verts.max(0).values)
    print("[Debug] Z min/max (view-space):", verts[:, 2].min().item(), verts[:, 2].max().item())

    # params
    H, W = 224, 224
    fovx = 41.039776
    Kp = fov_to_K_normalized(fovx, W, H)

    outputs = render_normals_disparity_silhouette(verts, faces, Kp, H, W, antialias=True, device="cuda")

    depth = outputs["depth"].cpu().numpy()       # [H,W,1]
    disp = outputs["disparity"].cpu().numpy()
    normal_rgb = outputs["normal_rgb"].cpu().numpy()
    sil = outputs["silhouette"].cpu().numpy()

    os.makedirs("./debug", exist_ok=True)

    # depth 线性可视化：用 near/far 归一化为 [0,1]
    near, far = outputs["near_far"]
    depth_vis = np.clip((depth - near) / (far - near), 0.0, 1.0)
    depth_vis_u8 = (depth_vis[..., 0] * 255.0).astype(np.uint8)
    cv.imwrite("./debug/depth.png", depth_vis_u8)
    depth_color = cv.applyColorMap(depth_vis_u8, cv.COLORMAP_VIRIDIS)
    cv.imwrite("./debug/depth_colormap.png", depth_color)

    # disparity 可视化（归一化到 [0,1]）
    if disp.max() > 0:
        disp_vis = (disp / (disp.max() + 1e-8))[..., 0]
        cv.imwrite("./debug/disparity.png", (disp_vis * 255).astype(np.uint8))
    else:
        cv.imwrite("./debug/disparity.png", np.zeros((H, W), dtype=np.uint8))

    # normal & silhouette
    cv.imwrite("./debug/silhouette.png", (sil[..., 0] * 255).astype(np.uint8))
    cv.imwrite("./debug/normal_rgb.png", (normal_rgb * 255).astype(np.uint8))

    print("[Info] Saved results to ./debug/: depth.png, depth_colormap.png, disparity.png, normal_rgb.png, silhouette.png")
