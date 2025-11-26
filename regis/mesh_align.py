import click
import numpy as np
import trimesh as tm
import pyvista as pv
from trimesh.registration import procrustes
from trimesh.proximity import closest_point
from tqdm import tqdm
from scipy.spatial import cKDTree

# def get_centroid_scale(mesh_or_pointcloud):
#     if isinstance(mesh_or_pointcloud, tm.PointCloud):
#         source_centroid = mesh_or_pointcloud.vertices.mean(axis=0)
#         source_scale = np.linalg.norm(mesh_or_pointcloud.vertices.max(axis=0) - mesh_or_pointcloud.vertices.min(axis=0))
#         return source_centroid, source_scale
#     return mesh_or_pointcloud.centroid, mesh_or_pointcloud.scale

def get_centroid_scale(mesh_or_pointcloud):
    if isinstance(mesh_or_pointcloud, tm.PointCloud):
        verts = mesh_or_pointcloud.vertices
    else:
        verts = mesh_or_pointcloud.vertices
    if len(verts) == 0:
        return np.zeros(3), 1.0
    centroid = verts.mean(axis=0)
    if len(verts) <= 1:
        return centroid, 1.0
    diffs = verts - centroid
    rms = np.sqrt(np.mean(np.sum(diffs**2, axis=1)))
    return centroid, rms

def compute_init_transform(source_mesh, target_mesh, fixed_scale):
    source_centroid, source_scale = get_centroid_scale(source_mesh)
    target_centroid, target_scale = get_centroid_scale(target_mesh)

    translation = target_centroid - source_centroid
    scale = target_scale / source_scale
    T = tm.transformations.translation_matrix(translation)
    if fixed_scale:
        return T
    S = tm.transformations.scale_matrix(scale, origin=source_centroid)
    return T @ S

def get_all_axis_aligned_rotations():
    rotations = []
    for coord in range(3):
        axis= np.zeros(3)
        axis[coord] = 1
        for angle in [-np.pi/2, np.pi, np.pi/2]:
            rotations.append(tm.transformations.rotation_matrix(angle, axis))
    return rotations

def get_all_axis_aligned_reflections():
    return [np.eye(4) * np.append(diag, 1)
                        for diag in [[1, 1, -1],
                                    [1, -1, 1],
                                    [-1, 1, 1],
                                    [-1, -1, 1],
                                    [-1, 1, -1],
                                    [1, -1, -1],
                                    [-1, -1, -1]]]

def icp(source_mesh,
        target_mesh,
        n_iter, 
        count_source=5_000,
        count_target=20_000,
        test_reflections=False,
        test_rotations=False,
        fixed_scale=False,
        outliers=0,
        on_surface=False,
        min_scale=0.5,
        max_scale=2.0,
        plot=False):
    cubes = [np.eye(4)]
    if test_reflections:
       cubes += get_all_axis_aligned_reflections()
    if test_rotations:
       cubes += get_all_axis_aligned_rotations()

    if isinstance(source_mesh, tm.PointCloud):
        source_points = source_mesh.vertices
        count_source = len(source_points)
    else:
        source_points = tm.sample.sample_surface_even(source_mesh, count_source)[0]

    if isinstance(target_mesh, tm.PointCloud):
        target_points = target_mesh.vertices
        count_target = len(target_points)
    else:
        target_points = tm.sample.sample_surface_even(target_mesh, count_target)[0]
    
    n_outliers = int(outliers*count_source)

    kdtree = cKDTree(target_points)
    best_cost_record = []
    best_p_dist_record = []
    all_cost_record = []
    all_p_dist_record = []
    best_of_all_cost = np.inf
    best_of_all_transform = np.eye(4)
    
    for cube in tqdm(cubes, total=len(cubes), ascii=True):
        transform = cube
        best_cost = np.inf
        best_transform = transform.copy()
        cost_record = []
        p_dist_record = []

        for iter in tqdm(range(n_iter), ascii=True, total=n_iter, leave=False):

            p = tm.transform_points(source_points, transform)

            if on_surface:
                q, dist = closest_point(target_mesh, p)[:2]
            else:
                dist, qi = kdtree.query(p)
                q = target_points[qi]

            if n_outliers > 0:
                sorted_dist_indices =np.argsort(dist)
                dist[sorted_dist_indices[-n_outliers:]] = 1
                inlier_indices = sorted_dist_indices[:-n_outliers]
                cost = dist[inlier_indices].mean()
                p_inlier = p[inlier_indices]
                q_inlier = q[inlier_indices]

            else:
                p_inlier = p
                q_inlier = q
                cost = dist.mean()
       
            next_transform = procrustes(p_inlier, q_inlier, reflection=False, return_cost=False, scale=not fixed_scale)

            transform = next_transform @ transform

            if not fixed_scale:
                # scale = np.linalg.norm(transform[:3, 0])
                # transform[:3, :3] /= scale
                # scale = np.clip(scale, min_scale, max_scale)
                # transform[:3, :3] *= scale
                # 更稳健的做法：使用SVD分解
                U, S, Vt = np.linalg.svd(transform[:3, :3])
                S = np.clip(S, min_scale, max_scale)
                transform[:3, :3] = U @ np.diag(S) @ Vt

            p_dist_record.append((p, dist))
            cost_record.append(cost)

            if cost < best_cost:
                best_cost = cost
                best_transform = transform
        
        all_cost_record += cost_record
        all_p_dist_record += p_dist_record

        if best_cost < best_of_all_cost:
            best_of_all_cost = best_cost
            best_of_all_transform = best_transform
            best_cost_record = cost_record
            best_p_dist_record = p_dist_record

    if plot:
        import imageio
        pv_q = pv.PolyData(source_points)
        pv_q['scalars'] = np.zeros(len(source_points))
        plotter = pv.Plotter(off_screen=True)  # 离屏渲染
        plotter.background_color='#0D1017'

        if isinstance(target_mesh, tm.PointCloud):
            plotter.add_mesh(target_mesh.vertices)
        else:
            plotter.add_mesh(target_mesh, color=(0.5,0.5,0.7), ambient=0.2, specular=0.5)

        frames = []
        for i, (points, scalars) in enumerate(all_p_dist_record):
            pv_q.points = points
            pv_q['scalars'] = scalars
            if 'pv_q' in plotter.meshes:
                plotter.remove_actor('pv_q')
            plotter.add_mesh(pv_q, name='pv_q', scalars='scalars', cmap='rainbow', show_scalar_bar=False)

            # 渲染当前帧到 numpy 图像
            img = plotter.screenshot(None, return_img=True)
            frames.append(img)

        # 保存视频
        imageio.mimsave('output_video.mp4', frames, fps=10)


    return best_of_all_transform, best_of_all_cost
        

@click.command()
@click.argument('source_mesh_path', type=click.Path(exists=True), required=True)
@click.argument('target_mesh_path', type=click.Path(exists=True), required=True)
@click.option('-tp', '--transform_path', type=str, default=None, required=False, help='Path to write 4x4 transform matrix')
@click.option('-tmp', '--transformed_mesh_path', type=str, default=None, required=False, help='Path to write transformed mesh')
@click.option('-fs', '--fixed_scale', is_flag=True, help='If present, set the scale of the source mesh fixed')
@click.option('-o', '--outliers', type=float, default=0.2, help='Ratio of expected outliers (main parameter to tweak)')
@click.option('-trot', '--test_rotations', is_flag=True, help='If present, test the rotations in coarse ICP phase')
@click.option('-tref', '--test_reflections', is_flag=True, help='If present, test the reflections in coarse ICP phase')
@click.option('-os', '--on_surface', is_flag=True, help='If present, use trimesh.proximity.closest_point instead of scipy\'s KdTree (slightly more accurate but a lot slower, not recommended)')
@click.option('-ir', '--iterations_coarse', type=int, default=50, help='Number of iterations for coarse ICP phase')
@click.option('-csr', '--count_source_coarse', type=int, default=1_000, help='Number of points on the source mesh for coarse ICP phase')
@click.option('-ctr', '--count_target_coarse', type=int, default=5_000, help='Number of points on the target mesh for coarse ICP phase')
@click.option('-if', '--iterations_fine', type=int, default=100, help='Number of iterations for fine ICP phase')
@click.option('-csf', '--count_source_fine', type=int, default=7_000, help='Number of points on the source mesh for fine ICP phase')
@click.option('-ctf', '--count_target_fine', type=int, default=10_000, help='Number of points on the target mesh for fine ICP phase')
@click.option('-mis', '--min_scale', type=float, default=0.7, help='Minimum scaling factor (to prevent the source to find a local minima by shrinking too much)')
@click.option('-mas', '--max_scale', type=float, default=1.3, help='Maximum scaling factor (to prevent the source to find a local minima by enlarging too much)')
@click.option('-p', '--plot', is_flag=True, help='If present, plot the registration steps with Pyvista')
def align_meshes(source_mesh_path, target_mesh_path, transform_path, transformed_mesh_path, fixed_scale, outliers,
                 test_rotations, test_reflections, on_surface,
                 iterations_coarse, count_source_coarse, count_target_coarse,
                 iterations_fine, count_source_fine, count_target_fine,
                 min_scale, max_scale, plot):

    source_mesh = tm.load(source_mesh_path, process=False, skip_materials=True)
    target_mesh = tm.load(target_mesh_path, process=False, skip_materials=True)

    if isinstance(source_mesh, tm.Scene):
        source_mesh = tm.util.concatenate(source_mesh.dump())
    if isinstance(source_mesh, tm.Scene):
        source_mesh = tm.util.concatenate(source_mesh.dump())
    # To = np.array([
    #         [ 0.33563303, -0.0569792 ,  0.13968643, -0.01187149],
    #         [ 0.15039468,  0.1531337 , -0.29889793,  0.02222771],
    #         [-0.01184779,  0.32971488,  0.1629607 , -0.90017929],
    #         [ 0.        ,  0.        ,  0.        ,  1.        ]
    #     ])

    # source_mesh.apply_transform(To)


    init_transform = compute_init_transform(source_mesh, target_mesh, fixed_scale)
    source_mesh.apply_transform(init_transform)

    transform_coarse, _ = icp(source_mesh, target_mesh, 
                                   n_iter=iterations_coarse, count_source=count_source_coarse, count_target=count_target_coarse, 
                                   test_reflections=test_reflections, test_rotations=test_rotations, fixed_scale=True, 
                                   outliers=outliers, on_surface=on_surface,
                                   min_scale=min_scale, max_scale=max_scale, plot=plot)

    source_mesh.apply_transform(transform_coarse)

    transform_fine, _ = icp(source_mesh, target_mesh, 
                            n_iter=iterations_fine, count_source=count_source_fine, count_target=count_target_fine,
                            outliers=outliers, on_surface=on_surface,
                            min_scale=0.95, max_scale=1.15, plot=False)

    source_mesh.apply_transform(transform_fine)

    final_transform = transform_fine @ transform_coarse @ init_transform

    if transform_path is not None:
        print(final_transform)
        np.save(transform_path, final_transform)

    if transformed_mesh_path is not None:
        source_mesh.export(transformed_mesh_path)

if __name__ == '__main__':
    align_meshes()