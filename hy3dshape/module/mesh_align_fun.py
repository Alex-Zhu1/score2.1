import copy
import numpy as np
import trimesh as tm
import pyvista as pv
from trimesh.registration import procrustes
from trimesh.proximity import closest_point
from tqdm import tqdm
from scipy.spatial import cKDTree

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
    scale = target_scale / source_scale if source_scale > 0 else 1.0
    # print(f'Initial scale: {scale}')
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
    return [np.diag(np.append(diag, 1))
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
        min_scale=0.3,
        max_scale=2.0,
        plot=False):
    cubes = [np.eye(4)]
    if test_reflections:
       cubes += get_all_axis_aligned_reflections()
    if test_rotations:
       cubes += get_all_axis_aligned_rotations()

    if isinstance(source_mesh, tm.PointCloud) or (not hasattr(source_mesh, 'faces') or len(source_mesh.faces) == 0):
        source_points = source_mesh.vertices
        count_source = len(source_points)
    else:
        source_points = tm.sample.sample_surface_even(source_mesh, count_source)[0]

    if isinstance(target_mesh, tm.PointCloud) or (not hasattr(target_mesh, 'faces') or len(target_mesh.faces) == 0):
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
    
    for cube in tqdm(cubes, total=len(cubes), ascii=True, disable=True):
        transform = cube
        best_cost = np.inf
        best_transform = transform.copy()
        cost_record = []
        p_dist_record = []

        for iter in tqdm(range(n_iter), ascii=True, total=n_iter, leave=False, disable=True):

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
       
            try:
                next_transform = procrustes(p_inlier, q_inlier, reflection=False, return_cost=False, scale=not fixed_scale)
            except np.linalg.LinAlgError:
                # SVD did not converge, skip this iteration
                continue

            transform = next_transform @ transform

            if not fixed_scale:
                scale = np.linalg.norm(transform[:3, 0])
                if scale > 0:
                    transform[:3, :3] /= scale
                    scale = np.clip(scale, min_scale, max_scale)
                    transform[:3, :3] *= scale

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

        pv_q = pv.PolyData(source_points)
        pv_q['scalars'] = np.zeros(len(source_points))
        plotter =pv.Plotter()
        plotter.background_color='#0D1017'
        
        if isinstance(target_mesh, tm.PointCloud):
            plotter.add_mesh(target_mesh.vertices)
        else:
            plotter.add_mesh(target_mesh, color=(0.5,0.5,0.7), ambient=0.2, specular=0.5)

        def cb(value):
            index = int(round(value))
            pv_q.points = all_p_dist_record[index][0]
            pv_q['scalars'] = all_p_dist_record[index][1]
            plotter.add_mesh(pv_q, name='pv_q', scalars='scalars', cmap='rainbow', show_scalar_bar=False)

        plotter.add_slider_widget(cb, [0, len(all_p_dist_record) -1], value=0, interaction_event='always')
        
        plotter.show()

    return best_of_all_transform, best_of_all_cost
        

def align_meshes(source_mesh_path=None, target_mesh_path=None, source_mesh=None,
                 transform_path=None, transformed_mesh_path=None, fixed_scale=False, outliers=0.2,
                 test_rotations=False, test_reflections=False, on_surface=False,
                 skip_coarse=True,
                 iterations_coarse=150, count_source_coarse=10_000, count_target_coarse=10_000,
                 iterations_fine=100, count_source_fine=10_000, count_target_fine=10_000,
                 min_scale=0.1, max_scale=3.0, plot=False):
    
    if source_mesh is None:
        source_mesh = tm.load(source_mesh_path, process=False, skip_materials=True)
    else:
        source_mesh = source_mesh
    target_mesh = tm.load(target_mesh_path, process=False, skip_materials=True)
    # x_source = copy.deepcopy(source_mesh)

    if isinstance(source_mesh, tm.Scene):
        source_mesh = tm.util.concatenate(source_mesh.dump())
    if isinstance(target_mesh, tm.Scene):
        target_mesh = tm.util.concatenate(target_mesh.dump())

    init_transform = compute_init_transform(source_mesh, target_mesh, fixed_scale)
    source_mesh.apply_transform(init_transform)

    skip_coarse = skip_coarse
    if skip_coarse:
        transform_coarse = np.eye(4)
    else:
        transform_coarse, _ = icp(source_mesh, target_mesh, 
                                    n_iter=iterations_coarse, count_source=count_source_coarse, count_target=count_target_coarse, 
                                    test_reflections=test_reflections, test_rotations=test_rotations, fixed_scale=True, 
                                    outliers=outliers, on_surface=on_surface,
                                    min_scale=min_scale, max_scale=max_scale, plot=plot)

        source_mesh.apply_transform(transform_coarse)
    if transformed_mesh_path is not None:
        source_mesh.export(transformed_mesh_path)

    transform_fine, _ = icp(source_mesh, target_mesh, 
                            n_iter=iterations_fine, count_source=count_source_fine, count_target=count_target_fine,
                            fixed_scale=True,
                            outliers=outliers, on_surface=on_surface, 
                            min_scale=0.7, max_scale=1.3, plot=plot)

    source_mesh.apply_transform(transform_fine)

    final_transform = transform_fine @ transform_coarse @ init_transform
    # print('Final Transform:\n', final_transform)
    # x_source.apply_transform(final_transform)
    if transform_path is not None:
        np.save(transform_path, final_transform)

    if transformed_mesh_path is not None:
        source_mesh.export(transformed_mesh_path)
        # x_source.export("hand_registered1.glb")

    return final_transform