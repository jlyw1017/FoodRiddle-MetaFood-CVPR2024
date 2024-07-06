"""Tools for working with meshes."""
import argparse
import os.path
from typing import Tuple

import numpy as np
import open3d as o3d
from scipy.spatial import ConvexHull, Delaunay, KDTree, Voronoi, cKDTree, distance
from scipy.optimize import least_squares
from sklearn.decomposition import PCA
from sklearn.linear_model import RANSACRegressor

from data_utils.colmap_reader import read_points3D_binary


FOOD_WITH_PLATE = [2, 3, 5, 6, 7, 8, 12, 14]


def read_mesh_file(file_path) -> o3d.geometry.TriangleMesh:
    """Reads a mesh file and returns an Open3D mesh object.

    Args:
        file_path (str): Path to the mesh file.

    Returns:
        o3d.geometry.TriangleMesh: Open3D mesh object.
    """
    mesh = o3d.io.read_triangle_mesh(file_path)
    return mesh


def extract_plane(points: np.ndarray, fit_twice: bool = False):
    """Extracts plane from point cloud.

    Args:
        points (np.ndarray): Points coordinates.
        fit_twice (bool): Whether to fix twice or not.
    """
    points = np.array(points)

    if fit_twice:
        ransac = RANSACRegressor(max_trials=1005, residual_threshold=0.1)
        ransac.fit(points[:, :2], points[:, 2])

        inlier_mask = ransac.inlier_mask_
        outlier_mask = np.logical_not(inlier_mask)

        pt2 = np.array(points)[outlier_mask]
        ransac_2 = RANSACRegressor(max_trials=100, residual_threshold=0.1)
        ransac_2.fit(pt2[:, :2], pt2[:, 2])

        inlier_mask = ransac.inlier_mask_
        outlier_mask = np.logical_not(inlier_mask)

        a, b = ransac_2.estimator_.coef_
        d = ransac_2.estimator_.intercept_
    else:
        ransac = RANSACRegressor(max_trials=100, residual_threshold=0.1)
        ransac.fit(points[:, :2], points[:, 2])
        inlier_mask = ransac.inlier_mask_
        outlier_mask = np.logical_not(inlier_mask)
        a, b = ransac.estimator_.coef_
        d = ransac.estimator_.intercept_

    normal_vector = np.array([a, b, -1])
    normal_vector = normal_vector / np.linalg.norm(normal_vector)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Adds color to plane.
    colors = np.zeros_like(points)
    colors[inlier_mask] = [0, 1, 0]
    colors[outlier_mask] = [1, 0, 0]
    pcd.colors = o3d.utility.Vector3dVector(colors)

    plane_size = 20
    plane_points = []
    for i in range(-plane_size, plane_size):
        for j in range(-plane_size, plane_size):
            x, y = i, j
            z = a * x + b * y + d
            plane_points.append([x, y, z])
    plane_points = np.array(plane_points)
    return normal_vector, plane_points


def find_basis_vectors(normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Finds basis vectors

    Args:
        normal (np.ndarray): the normal vector of a plane.
    """
    # Finds a vector which is
    if normal[0] == 0 and normal[1] == 0:
        tangent = np.array([1, 0, 0])
    else:
        tangent = np.array([-normal[1], normal[0], 0])
    tangent /= np.linalg.norm(tangent)

    # Uses Gram-Schmidt normalization to find the second basis.
    bitangent = np.cross(normal, tangent)
    bitangent /= np.linalg.norm(bitangent)
    return tangent, bitangent


def project_to_plane(points: np.ndarray, origin: np.ndarray,
                     axis_1: np.ndarray, axis_2: np.ndarray) -> np.ndarray:
    """Project points to the new coordinate.

    Args:
        points (np.ndarray): the points to be projected.
        origin (np.ndarray): the origin of the new coordinate.
        axis_1 (np.ndarray): an axis of the coordinate to project the points to.
        axis_2 (np.ndarray): another axis of the coordinate to project
         the points to.
    """
    points_relative = points - origin
    points_2D = np.array([
        np.dot(points_relative, axis_1),
        np.dot(points_relative, axis_2)]).T
    return points_2D


def project_back_to_3d(points_2d: np.ndarray, origin: np.ndarray,
                       axis_1: np.ndarray, axis_2: np.ndarray):
    """Project points to the new coordinate.

    Args:
        points_2d (np.ndarray): the points to be projected.
        origin (np.ndarray): the origin of the new coordinate.
        axis_1 (np.ndarray): an axis of the coordinate to project noto.
        axis_2 (np.ndarray): another axis of the coordinate to project onto.
    """
    points_3d = (points_2d[:, 0][:, np.newaxis] * axis_1 +
                 points_2d[:, 1][:, np.newaxis] * axis_2)
    points_3d += origin
    return points_3d


def project_points_to_plane(points: np.ndarray, plane_normal: np.ndarray,
                            plane_point: np.ndarray):
    """Projects the points to the plane.

    Args:
        points (np.ndarray): the points to be projected.
        plane_normal (np.ndarray): the plane normal.
        plane_point (np.ndarray): a point of the plane.
    """
    plane_normal = plane_normal / np.linalg.norm(plane_normal)
    projected_points = []
    for point in points:
        vector = point - plane_point
        projected_point = point - np.dot(vector, plane_normal) * plane_normal
        projected_points.append(projected_point)
    return np.array(projected_points)


def reverse_triangles(triangles: np.ndarray):
    """Reverses the direction of the triangles."""
    reversed_triangles = triangles.copy()
    # Swaps the second and the third points.
    reversed_triangles[:, [1, 2]] = reversed_triangles[:, [2, 1]]
    return reversed_triangles


def sort_points_counterclockwise(points: np.ndarray):
    """Sorts points in counterclockwise order.

    Parameters:
        points (numpy.ndarray): Points to be sorted.

    Returns:
        sorted_points (numpy.ndarray): Points sorted in counterclockwise order.
    """
    # Calculate the centroid of the points
    centroid = np.mean(points, axis=0)

    # Calculate the angle of each point with respect to the centroid
    angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])

    # Sort the points based on the angles in counterclockwise order
    sorted_indices = np.argsort(angles)
    sorted_points = points[sorted_indices]
    return sorted_points


def shrink_boundary_with_voronoi(boundary_points, shrink_factor=0.6):
    """Shrinks the boundary using Voronoi diagram.

    Parameters:
        boundary_points (numpy.ndarray): Boundary points.
        shrink_factor (float): Factor to control the shrinking amount (
        0 < shrink_factor < 1).

    Returns:
        shrunk_boundary_points (numpy.ndarray): Shrunk boundary points.
    """
    # Create Voronoi diagram from the boundary points
    vor = Voronoi(boundary_points)
    # Find the Voronoi vertices that lie inside the convex hull
    inside_vertices = []
    for vertex in vor.vertices:
        if vertex[0] < 1.5 and vertex[1] < 1:
            continue
        elif vertex[1] > 2:
            continue
        elif vertex[0] > 2:
            continue
        if vertex[0] < 1 and vertex[1] < 1.6:
            continue
        inside_vertices.append(vertex)

    # Create a KD-tree for efficient nearest-neighbor search
    tree = cKDTree(inside_vertices)

    # Shrink the boundary towards the nearest inside vertices
    shrunk_boundary_points = []
    for point in boundary_points:
        _, idx = tree.query(point)
        nearest_vertex = inside_vertices[idx]
        new_point = point + shrink_factor * (nearest_vertex - point)
        shrunk_boundary_points.append(new_point)
    shrunk_boundary_points = np.array(shrunk_boundary_points)
    return shrunk_boundary_points


def non_maximum_suppression(points, costs, distance_threshold):
    """Filters the given points.

    Args:
        points (np.ndarray): Points to filter.
        costs (np.ndarray): the costs of the points.
        distance_threshold (float): the distance threshold to determine the
        neighborhood of the points.
    """
    kdtree = KDTree(points)
    keep_indices = []

    for i, (point, cost) in enumerate(zip(points, costs)):
        indices = kdtree.query_ball_point(point, distance_threshold)
        if all(cost - costs[j] <= 0.01 for j in indices):
            keep_indices.append(i)

    # Keeps the points which are local maximum.
    filtered_points = np.array(points[keep_indices])
    filtered_costs = np.array(costs[keep_indices])
    return filtered_points, filtered_costs


def clip_mesh_by_plane(mesh, plane_origin, plane_normal):
    """Clips the bottom part of a mesh using a plane.

    Args:
        mesh (o3d.geometry.TriangleMesh): Input mesh.
        plane_origin (list or numpy.ndarray): Origin point of the plane.
        plane_normal (list or numpy.ndarray): Normal vector of the plane.

    Returns:
        clipped_mesh (o3d.geometry.TriangleMesh): Clipped mesh.
    """
    # Convert plane_origin and plane_normal to numpy array if not already
    plane_origin = np.array(plane_origin)
    plane_normal = np.array(plane_normal)

    # Calculate the distance from each vertex to the plane
    vertices = np.asarray(mesh.vertices)
    distances = np.dot(vertices - plane_origin, plane_normal)

    # Find vertices below the plane
    below_plane_indices = np.where(distances < 0)[0]

    # Create a mesh with only the vertices below the plane
    clipped_mesh = mesh.select_by_index(below_plane_indices)
    return clipped_mesh


def poisson_reconstruction_with_plane(
        combined_mesh, complement_pcd, output_mesh_path, if_sample: bool = False):
    """Reconstructs the mesh using a poisson reconstruction."""
    # Gets points from the mesh.
    if if_sample:
        combined_pcd = combined_mesh.sample_points_poisson_disk(
            number_of_points=10000)
    else:
        vertices = np.asarray(combined_mesh.vertices)
        combined_pcd = o3d.geometry.PointCloud()
        combined_pcd.points = o3d.utility.Vector3dVector(vertices)
        combined_pcd.normals = o3d.utility.Vector3dVector(
            combined_mesh.vertex_normals)
    o3d.visualization.draw_geometries([combined_mesh.sample_points_poisson_disk(
        number_of_points=10000)])

    combined_pcd = combined_pcd + complement_pcd
    combined_pcd.estimate_normals(
       search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30))

    # Performs poisson reconstruction.
    mesh_poisson, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        combined_pcd, depth=9)

    # Saves the mesh.
    o3d.io.write_triangle_mesh(output_mesh_path, mesh_poisson, write_ascii=True)
    o3d.visualization.draw_geometries([mesh_poisson, combined_mesh])
    return mesh_poisson


def fit_ellipse_or_circle(x, y):
    """Fits an ellipse or a circle to a set of 2D points based on their aspect ratio.

    Args:
        x (numpy.ndarray): x coordinates of the points.
        y (numpy.ndarray): y coordinates of the points.

    Returns:
        (params, is_circle): Parameters of the fitted ellipse or circle,
         and a flag indicating if it's a circle.
    """
    def ellipse_residuals(params, x, y):
        a, b, x0, y0, phi = params
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        x_rot = cos_phi * (x - x0) + sin_phi * (y - y0)
        y_rot = -sin_phi * (x - x0) + cos_phi * (y - y0)
        return (x_rot / a)**2 + (y_rot / b)**2 - 1

    def circle_residuals(params, x, y):
        r, x0, y0 = params
        return (x - x0)**2 + (y - y0)**2 - r**2

    x0, y0 = np.mean(x), np.mean(y)
    initial_params_ellipse = [np.std(x), np.std(y), x0, y0, 0]
    result_ellipse = least_squares(ellipse_residuals, initial_params_ellipse, args=(x, y))

    # Calculate aspect ratio
    a, b = result_ellipse.x[0], result_ellipse.x[1]
    aspect_ratio = max(a, b) / min(a, b)

    # If aspect ratio is too large, fit a circle instead
    if aspect_ratio > 3:
        initial_params_circle = [np.mean([a, b]), x0, y0]
        result_circle = least_squares(circle_residuals, initial_params_circle, args=(x, y))
        return result_circle.x, True
    else:
        return result_ellipse.x, False


def remove_outliers(points, threshold):
    """Removes outliers from an array of points based on the distance
     to other points.

    Args:
        points (numpy.ndarray): Array of points.
        threshold (float): Maximum distance threshold. Points farther than this
         threshold will be removed.

    Returns:
    - cleaned_points (numpy.ndarray): Array of points with outliers removed.
    """
    distances = distance.cdist(points, points)
    min_distances = distances.min(axis=1)
    cleaned_points = points[min_distances <= threshold]
    return cleaned_points


def remove_triangles_with_large_edges(mesh, threshold):
    """Removes triangles from a mesh if any of their edges are longer than a threshold.

    Parameters:
        mesh (o3d.geometry.TriangleMesh): Input mesh.
        threshold (float): Threshold for edge length. Triangles with any edge
        longer than this threshold will be removed.

    Returns:
        cleaned_mesh (o3d.geometry.TriangleMesh): Mesh with triangles removed.
    """
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    triangle_lengths = np.linalg.norm(
        vertices[triangles[:, 0]] - vertices[triangles[:, 1]], axis=1)
    triangle_lengths = np.maximum(triangle_lengths, np.linalg.norm(
        vertices[triangles[:, 1]] - vertices[triangles[:, 2]], axis=1))
    triangle_lengths = np.maximum(triangle_lengths, np.linalg.norm(
        vertices[triangles[:, 0]] - vertices[triangles[:, 2]], axis=1))

    indices_to_remove = np.where(triangle_lengths > threshold)[0]
    mesh.remove_triangles_by_index(indices_to_remove)
    mesh.remove_unreferenced_vertices()
    return mesh


def get_nearby_color(mesh_with_color, mesh_without_color):
    """Gets nearby colors from a mesh with color.

    Args:
        mesh_with_color (o3d.geometry): mesh with color.
        mesh_without_color (o3d.geometry): mesh without color.
    """
    colors_with_color = np.asarray(mesh_with_color.vertex_colors)
    vertices_without_color = np.asarray(mesh_without_color.vertices)

    # Uses the color from the nearest vertex of the colored mesh.
    colors_without_color = []
    kdtree = o3d.geometry.KDTreeFlann(mesh_with_color)
    for vertex in vertices_without_color:
        [_, idx, _] = kdtree.search_knn_vector_3d(vertex, 1)
        nearest_color = colors_with_color[idx[0]]
        colors_without_color.append(nearest_color)
    mesh_without_color.vertex_colors = o3d.utility.Vector3dVector(
        colors_without_color)
    return mesh_without_color


FACTOR_BY_IDX = {"1": (0.3, 0.97), "5": (0.7, 0.97)}


def complete_mesh_with_plane(mesh: o3d.geometry.TriangleMesh,
                             plane_normal: np.ndarray, plane_point: np.ndarray,
                             min_scale_factor: float, max_scale_factor: float,
                             use_sample: bool = True, debug: bool = False):
    """Completes mesh with the plane.

    Args:
        mesh (o3d.geometry): the mesh to complete.
        plane_normal (np.ndarray): the normal of the plane.
        plane_point (np.ndarray): a point of the plane.
        use_sample (bool): whether to use sampled.
        debug (bool): weather to show visual result.
    """
    if use_sample:
        pcd = mesh.sample_points_uniformly(number_of_points=1000)
        pcd = pcd.points
    else:
        pcd = mesh.vertices

    points = np.asarray(pcd)
    plane_normal = plane_normal / np.linalg.norm(plane_normal)
    distances = np.dot(points - plane_point, plane_normal)
    projection = points - distances[:, None] * plane_normal

    valid = np.where(np.abs(distances) < 0.2)
    projection = projection[valid]
    distances_ = distances[valid]

    projected_points, distances = non_maximum_suppression(
        projection, distances_, 0.1)

    tangent, bitangent = find_basis_vectors(plane_normal)
    points_2d = project_to_plane(projected_points, plane_point, tangent, bitangent)

    hull = ConvexHull(points_2d)
    hull_points_3d = projected_points[hull.vertices]
    hull_costs = np.abs(distances[hull.vertices])
    scale_factors = max_scale_factor - (max_scale_factor - min_scale_factor) * (
            hull_costs / hull_costs.max())

    scale_pts = hull_points_3d
    scaled_points = []
    center = np.mean(hull_points_3d, axis=0)
    for i, point in enumerate(scale_pts):
        scale_factor = scale_factors[i]
        scaled_point = center + scale_factor * (point - center)
        scaled_points.append(scaled_point)

    plane_border = scaled_points
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(plane_border)

    scaled_hull_points_2d = project_to_plane(
        np.array(plane_border), plane_point, tangent, bitangent)
    delaunay = Delaunay(scaled_hull_points_2d)
    reversed_triangles = reverse_triangles(delaunay.simplices)
    plane_mesh = o3d.geometry.TriangleMesh()
    plane_mesh.vertices = o3d.utility.Vector3dVector(plane_border)
    plane_mesh.triangles = o3d.utility.Vector3iVector(reversed_triangles)
    plane_mesh.compute_vertex_normals()

    combined_mesh = mesh + plane_mesh
    if debug:
        o3d.visualization.draw_geometries([mesh, point_cloud])
    return combined_mesh, plane_mesh


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mesh', type=str,
                        default='../demo_data/1/1.obj')
    parser.add_argument('-p', '--pts', type=str,
                        default='../demo_data/1/points3D.bin')
    parser.add_argument('-o', '--output', type=str,
                        default='../demo_data/output.obj')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_parser()
    reconstructed_mesh = o3d.io.read_triangle_mesh(args.mesh)
    index = os.path.splitext(os.path.split(args.mesh)[1])[0]

    pts = read_points3D_binary(args.pts)
    pt_coords = []
    for pt_id, pt in pts.items():
        pt_coords.append(pt.xyz)
    pt_coords = np.array(pt_coords)

    normal_vector, point_on_plane = extract_plane(
        pt_coords, index in FOOD_WITH_PLATE)

    MIN_SCALE_FACTOR, MAX_SCALE_FACTOR = FACTOR_BY_IDX[index]
    mesh, plane_mesh = complete_mesh_with_plane(
        reconstructed_mesh, normal_vector, point_on_plane[0], MIN_SCALE_FACTOR, MAX_SCALE_FACTOR)
    complement_pcd = plane_mesh.sample_points_poisson_disk(
        number_of_points=10000)

    mesh_poisson = poisson_reconstruction_with_plane(
        mesh, complement_pcd, args.output)
    mesh_poisson = get_nearby_color(reconstructed_mesh, mesh_poisson)
    o3d.visualization.draw_geometries([mesh_poisson])
