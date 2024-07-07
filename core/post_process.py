"""This module contains code for post process of mesh from generative modules."""
import argparse
import os

import numpy as np
import open3d as o3d
from sklearn.linear_model import RANSACRegressor


def compute_bounding_box_center_scale(pcd: o3d.geometry.PointCloud):
    """Computes the bounding box of a point cloud.

    Args:
        pcd (o3d.geometry.PointCloud): the point cloud to compute the bounding box.
    """
    bbox = pcd.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    extent = bbox.get_extent()
    return center, extent


def transform_mesh(mesh: o3d.geometry.TriangleMesh,
                   translation: np.ndarray, scale: float):
    """Performs transformation to a mesh.

    Args:
        mesh (o3d.geometry.TriangleMesh): the mesh to transform.
        translation (np.ndarray): the translation to apply to the mesh.
        scale (float): the scaling factor to apply to the mesh.
    """
    points = np.asarray(mesh.vertices)
    points = (points + translation) * scale
    mesh.vertices = o3d.utility.Vector3dVector(points)
    return mesh


def get_table_plane(table_pcd):
    """Gets a plane of the table.

    Args:
        table_pcd (o3d.geometry.PointCloud): the point cloud to extract
         the plane from.
    """
    # Gets plane.
    pt_coords = np.asarray(table_pcd.points)
    ransac = RANSACRegressor(max_trials=1005, residual_threshold=0.005)
    ransac.fit(pt_coords[:, :2], pt_coords[:, 2])
    a, b = ransac.estimator_.coef_
    d = ransac.estimator_.intercept_
    inlier_mask = ransac.inlier_mask_

    outlier_mask = np.logical_not(inlier_mask)
    normal_vector = np.array([a, b, -1])
    normal_vector = normal_vector / np.linalg.norm(normal_vector)
    point_on_plane = np.array([0, 0, d])  # 当 x=0, y=0 时的 z 值

    # Gets point cloud.
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pt_coords)
    colors = np.zeros_like(pt_coords)
    colors[inlier_mask] = [0, 1, 0]  # 内点为绿色
    colors[outlier_mask] = [1, 0, 0]  # 外点为红色
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return normal_vector, point_on_plane


def cartesian_to_polar(x, y):
    """Transform cartesian coordinates to polar coordinate.

    Args:
        x (float): x coordinate
        y (float): y coordinate

    Returns:
        r (float): radius in the polar coordinate.
        theta (float): angle in the polar coordinate.
    """
    r = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(y, x)
    return r, theta


def polar_to_cartesian(r, theta):
    """Transform polar coordinates to cartesian coordinate.

    Args:
        r (float): radius in polar coordinate.
        theta (float): angle in polar coordinate.

    Returns:
        x (float): x coordinate in cartesian coordinate.
        y (float): y coordinate in cartesian coordinate.
    """
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y


def scale_mesh_in_plane(mesh, height_direction, scale_factor):
    """Scales the mesh on the plane.

    Args:
        mesh (o3d.geometry.TriangleMesh): the mesh to scale.
        height_direction (np.ndarray): the height direction.
        scale_factor (float): the scale factor.
    """
    # Normalize the normal vector.
    height_direction = np.array(height_direction)
    height_direction = height_direction / np.linalg.norm(height_direction)

    # Gets direction perpendicular to the height.
    basis = np.eye(3)
    basis[:, 2] = height_direction
    basis[:, :2] = np.linalg.qr(basis[:, :2])[0]  # Gram-Schmidt orthogonalization

    # To new coordinate.
    vertices = mesh.vertices
    transformed_vertices = vertices @ basis

    # Gets the plane.
    xy = transformed_vertices[:, :2]
    z = transformed_vertices[:, 2]

    # To polar coordinate.
    r, theta = cartesian_to_polar(xy[:, 0], xy[:, 1])
    r_scaled = r * scale_factor

    # Back to cartesian.
    x_scaled, y_scaled = polar_to_cartesian(r_scaled, theta)
    scaled_vertices = np.vstack((x_scaled, y_scaled, z)).T

    # To original coordiante.
    inverse_basis = np.linalg.inv(basis)
    final_vertices = scaled_vertices @ inverse_basis
    mesh.vertices = o3d.utility.Vector3dVector(final_vertices)
    return mesh


def to_rotation_matrix(axis, angle):
    """Gets rotation matrix from axis and angle.

    Args:
        axis (np.ndarray): Vector along which to rotate.
        angle (np.ndarray): Angle to rotate around.
    """
    axis = axis / np.linalg.norm(axis)
    a = np.cos(angle / 2.0)
    b, c, d = -axis * np.sin(angle / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([
        [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
        [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
        [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]
    ])


def rotate_mesh(mesh, axis, angle):
    """Rotates the mesh."""
    # Gets rotation matrix.
    r_matrix = to_rotation_matrix(axis, angle)
    vertices = np.asarray(mesh.vertices)

    # Apply the transformation.
    rotated_vertices = vertices @ r_matrix.T
    mesh.vertices = o3d.utility.Vector3dVector(rotated_vertices)
    return mesh


def get_parser():
    """Gets parser"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--index', type=str,
                        default='../data/1.obj')
    parser.add_argument('-d', '--depth_pcd', type=str,
                        default='../demo_data/16/')
    parser.add_argument('-m', '--mesh', type=str,
                        default='../demo_data/16/mesh.ply')
    parser.add_argument('-o', '--output', type=str,
                        default='../demo_data/16/output.ply')
    return parser.parse_args()


def main():
    """Demo usage."""
    args = get_parser()
    SCALE_IN_Z = 0.0164515
    SCALE_XY = 0.038
    ROTATION_ANGLE = 80

    # Loads point cloud data.
    food_path = os.path.join(args.depth_pcd, "food.pcd")
    table_path = os.path.join(args.depth_pcd, "table.pcd")
    target_pcd = o3d.io.read_point_cloud(food_path)
    table_pcd = o3d.io.read_point_cloud(table_path)

    source_mesh = o3d.io.read_triangle_mesh(args.mesh)
    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(source_mesh.vertices)

    # Gets table plane.
    normal_vector, point_on_plane = get_table_plane(table_pcd)
    normal_vector = normal_vector / np.linalg.norm(normal_vector)

    # Use normal_vector as z direction.
    z_direction = normal_vector
    z_direction = z_direction / np.linalg.norm(z_direction)

    scale_z = SCALE_IN_Z
    scale_matrix_z = np.eye(4)
    scale_matrix_z[0:3, 0:3] += (scale_z - 1) * np.outer(z_direction,
                                                         z_direction)
    source_mesh.transform(scale_matrix_z)
    source_mesh = scale_mesh_in_plane(source_mesh, z_direction, SCALE_XY)

    # Transformation.
    source_mesh = rotate_mesh(source_mesh, z_direction, ROTATION_ANGLE)
    target_center, target_extent_xyz = compute_bounding_box_center_scale(
        target_pcd)
    translation = target_center - np.mean(np.asarray(source_mesh.vertices), axis=0)
    source_mesh = transform_mesh(source_mesh, translation, 1)

    o3d.visualization.draw_geometries([target_pcd, source_mesh])
    o3d.io.write_triangle_mesh(args.output, source_mesh, write_ascii=True)


if __name__ == "__main__":
    main()