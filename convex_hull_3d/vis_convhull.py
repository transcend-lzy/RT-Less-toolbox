import open3d as o3d
import argparse
import numpy as np

def triangle_pcd(pointA, pointB, pointC):
    '''
    定义三角形的点云
    :return:
    '''
    triangle_points = np.array([[pointA.x, pointA.y, pointA.z], [pointB.x, pointB.y, pointB.z], [pointC.x, pointC.y, pointC.z]], dtype=np.float32)
    lines = [[0, 1], [1, 2], [2, 0]]  # Right leg
    colors = [[0, 0, 1] for i in range(len(lines))]  # Default blue
    # 定义三角形的三个角点
    point_pcd = o3d.geometry.PointCloud()  # 定义点云
    point_pcd.points = o3d.utility.Vector3dVector(triangle_points)

    # 定义三角形三条连接线
    line_pcd = o3d.geometry.LineSet()
    line_pcd.lines = o3d.utility.Vector2iVector(lines)
    line_pcd.colors = o3d.utility.Vector3dVector(colors)
    line_pcd.points = o3d.utility.Vector3dVector(triangle_points)

    return line_pcd, point_pcd

def vis(obj, list_of_planes,convhull = None):
    # convhull_wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(convhull)
    # convhull_wireframe.paint_uniform_color(np.array([0, 0, 1.0]))
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window(window_name='Convex Hull Visualizer')
    visualizer.add_geometry(obj)
    for plane in list_of_planes:
        line_pcd, point_pcd = triangle_pcd(plane.pointA, plane.pointB, plane.pointC)
        visualizer.add_geometry(line_pcd)
        visualizer.add_geometry(point_pcd)
    # visualizer.add_geometry(convhull_wireframe)
    visualizer.run()
    visualizer.destroy_window()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize a mesh and its convex hull.')
    parser.add_argument('-m', type=str, required=True, help='The mesh model file.')
    parser.add_argument('-ch', type=str, required=True, help='The convex hull file.')
    args = parser.parse_args()

    mesh = o3d.io.read_triangle_mesh(args.m)
    mesh.compute_vertex_normals()

    convhull = o3d.io.read_triangle_mesh(args.ch)
    vis(mesh, convhull)
