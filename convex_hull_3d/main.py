import math
import sys
import open3d as o3d
import numpy as np
from convex_hull_3d.utils import *
from convex_hull_3d.plane import Plane

def create_vertices(ply_path):
    mesh = o3d.io.read_triangle_mesh(ply_path)
    mesh.compute_vertex_normals()
    print('\nLoaded mesh file %s' % ply_path)
    print('#vertices:', np.asarray(mesh.vertices).shape[0])
    print('#faces:', np.asarray(mesh.triangles).shape[0])
    extremes, points = search_init_points(mesh)

    if len(points) < 4:
        print("Less than 4 points so 1D or 2D")
        sys.exit()
    initial_line = initial_max(extremes)
    third_point = max_dist_line_point1(initial_line[0], initial_line[1], points)
    third_point1 = max_dist_line_point2(initial_line[0], initial_line[1], extremes)
    first_plane = Plane(initial_line[0], initial_line[1],
                        third_point)
    fourth_point = max_dist_plane_point(first_plane, points)
    possible_internal_points = [initial_line[0], initial_line[1], third_point,
                                fourth_point]
    second_plane = Plane(initial_line[0], initial_line[1], fourth_point)
    third_plane = Plane(initial_line[0], fourth_point, third_point)
    fourth_plane = Plane(initial_line[1], third_point, fourth_point)

    list_of_planes = []  # List containing all the planes
    list_of_planes.append(first_plane)
    list_of_planes.append(second_plane)
    list_of_planes.append(third_plane)
    list_of_planes.append(fourth_plane)
    for plane in list_of_planes:
        set_correct_normal(possible_internal_points, plane)

    first_plane.calculate_to_do(points)
    second_plane.calculate_to_do(points)
    third_plane.calculate_to_do(points)
    fourth_plane.calculate_to_do(points)
    any_left = True

    while any_left:
        any_left = False
        for working_plane in list_of_planes:
            if len(working_plane.to_do) > 0:
                any_left = True
                eye_point = find_eye_point(working_plane)  # Calculate the eye point of the face

                edge_list = set()
                visited_planes = []

                calc_horizon(visited_planes, working_plane, eye_point, edge_list,
                             list_of_planes)  # Calculate the horizon

                for internal_plane in visited_planes:  # Remove the internal planes
                    list_of_planes.remove(internal_plane)

                for edge in edge_list:  # Make new planes
                    new_plane = Plane(edge.pointA, edge.pointB, eye_point)
                    set_correct_normal(possible_internal_points, new_plane)

                    temp_to_do = set()
                    for internal_plane in visited_planes:
                        temp_to_do = temp_to_do.union(internal_plane.to_do)

                    new_plane.calculate_to_do(points, temp_to_do)

                    list_of_planes.append(new_plane)

    final_vertices = set()
    for plane in list_of_planes:
        final_vertices.add(plane.pointA)
        final_vertices.add(plane.pointB)
        final_vertices.add(plane.pointC)

    return final_vertices


if __name__ == '__main__':
    create_vertices()
