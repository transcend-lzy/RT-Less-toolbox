import open3d as o3d
import numpy as np
import csv
import os
import os.path as osp
from convex_hull_3d.main import create_vertices


def get_dis(point1, point2):
    len_all = 0
    for i in range(3):
        len_all += (point1[i] - point2[i]) ** 2
    return len_all ** 0.5


def get_dia():
    points = []

    ply_path = 'convex_hull_3d//ply//obj1.ply'  # The ply file path of the corresponding object

    points_hull = create_vertices(ply_path)

    for point in points_hull:
        points.append((float(point.x), float(point.y), float(point.z)))

    diameter = 0
    for index, point1 in enumerate(points):
        for point2 in points[(index + 1):]:
            if (diameter < get_dis(point1, point2)):
                point1max = point1
                point2max = point2
            diameter = max(get_dis(point1, point2), diameter)

    print(diameter * 1000)


if __name__ == '__main__':
    get_dia()
