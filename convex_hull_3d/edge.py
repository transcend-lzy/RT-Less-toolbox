from convex_hull_3d.point import Point
import math


class Edge:  # Make a object of type Edge which have two points denoting the vertices of the edges
    def __init__(self, pointA, pointB):
        self.pointA = pointA
        self.pointB = pointB

    def __str__(self):
        string = "Edge"
        string += "\n\tA: " + str(self.pointA.x) + "," + str(self.pointA.y) + "," + str(self.pointA.z)
        string += "\n\tB: " + str(self.pointB.x) + "," + str(self.pointB.y) + "," + str(self.pointB.z)
        return string

    def __hash__(self):
        return hash((self.pointA, self.pointB))

    def __eq__(self, other):
        if ((self.pointA == other.pointA) and (self.pointB == other.pointB)) or (
                (self.pointB == other.pointA) and (self.pointA == other.pointB)):
            return True

        return False
