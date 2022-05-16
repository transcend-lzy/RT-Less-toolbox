import math


class Point:  # Point class denoting the points in the space
    def __init__(self, x=None, y=None, z=None):
        self.x = x
        self.y = y
        self.z = z

    def __sub__(self, pointX):
        return Point(self.x - pointX.x, self.y - pointX.y, self.z - pointX.z)

    def __add__(self, pointX):
        return Point(self.x + pointX.x, self.y + pointX.y, self.z + pointX.z)

    def length(self):
        return math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)

    def __str__(self):
        return str(self.x) + "," + str(self.y) + "," + str(self.z)

    def __hash__(self):
        return hash((self.x, self.y, self.z))

    def __eq__(self, other):
        # print "Checking equality of Point"
        return (self.x == other.x) and (self.y == other.y) and (self.z == other.z)
