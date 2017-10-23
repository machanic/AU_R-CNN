from functools import cmp_to_key

import numpy as np


class Point(object):
    __slots__ = ('x', 'y')


center = Point()


def cmp_by_clockwise(a_point, b_point):
    cmp = lambda a,b : (a > b) - (a < b)
    a_x, a_y = a_point
    b_x, b_y = b_point
    if a_x - center.x >= 0 and b_x - center.x < 0:
        return -1
    if a_x - center.x < 0 and b_x - center.x >= 0:
        return 1
    if a_x - center.x == 0 and b_x - center.x == 0:
        if a_y - center.y >= 0 or b_y - center.y >= 0:
            return -cmp(a_y, b_y)
        return cmp(a_y, b_y)
    # compute the cross product of vectors (center -> a) x (center -> b)
    det = (a_x - center.x) * (b_y - center.y) - \
        (b_x - center.x) * (a_y - center.y)
    if det < 0:
        return -1
    if det > 0:
        return 1
    # points a and b are on the same line from the center
    # check which point is closer to the center
    d1 = (a_x - center.x) * (a_x - center.x) + \
        (a_y - center.y) * (a_y - center.y)
    d2 = (b_x - center.x) * (b_x - center.x) + \
        (b_y - center.y) * (b_y - center.y)
    if d1 > d2:
        return -1
    elif d1 < d2:
        return 1
    return 0


def sort_clockwise(point_array):
    global center

    center_coordinate = np.mean(point_array, axis=0)
    center.x = center_coordinate[0]
    center.y = center_coordinate[1]

    ret = sorted(point_array, key=cmp_to_key(cmp_by_clockwise))
    return np.array(ret)


if __name__ == "__main__":
    point_array = [(10, 20), (20, 20), (10, 30), (20, 30)]
    print(sort_clockwise(point_array))
