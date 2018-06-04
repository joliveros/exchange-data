import math


def roundup_to_nearest(value, interval=10.0):
    return math.ceil(value / interval) * interval
