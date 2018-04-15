import math
import pprint

printer = pprint.PrettyPrinter(indent=2)


def nice_print(value):
    printer.pprint(value)


def roundup_to_nearest(value, interval=10.0):
    return math.ceil(value / interval) * interval
