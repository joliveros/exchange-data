import os


def datafile_name(name):
    path = f'../data/{name}.json'
    dir = os.path.join(os.path.dirname(__file__))
    path = os.path.join(dir, path)
    return path
