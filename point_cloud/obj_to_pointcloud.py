import argparse
import sys
import os

import numpy as np
import pandas as pd

from pyntcloud.io import write_ply
from pyntcloud import PyntCloud


def parse_args():
    parser = argparse.ArgumentParser(
        prog='ObjToPointcloud',
        description='Converts .obj file to pointcloud')
    parser.add_argument('input_folder')
    parser.add_argument('output_folder')
    parser.add_argument('--num_points', type=int, default=1024)
    return parser.parse_args()


def triangle_area_multi(v1, v2, v3):
    return 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1), axis=1)


def obj_to_pointcloud(args, obj):
    filename = os.path.join(args.input_folder, obj)

    mesh = PyntCloud.from_file(filename)
    num_points = args.num_points

    v1, v2, v3 = mesh.get_mesh_vertices()
    areas = triangle_area_multi(v1, v2, v3)

    probabilities = areas / areas.sum()

    weighted_random_indices = np.random.choice(
        range(len(areas)), size=num_points, p=probabilities)

    v1 = v1[weighted_random_indices]
    v2 = v2[weighted_random_indices]
    v3 = v3[weighted_random_indices]

    u = np.random.rand(num_points, 1)
    v = np.random.rand(num_points, 1)
    is_a_problem = u + v > 1

    u[is_a_problem] = 1 - u[is_a_problem]
    v[is_a_problem] = 1 - v[is_a_problem]

    w = 1 - (u + v)

    result = (v1 * u) + (v2 * v) + (v3 * w)
    result = result.astype(np.float32)

    ret = pd.DataFrame()
    ret['x'] = result[:, 0]
    ret['y'] = result[:, 1]
    ret['z'] = result[:, 2]

    write_ply(os.path.join(args.output_folder, obj), points=ret)

def process_objs(args):
    for obj in os.listdir(args.input_folder):
        if obj.endswith('.obj'):
            try:
                obj_to_pointcloud(args, obj)
            except Exception as e:
                print('Unable to write OBJ: ', obj)
                print('Exception: ', e)

if __name__ == '__main__':
    args = parse_args()
    process_objs(args)
                                                 
