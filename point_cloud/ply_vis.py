import argparse

from pyntcloud import PyntCloud

parser = argparse.ArgumentParser(
    prog='PlyVis',
    description='Visualizes .ply pointclouds')
parser.add_argument('filename')
args = parser.parse_args()

pc = PyntCloud.from_file(args.filename)

pc.plot()
