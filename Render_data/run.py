from config import cfg
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--type', action='store', dest='type', type=str)
args = parser.parse_args()


def run_rendering():
    from blender.render_utils import Renderer, YCBRenderer
    # YCBRenderer.multi_thread_render()
    # renderer = YCBRenderer('037_scissors')
    for i in range(16,17):
        renderer=Renderer('obj{}'.format(i))
        renderer.run()


def run_fuse():
    from fuse.fuse import run
    run()


if __name__ == '__main__':
   # globals()['run_' + args.type]()
    run_rendering()
