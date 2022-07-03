from easydict import EasyDict
import os
import sys
import numpy as np

cfg = EasyDict()

"""
Path settings
"""
cfg.ROOT_DIR = '/home/lqz/chaoyue/Render_data'
cfg.DATA_DIR = os.path.join(cfg.ROOT_DIR, 'data')
cfg.MODEL_DIR = os.path.join(cfg.DATA_DIR, 'model')
cfg.REC_DIR = os.path.join(cfg.DATA_DIR, 'record')
cfg.FIGURE_DIR = os.path.join(cfg.ROOT_DIR, 'figure')
cfg.BLENDER_DIR = os.path.join(cfg.ROOT_DIR, "blender")


def add_path():
    for key, value in cfg.items():
        if 'DIR' in key:
            sys.path.insert(0, value)


add_path()
sys.path.extend([".", ".."])


"""
Data settings
"""
cfg.render_obj_ids = [16]
cfg.LINEMOD = os.path.join(cfg.DATA_DIR, 'metal')
cfg.LINEMOD_ORIG = os.path.join(cfg.DATA_DIR, 'metal')
cfg.OCCLUSION_LINEMOD = os.path.join(cfg.DATA_DIR, 'OCCLUSION_LINEMOD')
cfg.SUN = os.path.join(cfg.DATA_DIR, "SUN")

"""
Rendering setting
"""
cfg.BLENDER_PATH = '/home/lqz/chaoyue/blender-2.79-linux-glibc219-x86_64/blender'
cfg.NUM_SYN = 60  #渲染的数量
cfg.WIDTH = 640
cfg.HEIGHT = 480
cfg.low_azi = 0
cfg.high_azi = 40
cfg.low_ele = 10
cfg.high_ele = 40
cfg.low_theta = 10
cfg.high_theta = 40
cfg.cam_dist = 0.5
cfg.MIN_DEPTH = 0
cfg.MAX_DEPTH = 1

cfg.render_K=np.array([[1173.308073739741, 0., 329.138849381171],
                       [0., 1172.722440477151, 253.1055863816615],
                       [0., 0., 1.]],np.float32)
'''
cfg.linemod_K=np.array([[572.41140,0.       ,325.26110],
                        [0.       ,573.57043,242.04899],
                        [0.       ,0.       ,1.       ]],np.float32)
'''

cfg.linemod_cls_names=['obj1','obj2','obj3','duck','glue','iron','phone',
                       'benchvise','can','driller','eggbox','holepuncher','lamp']
cfg.occ_linemod_cls_names=['ape','can','cat','driller','duck','eggbox','glue','holepuncher']
cfg.linemod_plane=['can']

cfg.symmetry_linemod_cls_names=['glue','eggbox']


'''
pascal 3d +
'''
cfg.PASCAL = os.path.join(cfg.DATA_DIR, 'PASCAL3D')
cfg.pascal_cls_names=['aeroplane','bicycle','boat','bottle','bus','car',
                      'chair','diningtable','motorbike','sofa','train','tvmonitor']
cfg.pascal_size=128


'''
YCB
'''
cfg.ycb_sym_cls=[21,20,19,16,13] # foam_brick extra_large_clamp large_clamp wood_block bowl
cfg.ycb_class_num=21
