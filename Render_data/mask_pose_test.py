
from PIL import Image

import glob, pickle
import torch_geometric.transforms as T
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt
import os
import matplotlib.patches as patches
import yaml

def project(xyz, K, RT):
    """
    xyz: [N, 3]
    K: [3, 3]
    RT: [3, 4]
    """
    xyz = np.dot(xyz, RT[:, :3].T) + RT[:, 3:].T
    xyz = np.dot(xyz, K.T)
    xy = xyz[:, :2] / xyz[:, 2:]
    return xy

path='/home/reflex/文档/metal/cad'
corners=np.loadtxt(os.path.join(path,'1.txt'))
corners=corners*0.001
test_path="/home/reflex/文档/metal/new/3"
img_path = os.path.join('/home/reflex/Render_data/data/metal/renders/obj1/150.jpg')
#a=np.array(Image.open(img_path))
imgs=plt.imread(img_path)
#K_test_info=yaml.load(open(os.path.join(test_path, 'Intrinsic.yml'), 'r'))
#K_=K_test_info['Intrinsic']
#K_s=np.array(K_).reshape(3,3)

#blender_K=K_s
blender_K = np.array([[700.0, 0.0, 320.0], [0.0, 700.0, 240.0], [0.0, 0.0, 1.0]])
with open("/home/reflex/Render_data/data/metal/renders/obj1/150_RT.pkl", "rb") as f:
    pose = pickle.load(f)["RT"]

K = blender_K
#mask = (np.asarray(Image.open('/home/reflex/Render_data/data/metal/renders/obj3/0_depth.png'))).astype(np.uint8)
img = np.array(imgs)
#img[mask>0]+=np.asarray([0,128,0],np.uint8)
#plt.imshow(img)
corners_pred = project(corners, K, pose)

_, ax = plt.subplots(1)
ax.imshow(imgs)
ax.add_patch(
    patches.Polygon(xy=corners_pred[[0, 1, 3, 2, 0, 4, 6, 2]], fill=False, linewidth=1,
                    edgecolor='b'))
ax.add_patch(
    patches.Polygon(xy=corners_pred[[5, 4, 6, 7, 5, 1, 3, 7]], fill=False, linewidth=1,
                    edgecolor='b'))

p=6


