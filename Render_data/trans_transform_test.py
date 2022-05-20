import cv2

import lmdb
import numpy as np
import os

from PIL import Image
from plyfile import PlyData

from config import cfg
from transforms3d.euler import mat2euler

import pickle


def load_ply_model(model_path):
    ply = PlyData.read(model_path)
    data = ply.elements[0].data
    x = data['x']
    y = data['y']
    z = data['z']
    return np.stack([x, y, z], axis=-1)

rotation_transform = np.array([[1., 0., 0.],
                                   [0.,-1., 0.],
                                   [0., 0., -1.]])

blender_model = load_ply_model('/home/reflex/Render_data/data/metal/obj3/resize_obj3.ply')
orig_model = load_ply_model('/home/reflex/Render_data/data/metal/obj3/resize_obj3.ply')
blender_model = np.dot(blender_model, rotation_transform.T)
translation_transform = np.mean(orig_model, axis=0) - np.mean(blender_model, axis=0)
k= translation_transform
print(k)