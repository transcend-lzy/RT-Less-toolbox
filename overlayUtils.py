import pygame
from pygame.locals import *
import cv2
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import pickle
from ruamel import yaml
from read_stl import stl_model
import os.path as osp


def cube(tri):  # 提取一堆点
    glBegin(GL_TRIANGLES)  # 绘制多个三角形
    for Tri in tri:
        glColor3fv(Tri['colors'])
        glVertex3fv(
            (Tri['p0'][0], Tri['p0'][1], Tri['p0'][2]))
        glVertex3fv(
            (Tri['p1'][0], Tri['p1'][1], Tri['p1'][2]))
        glVertex3fv(
            (Tri['p2'][0], Tri['p2'][1], Tri['p2'][2]))

    glEnd()  # 实际上以三角面片的形式保存


def draw_cube_test(worldOrientation, worldLocation, tri, window, display):
    glPushMatrix()
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    pos = worldLocation[0]

    rm = worldOrientation.T

    rm[:, 0] = -rm[:, 0]
    rm[:, 1] = -rm[:, 1]

    xx = np.array([rm[0, 0], rm[1, 0], rm[2, 0]])
    yy = np.array([rm[0, 1], rm[1, 1], rm[2, 1]])
    zz = np.array([rm[0, 2], rm[1, 2], rm[2, 2]])
    obj = pos + zz

    gluLookAt(pos[0], pos[1], pos[2], obj[0], obj[1], obj[2], yy[0], yy[1], yy[2])
    cube(tri)
    glPopMatrix()
    pygame.display.flip()

    # Read the result
    string_image = pygame.image.tostring(window, 'RGB')
    temp_surf = pygame.image.fromstring(string_image, display, 'RGB')
    tmp_arr = pygame.surfarray.array3d(temp_surf)
    return (tmp_arr)  # 得到最后的图


def init(width, height, k_path):
    pygame.init()
    display = (width, height)
    window = pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    scale = 0.0001
    k = read_yaml(k_path)['Intrinsic']
    print(k)
    fx = k[0][2]  # 相机标定矩阵的值
    fy = k[1][2]
    cx = k[0][0]
    cy = k[1][1]
    glFrustum(-fx * scale, (width - fx) * scale, -(height - fy) * scale, fy * scale,
              (cx + cy) / 2 * scale, 20)  # 透视投影
    glClearDepth(1.0)
    glDepthFunc(GL_LESS)  # 设置深度测试函数
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_POINT_SMOOTH)
    glPolygonMode(GL_FRONT, GL_FILL)
    glPolygonMode(GL_BACK, GL_FILL)

    return display, window


def read_yaml(yaml_path):
    f = open(yaml_path, 'r', encoding='utf-8')
    cfg = f.read()
    dic = yaml.safe_load(cfg)

    return dic


def read_pkl(pkl_path):
    with open(pkl_path, "rb") as f:
        pose = pickle.load(f)["RT"]
    return pose


def vec2T(r, t):  # 将旋转和平移向量转换为rt矩阵  r和t都是3*1
    t = t.reshape(3, 1)
    R = cv2.Rodrigues(r)
    rtMatrix = np.c_[np.r_[R[0], np.array([[0, 0, 0]])], np.r_[t, np.array([[1]])]]
    return rtMatrix


def T2vec(R):  # 矩阵分解为旋转矩阵和平移向量
    r = R[:3, :3]
    t = R[:3, 3]
    return r, t


def creatC2m(rM2c, tM2c):  # 输入为r 3*3   t为 1*3
    """得到相机到模型的相对位姿

    Args:
        rM2c (3 * 3的矩阵): 模型到相机的旋转矩阵
        tM2c (1 * 3 的向量): 模型到相机的平移向量

    Returns:
        3 * 3的矩阵: 旋转矩阵
        1 * 3的向量: 平移向量
    """
    R = rM2c
    T = np.dot(-rM2c.T, tM2c.reshape(3, 1)).reshape(1, 3)
    return R, T


def show_photo(photo):  # 展示照片
    if (photo.shape[0] > 1000):
        photo = cv2.resize(photo, (int(2448.0 // 2), int(2048.0 // 2)))
    cv2.imshow("abc", photo)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def creatBottom(height, width):  # 生成要求大小的bottom图片
    img = np.zeros((height, width), dtype=np.uint8)
    bottom = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for i in range(3):
        bottom[:, :, i] = 0
    return bottom


def create_img_pose_render(W_Rm2c, W_Lm2c, cad_path, window, display, height, width):
    W_Rm2c = cv2.Rodrigues(np.matrix(W_Rm2c))[0]
    W_Lm2c = np.array([[W_Lm2c[0][0], W_Lm2c[0][1], W_Lm2c[0][2]]]).reshape(3, 1)
    rt = vec2T(W_Rm2c, W_Lm2c)
    W_Rm2c, W_Lm2c = T2vec(rt)
    W_Rc2m, W_Lc2m = creatC2m(W_Rm2c, W_Lm2c)
    tri = stl_model(cad_path).tri
    im = np.array(draw_cube_test(W_Rc2m, W_Lc2m, tri, window, display))
    pose_mask = np.zeros((height, width, 3))
    for i in range(3):
        pose_mask[:, :, i] = im[:, :, i].T
    return pose_mask


def create_img_pose_true(W_Rm2c, W_Lm2c, objId, cad_path, window, display, height, width):
    W_Rm2c = cv2.Rodrigues(np.matrix(W_Rm2c))[0]
    W_Lm2c = np.array([[W_Lm2c[0][0], W_Lm2c[0][1], W_Lm2c[0][2]]]).reshape(3, 1)
    rt = vec2T(W_Rm2c, W_Lm2c)
    W_Rm2c, W_Lm2c = T2vec(rt)
    W_Rc2m, W_Lc2m = creatC2m(W_Rm2c, W_Lm2c)
    tri = stl_model(osp.join(cad_path, objId + '.stl')).tri
    im = np.array(draw_cube_test(W_Rc2m, W_Lc2m, tri, window, display))
    pose_mask = np.zeros((height, width, 3))
    for i in range(3):
        pose_mask[:, :, i] = im[:, :, i].T
    return pose_mask


def create_img_render(gt_path, cad_path, window, display, height, width):
    pose = read_pkl(gt_path)
    W_Rm2c = np.array(pose[:3, :3])
    W_Lm2c = np.array(pose[:3, 3:]).reshape(1, 3)
    print(W_Lm2c)
    print(W_Rm2c)
    return create_img_pose_render(W_Rm2c, W_Lm2c, cad_path, window, display, height, width)


def create_img_true(info, cad_path, window, display, height, width):
    W_Rm2c = info['m2c_R']
    W_Lm2c = info['m2c_T']
    objId = str(info['obj_id'])
    return create_img_pose_true(W_Rm2c, W_Lm2c, objId, cad_path, window, display, height, width)


