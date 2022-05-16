import os
from overlayUtils import *


class OverLay():
    def __init__(self):
        self.img_path = '/home/wzz/4T/chaoyue/rtl/newRenders/obj1/1.jpg'
        self.R = [[-0.03089962, -0.83554637, 0.54855025],
                  [-0.9976629, -0.00767904, -0.06789459],
                  [0.06094141, -0.54936618, -0.83335644]]  # 3*3的矩阵
        self.T = [[-0.01920821, 0.00348976, 0.41109109]]  # 1 * 3的向量
        self.K = [[560.0, 0.0, 256.0], [0.0, 560.0, 192.0], [0.0, 0.0, 1.0]]  # 3 * 3的矩阵
        self.cad_path = '/home/wzz/4T/chaoyue/rtl/CADmodels/stl/1.stl'  # stl文件
        self.save_path = './vis/1.jpg'

    def init(self, k, width, height):
        pygame.init()
        display = (width, height)
        window = pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
        scale = 0.0001
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

    def create_img(self):
        img = cv2.imread(self.img_path)
        height = img.shape[0]
        width = img.shape[1]
        display, window = self.init(self.K, width, height)
        W_Rm2c = cv2.Rodrigues(np.matrix(self.R))[0]
        W_Lm2c = np.array([[self.T[0][0], self.T[0][1], self.T[0][2]]]).reshape(3, 1)
        rt = vec2T(W_Rm2c, W_Lm2c)
        W_Rm2c, W_Lm2c = T2vec(rt)
        W_Rc2m, W_Lc2m = creatC2m(W_Rm2c, W_Lm2c)
        tri = stl_model(self.cad_path).tri
        im = np.array(draw_cube_test(W_Rc2m, W_Lc2m, tri, window, display))
        pose_mask = np.zeros((height, width, 3))
        for i in range(3):
            pose_mask[:, :, i] = im[:, :, i].T
        return pose_mask

if __name__ == '__main__':
    overlay = OverLay()
    show_photo(overlay.create_img())