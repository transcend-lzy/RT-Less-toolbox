import os
from overlayUtils import *
'''
overlay渲染图像,将根据位姿渲染的图像画在原始图像上
'''
isInit = False


class OverLayRender:
    def __init__(self):
        self.img_root_path = '/home/wzz/4T/chaoyue/rtl/newRenders'
        self.img_index_start = 1  # 能取到
        self.img_index_end = 2  # 取不到
        self.set_index_start = 1  # 能取到
        self.set_index_end = 2  # 取不到
        self.cad_root_path = '/home/wzz/4T/chaoyue/rtl/CADmodels/stl'
        self.save_path_root = './vis/renders'
        if not osp.exists(self.save_path_root):
            os.mkdir(self.save_path_root)

    def create_overlay(self):
        global isInit
        try:
            for i in range(self.set_index_start, self.set_index_end):
                set_root_path = osp.join(self.img_root_path, 'obj' + str(i))
                save_path = osp.join(self.save_path_root, 'obj' + str(i))
                if not osp.exists(save_path):
                    os.mkdir(save_path)
                k_path = osp.join(self.img_root_path, 'Intrinsic.yml')
                cad_path = osp.join(self.cad_root_path, str(i) + '.stl')
                for j in range(self.img_index_start, self.img_index_end):
                    img = cv2.imread(osp.join(set_root_path, str(j) + '.jpg'))
                    height = img.shape[0]
                    width = img.shape[1]
                    # if not isInit:
                    display, window = init(width, height, k_path)
                    gt_path = osp.join(set_root_path, str(j) + '_RT.pkl')
                    bottom = creatBottom(height, width)
                    pose_mask = create_img_render(gt_path, cad_path, window, display, height, width)
                    # self.show_photo(pose_mask)
                    bottom = cv2.addWeighted(bottom.astype(np.uint8), 1, pose_mask.astype(np.uint8), 1, 0)
                    overlay = cv2.addWeighted(img, 0.5, bottom, 1, 0)
                    cv2.imwrite(osp.join(save_path, str(j) + '.jpg'), overlay)
                    pygame.quit()
        finally:
            pygame.quit()


if __name__ == "__main__":
    overlay_render = OverLayRender()
    overlay_render.create_overlay()
