import os
from overlayUtils import *
'''
overlay真实图像,将根据位姿渲染的图像以及bbox都画在图像上
'''
isTest, isInit = False, False


class OverLayTrue:
    def __init__(self):
        global isTest, isInit
        self.img_root_path = '/home/wzz/4T/chaoyue/rtl/test512/Testing images(512#512)'
        if self.img_root_path.split('/')[-2][1] == 'e':
            isTest = True
        self.img_index_start = 1  # 能取到
        self.img_index_end = 10  # 取不到
        self.set_index_start = 1  # 能取到
        self.set_index_end = 3  # 取不到
        self.cad_path = '/home/wzz/4T/chaoyue/rtl/CADmodels/stl'
        self.hasBbox = True
        self.hasBboxOffset = True
        self.save_path_root = './vis'
        if not osp.exists(self.save_path_root):
            os.mkdir(self.save_path_root)
        self.display = None
        self.window = None

    def create_overlay(self):
        global isTest, isInit, bbox, bbox_offset
        try:
            for i in range(self.set_index_start, self.set_index_end):
                if isTest:
                    set_root_path = osp.join(self.img_root_path, 'scene' + str(i))
                    save_path = osp.join(self.save_path_root, 'scene' + str(i))
                else:
                    set_root_path = osp.join(self.img_root_path, 'obj' + str(i))
                    save_path = osp.join(self.save_path_root, 'obj' + str(i))
                if not osp.exists(save_path):
                    os.mkdir(save_path)

                bbox_path = osp.join(set_root_path, 'bbox.yml')
                bbox_offset_path = osp.join(set_root_path, 'bboxOffset.yml')
                if not osp.exists(bbox_offset_path):
                    self.hasBboxOffset = False
                gt_path = osp.join(set_root_path, 'gt.yml')
                k_path = osp.join(set_root_path, 'Intrinsic.yml')

                for j in range(self.img_index_start, self.img_index_end):
                    img = cv2.imread(osp.join(set_root_path, 'rgb', str(j) + '.png'))
                    height = img.shape[0]
                    width = img.shape[1]
                    display, window = init(width, height, k_path)
                    infos = read_yaml(gt_path)[str(j)]
                    if self.hasBbox:
                        bbox = read_yaml(bbox_path)[str(j)]
                    if self.hasBboxOffset:
                        bbox_offset = read_yaml(bbox_offset_path)[str(j)]
                    bottom = creatBottom(height, width)
                    for index, info in enumerate(infos):
                        pose_mask = create_img_true(info, self.cad_path, window, display, height, width)
                        bottom = cv2.addWeighted(bottom.astype(np.uint8), 1, pose_mask.astype(np.uint8), 1, 0)
                        if self.hasBbox:
                            xywh = bbox[index]['xywh']
                            cv2.rectangle(img, (xywh[0], xywh[1]), (xywh[0] + xywh[2], xywh[1] + xywh[3]), (0, 0, 255),
                                          2)  # 红色
                        if self.hasBboxOffset:
                            xywh_off = bbox_offset[index]['xywh']
                            cv2.rectangle(img, (xywh_off[0], xywh_off[1]),
                                          (xywh_off[0] + xywh_off[2], xywh_off[1] + xywh_off[3]), (0, 255, 0), 2)  # 绿色
                    overlay = cv2.addWeighted(img, 0.5, bottom, 1, 0)
                    # self.show_photo(overlay)
                    cv2.imwrite(osp.join(save_path, str(j) + '.png'), overlay)
                    pygame.quit()
        finally:
            pygame.quit()


if __name__ == "__main__":
    overlayTrue = OverLayTrue()
    overlayTrue.create_overlay()
