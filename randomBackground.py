import os
import cv2
import random
import numpy as np
import os.path as osp
from PIL import Image
'''
随机替换背景
'''

class RandomBg:
    def __init__(self):
        self.sun_path = "/home/wzz/4T/chaoyue/SUN2012pascalformat/JPEGImages"
        self.img_path = '/home/wzz/4T/chaoyue/rtl/newRenders/obj1/1.jpg'
        self.mask_path = '/home/wzz/4T/chaoyue/rtl/newRenders/obj1/1_depth.png'
        self.render = True

    def random_crop(self, bg_img, img):
        # size is 512*512
        img_h, img_w = img.shape[:2]
        bg_h, bg_w = bg_img.shape[:2]
        y = np.random.randint(0, bg_h - img_h)
        x = np.random.randint(0, bg_w - img_w)

        image = bg_img[y:y + img_h, x:x + img_w, :]

        return image

    def random_flip(self, image, axis=None):
        # 以30%的可能性翻转图片，axis 0 垂直翻转，1水平翻转
        flip_prop = np.random.randint(low=0, high=3)
        axis = np.random.randint(0, 2)
        if flip_prop == 0:
            image = cv2.flip(image, axis)

        return image

    def random_background(self):
        '''
        随机背景替换
        :return:
        '''
        img = cv2.imread(self.img_path)
        mask = np.asarray(cv2.imread(self.mask_path, 0)).astype(np.uint8)
        height, width = img.shape[:2]
        bg_imgs = os.listdir(self.sun_path)
        random_img_path = random.choice(bg_imgs)
        random_img = cv2.imread(osp.join(self.sun_path, random_img_path))
        row, col = random_img.shape[:2]
        if row < height or col < width:
            if (row < col):
                random_img = cv2.resize(random_img, dsize=(height * 2, int((height * 2 / row) * col)))
            else:
                random_img = cv2.resize(random_img, dsize=(int((width * 2 / col) * row), width * 2))
        random_img = self.random_crop(random_img, img)
        random_img = self.random_flip(random_img)
        mask_img = cv2.bitwise_and(img, img, mask=mask)

        np.place(mask, mask > 0, 255)
        mask_inv = cv2.bitwise_not(mask)
        background = cv2.bitwise_and(random_img, random_img, mask=mask_inv)
        background = cv2.medianBlur(background, 3)
        last_img = cv2.add(mask_img, background)
        last_img = cv2.medianBlur(last_img, 3)
        cv2.imshow("test", last_img)
        cv2.waitKey(0)
        return last_img

if __name__ == '__main__':
    randomBg = RandomBg()
    for i in range(10):
        index = random.randint(0, 500)
        randomBg.img_path = osp.join('/home/wzz/4T/chaoyue/rtl/train512/Training images(512#512)/obj1/rgb', str(index) + '.png')
        randomBg.mask_path = osp.join('/home/wzz/4T/chaoyue/rtl/train512/Training images(512#512)/obj1/mask', str(index) + '.png')
        randomBg.random_background()
