import cv2
import random
from torch.utils.data import DataLoader, Dataset
import torch
from PIL import Image
from utils import load_ply, rotate_img
import glob, pickle
import numpy as np
import os.path as osp
import os
import yaml
import matplotlib
import tqdm


matplotlib.use('TKAgg')
import matplotlib.patches as patches
from matplotlib import pyplot as plt

blender_K = np.array([[560.0, 0.0, 256.0], [0.0, 560.0, 192.0], [0.0, 0.0, 1.0]])


class Metal_Dataset(Dataset):
    def __init__(self, root, cls, is_train=True, scene=None, index=None):

        super(Metal_Dataset, self).__init__()
        self.root = root
        self.train_root = osp.join(self.root, "train512")
        self.test_root = osp.join(self.root, "test512")
        self.cls = cls
        self.data_paths = []
        self.is_training = is_train
        self.render_dir = '/home/wzz/4T/chaoyue/rtl/newRenders'  # 渲染数据集的路径
        self.random_num = 660  #真实数据的数量
        self.random_render_num = 340  # 使用渲染图像的个数
        self.scene = scene
        self.index = index
        self.sun_path = "/home/wzz/4T/chaoyue/SUN2012pascalformat"  # sun数据集的路径
        if is_train:
            self.path = osp.join(self.train_root, self.cls)
            K = yaml.load(open(osp.join(self.path, 'Intrinsic.yml'), 'r'))
            K = np.array(K['Intrinsic']).reshape(3, 3)
            self.K = K
            self.train_pose = yaml.load(open(osp.join(self.path, 'gt.yml'), 'r'))
            self.data_paths = self.get_train_data_path(self.train_root, self.cls)

        else:
            self.path = osp.join(self.test_root, self.scene)
            K = yaml.load(open(osp.join(self.path, 'Intrinsic.yml'), 'r'))
            K = np.array(K['Intrinsic']).reshape(3, 3)
            self.K = K
            self.train_pose = yaml.load(open(osp.join(self.path, 'gt.yml'), 'r'))
            self.data_paths = self.get_test_data_path(self.test_root, self.scene)

        self.mesh_model = load_ply(osp.join(self.root, 'CADModels', "ply", "{}.ply".format(cls)))

    def get_test_data_path(self, root, cls):

        paths_list = []
        paths = {}

        count = os.listdir(osp.join(root, cls, "rgb"))

        train_inds = [int(ind.replace(".png", "")) for ind in count]
        train_inds.sort()

        train_img_path = osp.join(root, cls, "rgb")
        mask_path = osp.join(root, cls, "mask")  # use test data train

        for idx in train_inds:
            idx = int(idx)
            img_name = "{}.png".format(idx)
            obj_id = self.cls[3:]
            obj_id = int(obj_id)
            mask_name = "{}_{}.png".format(idx, obj_id)
            paths["img_path"] = osp.join(train_img_path, img_name)
            paths["mask_path"] = osp.join(mask_path, mask_name)
            paths["type"] = "test"
            paths_list.append(paths.copy())

        return paths_list

    def get_train_data_path(self, root, cls):
        paths_list = []
        paths = {}
        render_num = len(glob.glob(osp.join(self.render_dir, cls, "*.pkl")))
        render_inds = [random.randint(0, render_num - 1) for i in range(self.random_render_num)]
        render_inds.sort()

        count = random.sample(os.listdir(osp.join(root, cls, "rgb")), self.random_num)

        train_inds = [int(ind.replace(".png", "")) for ind in count]
        train_inds.sort()

        train_img_path = osp.join(root, cls, "rgb")
        mask_path = osp.join(root, cls, "mask")  # use test data train
        for idx in train_inds:
            idx = int(idx)
            img_name = "{}.png".format(idx)
            mask_name = "{}.png".format(idx)
            paths["img_path"] = osp.join(train_img_path, img_name)
            paths["mask_path"] = osp.join(mask_path, mask_name)
            paths["type"] = "true"
            paths_list.append(paths.copy())

        for idx in render_inds:
            img_name = "{}.jpg".format(idx)
            mask_name = "{}_depth.png".format(idx)
            pose_name = "{}_RT.pkl".format(idx)

            paths["img_path"] = osp.join(self.render_dir, cls, img_name)
            paths["mask_path"] = osp.join(self.render_dir, cls, mask_name)
            paths["pose_path"] = osp.join(self.render_dir, cls, pose_name)
            paths["type"] = "render"
            paths_list.append(paths.copy())

        return paths_list

    def get_data(self, path):
        img = np.array(Image.open(path["img_path"]))
        if path["type"] == "true":
            mask = (np.asarray(cv2.imread(path["mask_path"], 0)) != 0).astype(np.uint8)
            # if mask.shape == (512,512,3):
            # mask=mask[:,:,0]
            # mask=mask[:,:,0] #3 channel to 1 channel
            idx = int(osp.basename(path["img_path"]).replace(".png", ""))
            idx_bmp = str(idx)
            instance_gt = self.train_pose[idx_bmp][0]
            R = np.array(instance_gt['m2c_R']).reshape(3, 3)
            t = np.array(instance_gt['m2c_T']).reshape(3, 1)
            pose = np.concatenate([R, t], axis=1)
            K = self.K


        elif path["type"] == "render":
            with open(path["pose_path"], "rb") as f:
                pose = pickle.load(f)["RT"]

            K = blender_K
            mask = (np.asarray(Image.open(path["mask_path"]))).astype(np.uint8)
            mask = mask[:, :, 0]
        elif path["type"] == "test":
            mask = (np.asarray(cv2.imread(path["mask_path"], 0)) != 0).astype(np.uint8)
            # mask = mask[:, :, 0]  # 3 channel to 1 channel
            idx = int(osp.basename(path["img_path"]).replace(".png", ""))
            idx_bmp = str(idx)
            instance_gt = self.train_pose[idx_bmp][self.index]  # test
            R = np.array(instance_gt['m2c_R']).reshape(3, 3)
            t = np.array(instance_gt['m2c_T']).reshape(3, 1)
            pose = np.concatenate([R, t], axis=1)
            K = self.K

        return img, mask, pose, K

    def get_test_data(self, path):

        img = np.array(Image.open(path["img_path"]))
        return img

    def augment(self, img, mask, pose, K):

        img = np.asarray(img).astype(np.uint8)
        foreground = np.sum(mask)
        if foreground > 400:
            # randomly mask out to add occlusion
            R = np.eye(3, dtype=np.float32)
            R_orig = pose[:3, :3]
            T_orig = pose[:3, 3]

            img, mask, R = rotate_img(img, mask, T_orig, K, -30, 30)

            new_R = np.dot(R, R_orig)
            pose[:3, :3] = new_R

        return img, mask, pose

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

    def random_background(self, img, mask):
        '''
        随机背景替换
        :param img:
        :param mask:
        :return:
        '''
        height, width = img.shape[:2]
        random_img_path = random.choice(os.listdir(osp.join(self.sun_path, 'JPEGImages')))
        random_img = cv2.imread(osp.join(self.sun_path, 'JPEGImages', random_img_path))
        row, col = random_img.shape[:2]
        if row <= height or col <= width:
            if row < col:
                random_img = cv2.resize(random_img, dsize=(height * 2, int((height * 2 / row) * col)))
            else:
                random_img = cv2.resize(random_img, dsize=(int((width * 2 / col) * row), width * 2))
        random_img = self.random_crop(random_img, img)
        random_img = self.random_flip(random_img)
        mask_img = cv2.bitwise_and(img, img, mask=mask)

        maskCopy = np.array(mask)
        np.place(maskCopy, maskCopy > 0, 255)
        mask_inv = cv2.bitwise_not(maskCopy)
        background = cv2.bitwise_and(random_img, random_img, mask=mask_inv)
        background = cv2.medianBlur(background, 3)
        last_img = cv2.add(mask_img, background)
        last_img = cv2.medianBlur(last_img, 3)
        # cv2.imshow("test", last_img)
        # cv2.waitKey(0)
        return last_img

    def __getitem__(self, index):

        path = self.data_paths[index]
        img, mask, pose, K = self.get_data(path)
        if self.is_training:
            img, mask, pose = self.augment(img, mask, pose, K)
            alpha = random.uniform(0.6, 1.6)
            beta = random.randint(-25, 25)
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
            img = self.random_background(img, mask)
        mask = np.asarray(mask).astype(np.uint8)

        img = img / 255.0

        img -= [0.419, 0.427, 0.424]  # train data for resnet
        img /= [0.184, 0.206, 0.197]

        mask = np.asarray(mask).astype(np.uint8)

        img = torch.tensor(img, dtype=torch.float32).permute((2, 0, 1))

        return img, mask, pose.astype(np.float32), K

    def __len__(self):
        return len(self.data_paths)
