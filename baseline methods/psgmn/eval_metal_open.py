import os.path as osp
import os
import torch
import tqdm
from utils.utils import load_ply, project, mesh_project
from network.psgmn import psgmn
import numpy as np
import cv2
from scipy import spatial
import matplotlib.patches as patches
from visualize_test import visualize_mask, visualize_points_3d, visualize_bounding_box, visualize_overlap_mask
from torchvision import transforms
import time
import matplotlib

matplotlib.use('TKAgg')
from matplotlib import pyplot as plt
import yaml
import math

cuda = torch.cuda.is_available()
toPIL = transforms.ToPILImage()

class evaluator:
    def __init__(self, args, model, test_loader, device, obj_index=None):

        self.args = args
        self.cadpath = osp.join(args.data_path, "CADModels", "ply")
        self.mesh_model = load_ply(
            osp.join(
                self.cadpath, "{}.ply".format(args.class_type)
            )
        )
        corners = np.loadtxt(os.path.join(self.cadpath, '{}.txt'.format(args.class_type)))
        self.corners = corners
        self.pts_3d = self.mesh_model["pts"] * 1000
        self.device = device
        self.model = model
        self.proj_2d = []
        self.proj_2d_mean = []
        self.x_error_all = []
        self.y_error_all = []
        self.z_error_all = []
        self.alpha_error_all = []
        self.beta_error_all = []
        self.gama_error_all = []
        self.error = 0
        self.add = []
        diameters = np.load(osp.join(args.data_path, "train512"))
        self.diameter = diameters[args.class_type]
        '''
        rtDic
        For partially symmetric objects, the translation and rotation errors and the project_2d
        need to be multiplied by an rt matrix to obtain a new result to eliminate the duality
        '''
        self.rtDic = np.load(osp.join(args.data_path, 'is_syn.npy'), allow_pickle=True)
        self.data_loader = test_loader
        self.true_num = 0
        self.false_num = 0
        self.index = 0
        self.obj_index = obj_index
        self.scene = args.scene

    def evaluate(self):

        self.model.eval()
        print("model class type:{}".format(self.args.class_type))
        print("scene number is :{}".format(self.scene))
        bbox = yaml.load(open(osp.join(self.args.data_path, "test512", self.scene, 'bboxOffset.yml'), 'r'))
        idx = 1
        outpath = os.path.join('./new_result4/{}'.format(self.scene), '{}.txt'.format(self.args.class_type))
        if not osp.exists('./new_result4/{}'.format(self.scene)):
            os.makedirs('./new_result4/{}'.format(self.scene))
        with torch.no_grad():
            for data in tqdm.tqdm(self.data_loader, leave=False, desc="val"):
                if cuda:
                    img, mask, pose, K = [x.to(self.device) for x in data]
                else:
                    img, mask, pose, K = data
                batch_size = img.shape[0]
                bbox_img = np.zeros([batch_size, 512, 512])
                for i in range(batch_size):
                    photo_name = '{}'.format(idx)
                    x = bbox[photo_name][self.obj_index]['xywh'][0]
                    y = bbox[photo_name][self.obj_index]['xywh'][1]
                    w = bbox[photo_name][self.obj_index]['xywh'][2]
                    h = bbox[photo_name][self.obj_index]['xywh'][3]
                    bbox_img[i, y:y + h + 1, x:x + w + 1] = 1
                    idx = idx + 1

                S, mask_pred = self.model(img, bbox_img=bbox_img)

                self.calculate_projection2d_add(
                    S, mask_pred, mask, pose, K, self.args.class_type, img
                )

                pred_pose = self.get_pose(
                    S, mask_pred, mask, pose, K, self.args.class_type
                )
                self.calculate_tra_and_rot(pose, pred_pose, K, self.args.class_type)

        result = open(outpath, "a")

        proj2d = np.mean(self.proj_2d)
        add = np.mean(self.add)
        x = np.mean(self.x_error_all)
        y = np.mean(self.y_error_all)
        z = np.mean(self.z_error_all)
        alpha = np.mean(self.alpha_error_all)
        beta = np.mean(self.beta_error_all)
        gamma = np.mean(self.gama_error_all)
        proj2dmean = np.mean(self.proj_2d_mean)
        print("2d projection mean:{}".format(proj2dmean), file=result)
        print("2d projections metric: {}".format(proj2d), file=result)
        print('x error:{} mm'.format(x), file=result)
        print('y error:{} mm'.format(y), file=result)
        print('z error:{} mm'.format(z), file=result)
        print('alpha error:{} °'.format(alpha), file=result)
        print('beta error:{} °'.format(beta), file=result)
        print('gamaa error:{} °'.format(gamma), file=result)
        print('error number is {}'.format(self.error), file=result)
        print("ADD metric: {}".format(add), file=result)
        result.close()

    def get_pose(self, S, mask_pred, mask, pose, K, cls):
        pred_path = os.path.join(os.getcwd(), "posedata", "pred_{}.csv".format(self.args.class_type))
        gt_path = os.path.join(os.getcwd(), "posedata", "gt_{}.csv".format(self.args.class_type))
        mask_pred = mask_pred.detach().cpu().numpy().astype(np.uint8)
        S = S.permute(1, 0).detach().cpu().numpy()

        pose = pose.detach().cpu().numpy()
        K = K.detach().cpu().numpy()

        valid_mask = np.where(mask_pred)
        batch = valid_mask[0]
        v = valid_mask[1]
        u = valid_mask[2]

        S_key_points = S

        batch_size = mask_pred.shape[0]
        org_pose = np.concatenate([np.eye(3), np.zeros((3, 1))], axis=-1)
        for i in range(batch_size):
            K_s = K[i]
            gt_pose = pose[i]
            mesh_pts_2d = mesh_project(self.mesh_model["pts"], K_s, gt_pose)
            gd_v = v[batch == i]
            gd_u = u[batch == i]
            S_mat = S_key_points[:, batch == i]
            if S_mat.size is 0:
                self.error += 1
                if i == 0:
                    pred_pose_last = np.zeros((1, 3, 4))
                if i != 0:
                    pred_pose = np.zeros((1, 3, 4))
                    pred_pose_last = np.append(pred_pose_last, pred_pose, axis=0)
                continue
            point_idx = np.argmax(S_mat, axis=0)
            pts_2d = np.array([gd_u, gd_v], dtype=np.int32).transpose((1, 0))

            if pts_2d.shape[0] <= 5:
                self.error += 1
                if i == 0:
                    pred_pose_last = np.zeros((1, 3, 4))
                if i != 0:
                    pred_pose = np.zeros((1, 3, 4))
                    pred_pose_last = np.append(pred_pose_last, pred_pose, axis=0)
                continue

            pts_3d = self.pts_3d[point_idx, :]
            pred_pose = self.pnp(pts_3d, pts_2d, K_s)
            with open(pred_path, 'ab') as f:
                np.savetxt(f, pred_pose)
            with open(gt_path, 'ab') as f:
                np.savetxt(f, gt_pose)
            if i == 0:
                pred_pose_last = pred_pose
                pred_pose_last = np.expand_dims(pred_pose_last, axis=0)

            pred_pose = np.expand_dims(pred_pose, axis=0)

            if i != 0:
                pred_pose_last = np.append(pred_pose_last, pred_pose, axis=0)

        return pred_pose_last

    def calculate_tra_and_rot(self, pose, pred_pose, K, cls):
        pose = pose.detach().cpu().numpy()
        K = K.detach().cpu().numpy()
        batch_size = pose.shape[0]
        for i in range(batch_size):
            if pred_pose[i, :, :].all() == 0:
                continue

            if cls in ["obj1", "obj2", 'obj5', 'obj14', 'obj17', 'obj18', 'obj24', 'obj26', 'obj29', 'obj33']:
                flag = self.add_metric(pred_pose[i, :, :], pose[i, :, :], syn=True, onlyflag = True)
            else:
                flag = self.add_metric(pred_pose[i, :, :], pose[i, :, :], onlyflag = True)
            if not flag:
                continue

            if self.args.class_type in ["obj1", "obj2", 'obj5', 'obj14', 'obj17', 'obj18', 'obj24', 'obj26', 'obj29',
                                        'obj33']:
                rot = self.rtDic[int(self.args.class_type[3:]) - 1]
                pred_pose2 = np.dot(pred_pose[i], rot)
                ori = np.linalg.norm(pose[i] - pred_pose[i])
                new = np.linalg.norm(pose[i] - pred_pose2)
                if new < ori:
                    pred_pose[i] = pred_pose2

            rot = pose[i, :, :3]
            tra = pose[i, :, 3:].reshape(1, 3)
            pred_rot = pred_pose[i, :, :3]
            pred_tra = pred_pose[i, :, 3:].reshape(1, 3)
            tra_error = (tra - pred_tra) * 1000

            x_error = math.fabs(tra_error[:, 0])
            y_error = math.fabs(tra_error[:, 1])
            z_error = math.fabs(tra_error[:, 2])

            self.x_error_all.append(x_error)
            self.y_error_all.append(y_error)
            self.z_error_all.append(z_error)

            sy = math.sqrt(rot[2, 1] * rot[2, 1] + rot[2, 2] * rot[2, 2])
            alpha = math.atan2(rot[2, 1], rot[2, 2])
            beta = math.atan2(-rot[2, 0], sy)
            gamma = math.atan2(rot[1, 0], rot[0, 0])

            pred_sy = math.sqrt(pred_rot[2, 1] * pred_rot[2, 1] + pred_rot[2, 2] * pred_rot[2, 2])
            pred_alpha = math.atan2(pred_rot[2, 1], pred_rot[2, 2])
            pred_beta = math.atan2(-pred_rot[2, 0], pred_sy)
            pred_gamma = math.atan2(pred_rot[1, 0], pred_rot[0, 0])

            alpha_error = math.fabs((math.fabs(alpha) - math.fabs(pred_alpha)) * 180 / math.pi)
            beta_error = math.fabs((math.fabs(beta) - math.fabs(pred_beta)) * 180 / math.pi)
            gamma_error = math.fabs((math.fabs(gamma) - math.fabs(pred_gamma)) * 180 / math.pi)

            self.alpha_error_all.append(alpha_error)
            self.beta_error_all.append(beta_error)
            self.gama_error_all.append(gamma_error)

    def calculate_projection2d_add(self, S, mask_pred, mask, pose, K, cls, img):
        mask_pred = mask_pred.detach().cpu().numpy().astype(np.uint8)

        mask_gt = mask.detach().cpu().numpy()

        S = S.permute(1, 0).detach().cpu().numpy()

        pose = pose.detach().cpu().numpy()
        K = K.detach().cpu().numpy()

        valid_mask = np.where(mask_pred)
        batch = valid_mask[0]
        v = valid_mask[1]
        u = valid_mask[2]

        S_key_points = S

        batch_size = mask_pred.shape[0]
        match_loss_ = []
        total_error = []
        total_iou = []
        org_pose = np.concatenate([np.eye(3), np.zeros((3, 1))], axis=-1)
        for i in range(batch_size):
            K_s = K[i]
            gt_pose = pose[i]
            mesh_pts_2d = mesh_project(self.mesh_model["pts"], K_s, gt_pose)
            # iou = (mask_pred[i] & mask_gt[i]).sum() / (mask_pred[i] | mask_gt[i]).sum()
            gd_v = v[batch == i]
            gd_u = u[batch == i]
            S_mat = S_key_points[:, batch == i]
            if S_mat.size is 0:
                self.proj_2d.append(False)
                self.add.append(False)
                continue
            point_idx = np.argmax(S_mat, axis=0)
            pts_2d = np.array([gd_u, gd_v], dtype=np.int32).transpose((1, 0))

            if pts_2d.shape[0] <= 5:
                self.proj_2d.append(False)
                self.add.append(False)
                continue

            pts_3d = self.pts_3d[point_idx, :]
            pred_pose = self.pnp(pts_3d, pts_2d, K_s)

            self.projection_2d(pred_pose, gt_pose, K_s)
            if cls in ["obj1", "obj2", 'obj5', 'obj14', 'obj17', 'obj18', 'obj24', 'obj26', 'obj29', 'obj33']:
                self.add_metric(pred_pose, gt_pose, syn=True)
            else:
                self.add_metric(pred_pose, gt_pose)

    def pnp(self, points_3d, points_2d, camera_matrix, method=cv2.SOLVEPNP_ITERATIVE):  # SOLVEPNP_ITERATIVE

        try:
            dist_coeffs = self.pnp.dist_coeffs
        except:
            dist_coeffs = np.array([[-0.083596], [0.094005], [0.001208], [-0.0008612969], [0]])

        assert (
                points_3d.shape[0] == points_2d.shape[0]
        ), "points 3D and points 2D must have same number of vertices"
        if method == cv2.SOLVEPNP_EPNP:
            points_3d = np.expand_dims(points_3d, 0)
            points_2d = np.expand_dims(points_2d, 0)

        points_2d = np.ascontiguousarray(points_2d.astype(np.float64))
        points_3d = np.ascontiguousarray(points_3d.astype(np.float64))
        camera_matrix = camera_matrix.astype(np.float64)

        if points_2d.shape[0] < 30:
            _, R_exp, t = cv2.solvePnP(
                points_3d, points_2d, camera_matrix, dist_coeffs, flags=method
            )
        else:

            _, R_exp, t, inliers = cv2.solvePnPRansac(
                points_3d, points_2d, camera_matrix, dist_coeffs, reprojectionError=4)
        R, _ = cv2.Rodrigues(R_exp)

        return np.concatenate([R, t / 1000], axis=-1)

    def projection_2d(self, pose_pred, pose_targets, K, threshold=5):
        if self.args.class_type in ["obj1", "obj2", 'obj5', 'obj14', 'obj17', 'obj18', 'obj24', 'obj26', 'obj29',
                                    'obj33']:
            rot = self.rtDic[int(self.args.class_type[3:]) - 1]
            pose_pred2 = np.dot(pose_pred, rot)
            ori = np.linalg.norm(pose_targets - pose_pred)
            new = np.linalg.norm(pose_targets - pose_pred2)
            if new < ori:
                pose_pred = pose_pred2

        model_2d_pred = project(self.mesh_model["pts"], K, pose_pred)
        model_2d_targets = project(self.mesh_model["pts"], K, pose_targets)
        proj_mean_diff = np.mean(
            np.linalg.norm(model_2d_pred - model_2d_targets, axis=-1)
        )

        print(proj_mean_diff)

        if proj_mean_diff < threshold:
            self.true_num = self.true_num + 1
            self.proj_2d_mean.append(proj_mean_diff)
        else:
            self.false_num = self.false_num + 1
        print("True:{} False:{}".format(self.true_num, self.false_num))
        proj2dmean = np.mean(self.proj_2d_mean)
        print("2d projection mean:{}".format(proj2dmean))
        self.proj_2d.append(proj_mean_diff < threshold)

    def add_metric(self, pose_pred, pose_targets, syn=False, percentage=0.1, onlyflag = False):
        diameter = self.diameter * percentage
        model_pred = (
                np.dot(self.mesh_model["pts"], pose_pred[:, :3].T) + pose_pred[:, 3]
        )
        model_targets = (
                np.dot(self.mesh_model["pts"], pose_targets[:, :3].T) + pose_targets[:, 3]
        )

        if syn:
            mean_dist_index = spatial.cKDTree(model_pred)
            mean_dist, _ = mean_dist_index.query(model_targets, k=1)
            mean_dist = np.mean(mean_dist)
        else:
            mean_dist = np.mean(np.linalg.norm(model_pred - model_targets, axis=-1))
        if mean_dist > diameter:
            self.error += 1
        if not onlyflag:
            self.add.append(mean_dist < diameter)
        return mean_dist < diameter

    def cm_degree_5_metric(self, pose_pred, pose_targets, icp=False):
        translation_distance = (
                np.linalg.norm(pose_pred[:, 3] - pose_targets[:, 3]) * 100
        )
        rotation_diff = np.dot(pose_pred[:, :3], pose_targets[:, :3].T)
        trace = np.trace(rotation_diff)
        trace = trace if trace <= 3 else 3
        angular_distance = np.rad2deg(np.arccos((trace - 1.0) / 2.0))
        if icp:
            self.icp_cmd5.append(translation_distance < 5 and angular_distance < 5)
        else:
            self.cmd5.append(translation_distance < 5 and angular_distance < 5)
