import argparse
import os.path as osp
import os
import torch
import tqdm
import numpy as np
from scipy import spatial
import yaml
import math
import json
from pathlib import Path
from utils import load_ply
import os

os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'



class evaluator:
    def __init__(self, args, class_type):
        self.args = args
        self.class_type = class_type
        self.cad_path = args.cad_path
        self.scene = 1
        with open(osp.join(self.cad_path, 'models_info.json'), "r") as json_file:
            self.model_info = json.load(json_file)
        with open(args.target_path, "r") as json_file:
            self.gt = json.load(json_file)
        self.pred_pose = np.load(args.result_path)
        self.mesh_models = []
        self.generate_meshes()
        self.x_error_all = []
        self.y_error_all = []
        self.z_error_all = []
        self.alpha_error_all = []
        self.beta_error_all = []
        self.gama_error_all = []
        self.error = 0
        self.add = []
        '''
        rtDic
        For partially symmetric objects, the translation and rotation errors and the project_2d
        need to be multiplied by an rt matrix to obtain a new result to eliminate the duality
        '''
        self.rtDic = np.load('is_syn.npy', allow_pickle=True)

    def generate_meshes(self):
        for dir in os.listdir(osp.join(self.cad_path, 'ply')):
            self.mesh_models.append(load_ply(
                osp.join(
                    self.cad_path, 'ply', dir
                )
            ))

    def evaluate(self):
        outpath = os.path.join('./new_result4/{}'.format(self.scene), '{}.txt'.format(self.class_type))
        if not osp.exists('./new_result4/{}'.format(self.scene)):
            os.makedirs('./new_result4/{}'.format(self.scene))

        index = 0
        for i in range(len(self.gt)):
            gt_cur = self.gt[str(i)]
            for j in range(len(gt_cur)):
                rt_pred = self.pred_pose[self.args.is_refine][index]
                index += 1
                r = gt_cur[j]["cam_R_m2c"]
                t = gt_cur[j]["cam_t_m2c"]
                obj_id = gt_cur[j]["obj_id"]
                obj_id = 'obj' + str(obj_id)
                if not obj_id == self.class_type:
                    continue
                r = np.reshape(r, (3, 3))
                t = np.reshape(t, (3, 1))
                rt_target = np.concatenate((r, t), axis=1)
                if obj_id in ["obj1", "obj2", 'obj5', 'obj14', 'obj17', 'obj18', 'obj24', 'obj26', 'obj29', 'obj33']:
                    self.add_metric(rt_pred, rt_target, obj_id, syn=True)
                else:
                    self.add_metric(rt_pred, rt_target, obj_id)
                if not self.add[-1]:
                    self.calculate_tra_and_rot(rt_target, rt_pred, obj_id)
        result = open(outpath, "a")

        add = np.mean(self.add)
        x = np.mean(self.x_error_all)
        y = np.mean(self.y_error_all)
        z = np.mean(self.z_error_all)
        alpha = np.mean(self.alpha_error_all)
        beta = np.mean(self.beta_error_all)
        gamma = np.mean(self.gama_error_all)
        print('x error:{} mm'.format(x), file=result)
        print('y error:{} mm'.format(y), file=result)
        print('z error:{} mm'.format(z), file=result)
        print('alpha error:{} °'.format(alpha), file=result)
        print('beta error:{} °'.format(beta), file=result)
        print('gamaa error:{} °'.format(gamma), file=result)
        print('error number is {}'.format(self.error), file=result)
        print("ADD metric: {}".format(add), file=result)
        result.close()

    def add_metric(self, pose_pred, pose_targets, obj_id, syn=False, percentage=0.1, onlyflag=False):
        diameter = self.model_info[obj_id[3:]]['diameter'] * percentage
        self.mesh_model = self.mesh_models[int(obj_id[3:]) - 1]
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


    def calculate_tra_and_rot(self, pose, pred_pose, cls):

        if pred_pose.all() == 0:
            return

        if cls in ["obj1", "obj2", 'obj5', 'obj14', 'obj17', 'obj18', 'obj24', 'obj26', 'obj29',
                                    'obj33']:
            rot = self.rtDic[int(cls[3:]) - 1]
            pred_pose2 = np.dot(pred_pose, rot)
            ori = np.linalg.norm(pose - pred_pose)
            new = np.linalg.norm(pose - pred_pose2)
            if new < ori:
                pred_pose = pred_pose2

        rot = pose[ :, :3]
        tra = pose[ :, 3:].reshape(1, 3)
        pred_rot = pred_pose[ :, :3]
        pred_tra = pred_pose[ :, 3:].reshape(1, 3)
        tra_error = (tra - pred_tra)


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path", type=str, default='./rtl-0-poses.npy')
    parser.add_argument("--target_path", type=str, default='./scene_gt.json')
    parser.add_argument("--cad_path", type=str, default='./CADmodels')
    parser.add_argument("--is_refine", type=int, default=0)
    args = parser.parse_args()
    for i in range(3,6):
        class_type = 'obj' + str(i)
        a = evaluator(args,class_type)
        a.evaluate()

