import os.path as osp
import os
import numpy as np
import argparse
import torch
import torch_geometric.transforms as T
from torch_geometric.nn import DataParallel
from torch.utils.data import DataLoader
from dataset.rtl import Metal_Dataset
from network.psgmn import psgmn
import time
from eval_metal import evaluator
import warnings
from tensorboardX import SummaryWriter
from BDP import BalancedDataParallel
import goto
from dominate.tags import label

from goto import with_goto
import yaml

warnings.filterwarnings("ignore")

cuda = torch.cuda.is_available()


def train(model, dataloader, optimizer, device, epoch, writer1):
    model.train()
    total_loss = 0.0
    iter = 0
    start = time.time()
    for data in dataloader:
        iter += 1
        if cuda:

            img, mask, pose, K = [x.to(device) for x in data]
        else:
            img, mask, pose, K = data

        loss = model(img, mask, pose, K)

        final_loss = torch.mean(loss['seg']) + torch.mean(loss['match'])
        m_loss = torch.mean(loss['match'])
        match_loss = torch.mean(loss['match']).item()
        seg_loss = torch.mean(loss['seg']).item()

        loss_item = final_loss.item()
        total_loss += loss_item
        if iter % 50 == 0:
            print(f'loss:{loss_item:.4f}  seg_loss:{seg_loss:.4f} match_loss:{match_loss:.4f}')  #

        optimizer.zero_grad()
        final_loss.backward()
        optimizer.step()
    duration = time.time() - start
    writer1.add_scalar('seg_loss', seg_loss, global_step=epoch)
    writer1.add_scalar('match_loss', match_loss, global_step=epoch)
    print('Time cost:{}'.format(duration))
    return total_loss / len(dataloader.dataset)


def load_network(net, optimizer, model_dir, resume=True, epoch=-1, strict=False):
    if not resume:
        return 0
    if not os.path.exists(model_dir):
        return 0
    pths = [int(pth.split(".")[0]) for pth in os.listdir(model_dir) if "pkl" in pth and "opt" not in pth]
    if len(pths) == 0:
        return 0
    if epoch == -1:
        pth = max(pths)
    else:
        pth = epoch

    print("Load model: {}".format(os.path.join(model_dir, "{}.pkl".format(pth))))
    print("Load optimizer: {}".format(os.path.join(model_dir, "opt{}.pkl".format(pth))))
    pretrained_model = torch.load(os.path.join(model_dir, "{}.pkl".format(pth)))
    net.load_state_dict(pretrained_model, strict=strict)
    if osp.exists(osp.join(model_dir, "opt{}.pkl".format(pth))):
        optimizer.load_state_dict(torch.load(os.path.join(model_dir, "opt{}.pkl".format(pth))))

    return pth


def adjust_learning_rate(optimizer, epoch, init_lr, writer1):
    """Sets the learning rate to the initial LR decayed by 10 every 20 epochs"""
    lr = init_lr * (0.5 ** (epoch // 20))
    print("LR:{}".format(lr))
    writer1.add_scalar('lr', lr, global_step=epoch)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


@with_goto
def main(args):
    # load dataset
    index = args.index
    scene_i = args.scene[5:]
    label.begin
    args.scene = 'scene{}'.format(scene_i)
    index_local = index
    if args.train:
        index = index + 1
        obj = "obj{}".format(index)
        args.class_type = obj
    if not osp.exists(osp.join(os.getcwd(), 'runs', args.class_type)):
        os.makedirs(osp.join(os.getcwd(), 'runs', args.class_type))
    writer1 = SummaryWriter('runs/{}'.format(args.class_type))
    if args.train:
        train_set = Metal_Dataset(args.data_path, args.class_type)
        train_loader = DataLoader(
            train_set, batch_size=args.batch_size, shuffle=True, num_workers=8
        )
    else:
        scene_idx = int(scene_i)

        # index=3
        obj_id = sceneDic[scene_idx][index]
        scene_len = sceneDic[scene_idx].__len__()
        if obj_id > 25:
            index = index_local + 1
            if index == scene_len:
                index = 0
                scene_i = int(scene_i) + 1
            goto.begin

        args.class_type = "obj{}".format(obj_id)
        test_set = Metal_Dataset(args.data_path, args.class_type, is_train=False, scene=args.scene, index=index)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=8)

    device = torch.device(
        "cuda:0" if cuda else "cpu"
    )  # +re.split(r",",args.gpu_id)[3]
    dnn_model_dir = osp.join("model2", args.class_type)
    mesh_model_dir = osp.join(args.data_path, 'CADModels', 'ply', "{}.ply".format(args.class_type))
    psgmnet = psgmn(mesh_model_dir)
    if args.eval:
        psgmnet = BalancedDataParallel(0, psgmnet, device_ids=[0, 1])
    if args.train:
        psgmnet = BalancedDataParallel(2, psgmnet, device_ids=[0, 1])
    psgmnet = psgmnet.to(device)
    optimizer = torch.optim.Adam(psgmnet.parameters(), lr=args.lr)

    # code for evaluation
    if args.eval:
        metal_eval = evaluator(args, psgmnet, test_loader, device, obj_index=index)
        load_network(psgmnet, optimizer, dnn_model_dir, epoch=args.used_epoch)
        metal_eval.evaluate()
        index = index_local + 1
        if index == scene_len:
            index = 0
            scene_i = int(scene_i) + 1
            scene_index = scene_i

        goto.begin
        return

    if args.train:

        # start_epoch= 1
        start_epoch = load_network(psgmnet, optimizer, dnn_model_dir) + 1

        for epoch in range(start_epoch, args.epochs + 1):
            print("current class:{}".format(args.class_type))

            loss = train(psgmnet, train_loader, optimizer, device, epoch, writer1)
            writer1.add_scalar('total_loss', loss * args.batch_size, global_step=epoch)

            adjust_learning_rate(optimizer, epoch, args.lr, writer1)

            print(f'Epoch: {epoch:02d}, Loss: {loss * args.batch_size:.4f}')
            if epoch % 50 == 0:

                if not osp.exists(osp.join(os.getcwd(), 'model2', args.class_type)):
                    os.makedirs(osp.join(os.getcwd(), 'model2', args.class_type))
                torch.save(psgmnet.state_dict(), osp.join('model2', args.class_type, '{}.pkl'.format(epoch)))
                torch.save(optimizer.state_dict(), osp.join('model2', args.class_type, 'opt{}.pkl'.format(epoch)))
            if epoch % 150 == 0:
                goto.begin


if __name__ == "__main__":
    root = '/home/wzz/4T/chaoyue/rtl'
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", type=bool, default=False)
    parser.add_argument("--index", type=int, default=25)
    parser.add_argument("--data_path", type=str, default=root)
    parser.add_argument("--class_type", type=str, default="obj1")
    parser.add_argument("--scene", type=str, default="scene24")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--train", type=bool, default=False)
    parser.add_argument("--gpu_id", help="GPU_ID", type=str, default="0,1")
    parser.add_argument("--occ", type=bool, default=False)
    parser.add_argument("--used_epoch", type=int, default=-1)
    args = parser.parse_args()
    if args.eval:
        args.train = False
        args.index = 0
        sceneDic = yaml.load(open(osp.join(args.data_path, 'test512', 'sceneObjs.yml'), 'r'))
    else:
        args.train = True

    main(args)
