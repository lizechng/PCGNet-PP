import torch
from time import time
from tensorboardX import SummaryWriter
import numpy as np
import os
from collections import OrderedDict

import torch.nn.functional as F
import open3d as o3d

from .model_dep import PCGNet
from .depth.models import DepthModel
from .models import compile_model

from .tools import SimpleLoss, get_batch_iou, get_val_info

from .dataloader import get_loader
from .radar_loader import radar_preprocessing
from .CenterNetLoss import CtdetLoss


class CrossEntropyLoss2d(torch.nn.Module):
    def __init__(self, weight=None):
        super(CrossEntropyLoss2d, self).__init__()
        self.loss = torch.nn.NLLLoss(weight)

    def forward(self, outputs, targets):
        return self.loss(torch.nn.functional.log_softmax(outputs, dim=1), targets.long())


def MyCrossEntropyLoss2d(output, target, mask):
    '''
    output: bsz, channels, height, width
    target: bsz, height, width
    mask: bsz, height, width
    '''
    assert target.shape == mask.shape, ''
    log_softmax = torch.nn.LogSoftmax(dim=1)(output)
    bsz, h, w = target.shape
    loss = 0
    for b in range(bsz):
        ind = target[b, :, :].type(torch.int64).unsqueeze(0)
        pred = log_softmax[b, :, :, :]
        pvalue = -pred.gather(0, ind)
        msk = (mask[b:b + 1, :, :] > 0).detach()
        if pvalue[msk].shape[0] > 0:
            loss = loss + torch.mean(pvalue[msk])
        else:
            bsz = bsz - 1
    if bsz == 0:
        return torch.mean(pvalue) * 0
    return loss / bsz


def BevCrossEntropyLoss2d(pred, target, msk):
    msk = msk.bool()

    if msk.sum() == 0:
        return torch.tensor(0.0)

    pred_flat = pred.permute(0, 2, 3, 1).reshape(-1, pred.size(1))
    target_flat = target.reshape(-1)
    msk_flat = msk.reshape(-1)

    pred_masked = pred_flat[msk_flat]
    target_masked = target_flat[msk_flat]

    loss = F.cross_entropy(pred_masked, target_masked)

    return loss


class H_MSELoss(torch.nn.Module):
    def __init__(self):
        super(H_MSELoss, self).__init__()

    def forward(self, prediction, gt):
        err = prediction - gt
        mask = (gt > -100).detach()
        mse_loss = torch.mean((err[mask]) ** 2)
        return mse_loss


class D_MSELoss(torch.nn.Module):
    def __init__(self):
        super(D_MSELoss, self).__init__()

    def forward(self, prediction, gt):
        if prediction.shape[1] == 71:
            depth_bins = torch.linspace(4, 74, 71, device=prediction.device)
            max_indices = torch.argmax(prediction, dim=1)
            depth = depth_bins[max_indices]
            # depth = torch.sum(prediction.softmax() * depth_bins.view(1, -1, 1, 1), dim=1)
            prediction = depth
        err = prediction - gt
        mask = torch.logical_and(gt >= 4, gt <= 75).detach()
        # mse_loss = torch.mean((err[mask]) ** 2)
        mse_loss = torch.mean(abs(err[mask]))
        return mse_loss


class D_CELoss(torch.nn.Module):
    def __init__(self):
        super(D_CELoss, self).__init__()

    def forward(self, pred, target):
        msk = torch.logical_and(target >= 0, target <= 70)

        pred_flat = pred.permute(0, 2, 3, 1).reshape(-1, pred.size(1))
        target_flat = target.reshape(-1)
        msk_flat = msk.reshape(-1)

        pred_masked = pred_flat[msk_flat]
        target_masked = target_flat[msk_flat]

        loss = F.cross_entropy(pred_masked, target_masked)

        return loss


class BCELoss(torch.nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.loss_fn = torch.nn.BCELoss()

    def forward(self, ypred, ytgt, mask):
        loss = self.loss_fn(ypred * mask, ytgt * mask)
        return loss


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def infer(
        gpuid=0,
        H=256, W=960,
        final_dim=(256, 960),
        rand_flip=True,
        max_grad_norm=5.0,
        pos_weight=2.13,
        data_path='./',
        logdir='./runs',

        multi_gpu=False,
        pre_train=False,
        pre_train_path=None,
        resume=True,
        resume_path=None,

        train_name=None,
        val_name=None,

        depthModel_pth=None,
        PCGModel_pth=None,

        xbound=[-75.0, 74.75, 0.25],
        ybound=[74.75, 0, -0.25],
        zbound=[-10.0, 10.0, 40.0],
        dbound=[4.0, 75.0, 1.0],

        bsz=4,
        nworkers=10,
        lr=0.00005,
        weight_decay=1e-7,
):
    grid_conf = {
        'xbound': xbound,
        'ybound': ybound,
        'zbound': zbound,
        'dbound': dbound,
    }
    data_aug_conf = {
        'final_dim': final_dim,
        'H': H, 'W': W,
        'rand_flip': rand_flip,
        'Ncams': 1,
    }

    args = {
        'data_path': data_path,
        'batch_size': bsz,
        'train_name': train_name,
        'val_name': val_name,
        'nworkers': 2,
        'val_batch_size': 1,
        'nworkers_val': 2,
    }

    torch.backends.cudnn.benchmark = True
    device = torch.device('cpu') if gpuid < 0 else torch.device(f'cuda:{gpuid}')
    model = compile_model(outC=2).cuda()

    dataset = radar_preprocessing(args['data_path'], args['train_name'], args['val_name'])
    dataset.prepare_dataset()
    train_loader, test_loader = get_loader(args, dataset)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min',
                                                           factor=0.7,
                                                           threshold=0.001,
                                                           patience=7)

    depthPred = DepthModel().cuda()
    depthCheckpoint = torch.load(depthModel_pth)
    depthPred.load_state_dict(depthCheckpoint['model_state_dict'])

    PCGModel = PCGNet(grid_conf, data_aug_conf, outC=2).cuda()
    PCGCheckpoint = torch.load(PCGModel_pth)
    PCGModel.load_state_dict(PCGCheckpoint['model_state_dict'])

    epoch = 0
    if pre_train:
        checkpoint = torch.load(pre_train_path)
        state_dict = checkpoint['model_state_dict']
        if multi_gpu:
            model.load_state_dict(state_dict)
            model = torch.nn.DataParallel(model).cuda(gpuid)
        else:
            model.load_state_dict(state_dict)
            model = model.cuda(gpuid)
    elif resume:
        checkpoint = torch.load(resume_path)
        state_dict = checkpoint['model_state_dict']
        opt.load_state_dict(checkpoint['opt_state_dict'])
        epoch = checkpoint['epoch']
        if multi_gpu:
            model.load_state_dict(state_dict)
            model = torch.nn.DataParallel(model).cuda(gpuid)
        else:
            model.load_state_dict(state_dict)
            model = model.cuda(gpuid)
    else:
        if multi_gpu:
            model = torch.nn.DataParallel(model).cuda(gpuid)
        else:
            model = model.cuda(gpuid)

    h_mse = H_MSELoss().cuda(gpuid)
    d_ce = D_CELoss().cuda(gpuid)
    d_mse = D_MSELoss().cuda(gpuid)
    l_det = CtdetLoss().cuda(gpuid)

    writer = SummaryWriter(logdir=logdir)

    model.train()
    counter = 0
    with torch.no_grad():
        while True:
            np.random.seed()

            # AverageMeter()
            losses = AverageMeter()
            box_loss = AverageMeter()
            fg_loss = AverageMeter()
            ht_loss = AverageMeter()

            t0 = time()

            for batchi, ret in enumerate(train_loader):

                opt.zero_grad()

                for k in ret:
                    ret[k] = ret[k].to(device)

                depth = depthPred(ret['img'].to(device))[('disp', 0)][:, 0, ::2, ::2] * 75

                main_msk, main_ht, radar_features, cam_features = PCGModel(
                    ret['img'].to(device),
                    ret['radarMap'].to(device),
                    ret['intrinsic_inv'].to(device),
                    ret['extrinsic_inv'].to(device),
                    depth,
                )

                occ_msk, occ_ht, occ_box = model(
                    radar_features,
                    cam_features
                )

                infer_vis = True
                if infer_vis:
                    bevMsk = main_msk[0, :, :, :].argmax(dim=0).detach().cpu().numpy()
                    bevHt = main_ht[0, :, :, :].permute(1, 2, 0).detach().cpu().numpy()
                    fovMsk = ret['fovMsk'][0].detach().cpu().numpy()
                    lidarBev = ret['lidarBev'][0].detach().cpu().numpy()
                    lidarHt = ret['lidarHt'][0].detach().cpu().numpy()
                    bevMsk = bevMsk * fovMsk
                    lidarBev = lidarBev * fovMsk

                    points, gt = [], []
                    for i in range(bevMsk.shape[0]):
                        for j in range(bevMsk.shape[1]):
                            if bevMsk[i, j] == 1:
                                if bevHt[i, j, 1] > bevHt[i, j, 0]:
                                    for h in np.arange(bevHt[i, j, 0], bevHt[i, j, 1] + 0.1, 0.1):
                                        points.append([j * 0.25 - 75, (299 - i) * 0.25, h])
                                else:
                                    points.append([j * 0.25 - 75, (299 - i) * 0.25, bevHt[i, j, 0]])
                    points = np.asarray(points)

                    for i in range(lidarBev.shape[0]):
                        for j in range(lidarBev.shape[1]):
                            if lidarBev[i, j] == 1:
                                if lidarHt[i, j, 1] > lidarHt[i, j, 0]:
                                    for h in np.arange(lidarHt[i, j, 0], lidarHt[i, j, 1] + 0.01, 0.01):
                                        gt.append([j * 0.25 - 75, (299 - i) * 0.25, h])
                                else:
                                    gt.append([j * 0.25 - 75, (299 - i) * 0.25, lidarHt[i, j, 0]])
                    gt = np.asarray(gt)

                    print(points.shape, gt.shape)

                    pts_color = np.asarray([[1, 0, 0] for i in range(points.shape[0])])
                    gt_color = np.asarray([[0, 1, 0] for i in range(gt.shape[0])])

                    vis = o3d.visualization.Visualizer()
                    vis.create_window(window_name='vis', width=1920, height=1080)
                    pointcloud = o3d.geometry.PointCloud()
                    pointcloud.points = o3d.utility.Vector3dVector(points)
                    pointcloud.colors = o3d.utility.Vector3dVector(pts_color)
                    vis.add_geometry(pointcloud)

                    pointcloud = o3d.geometry.PointCloud()
                    pointcloud.points = o3d.utility.Vector3dVector(gt)
                    pointcloud.colors = o3d.utility.Vector3dVector(gt_color)
                    vis.add_geometry(pointcloud)

                    vis.get_render_option().background_color = np.asarray([0.85, 0.85, 0.85])
                    vis.get_render_option().point_size = 2
                    vis.get_render_option().show_coordinate_frame = False
                    vis.run()
                    vis.destroy_window()

                print(occ_box)
                raise None
                # Detection Box Visualization
                im = ret['lidarBev'].detach().cpu().numpy()[0]
                boxes = occ_box[0].detach().cpu().numpy()[0]

                from PIL import Image
                import cv2
                # Image.fromarray(np.uint8(im*255)).show()
                im_RGB = np.stack([im*255, im*255, im*255], axis=-1)
                print(im_RGB.shape)
                for b in boxes:
                    x1, y1, x2, y2, score, cls = b
                    if score < 0.6:
                        continue
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    cv2.rectangle(im_RGB, (x1, y1), (x2, y2), (255, 0, 0), 2)
                Image.fromarray(np.uint8(im_RGB)).show()

                raise None

                l_box, lbx = l_det(occ_box, ret)

                MSK = ret['lidarBev'] * ret['objMsk']
                l_fg = BevCrossEntropyLoss2d(occ_msk, MSK.long(), ret['fovMsk'])
                l_ht = h_mse(occ_ht[:, 0] * MSK, ret['lidarHt'][:, :, :, 0] * MSK) + \
                       h_mse(occ_ht[:, 1] * MSK, ret['lidarHt'][:, :, :, 1] * MSK)

                loss = l_box + l_fg + l_ht

                # AverageMeter()
                box_loss.update(l_box.item(), ret['img'].size(0))
                fg_loss.update(l_fg.item(), ret['img'].size(0))
                ht_loss.update(l_ht.item(), ret['img'].size(0))
                losses.update(loss.item(), ret['img'].size(0))

                counter += 1
                t1 = time()

            print(f'epoch-{epoch:3d}|{time() - t0:.3f}, '
                  f'box:{box_loss.avg:.4f}, foreground:{fg_loss.avg:.4f}, '
                  f'height:{ht_loss.avg:.4f}, '
                  f'total:{losses.avg:.4f}')
            break
            # LR plateaued
