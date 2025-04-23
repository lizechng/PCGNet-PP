import torch
from time import time
from tensorboardX import SummaryWriter
import numpy as np
import os
from collections import OrderedDict

import torch.nn.functional as F

from .model_dep import PCGNet
from .depth.models import DepthModel as DepthNet
from .models_det import compile_model as DetNet
from .model_occ import compile_model

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


def train(
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
        DetModel_pth=None,

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

    depthModel = DepthNet().cuda()
    depthCheckpoint = torch.load(depthModel_pth)
    depthModel.load_state_dict(depthCheckpoint['model_state_dict'])

    PCGModel = PCGNet(grid_conf, data_aug_conf, outC=2).cuda()
    PCGCheckpoint = torch.load(PCGModel_pth)
    PCGModel.load_state_dict(PCGCheckpoint['model_state_dict'])

    DetModel = DetNet(outC=2).cuda()
    DetCheckpoint = torch.load(DetModel_pth)
    DetModel.load_state_dict(DetCheckpoint['model_state_dict'])

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
    while True:
        np.random.seed()

        # AverageMeter()
        losses = AverageMeter()
        fg_loss = AverageMeter()
        ht_loss = AverageMeter()

        t0 = time()
        for batchi, ret in enumerate(train_loader):

            opt.zero_grad()

            for k in ret:
                ret[k] = ret[k].to(device)

            with torch.no_grad():
                depth = depthModel(ret['img'].to(device))[('disp', 0)][:, 0, ::2, ::2] * 75

                main_msk, main_ht, radar_features, cam_features = PCGModel(
                    ret['img'].to(device),
                    ret['radarMap'].to(device),
                    ret['intrinsic_inv'].to(device),
                    ret['extrinsic_inv'].to(device),
                    depth,
                )

                difFeat, det = DetModel(
                    radar_features,
                    cam_features
                )

            msk, ht, det_msk = model(
                difFeat,
                det
            )

            l_fg = BevCrossEntropyLoss2d(msk, ret['lidarBev'].long(), det_msk)
            l_ht = h_mse(ht[:, 0] * det_msk, ret['lidarHt'][:, :, :, 0] * det_msk) + \
                   h_mse(ht[:, 1] * det_msk, ret['lidarHt'][:, :, :, 1] * det_msk)

            loss = l_fg + l_ht

            # AverageMeter()
            losses.update(loss.item(), ret['img'].size(0))
            fg_loss.update(l_fg.item(), ret['img'].size(0))
            ht_loss.update(l_ht.item(), ret['img'].size(0))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            opt.step()

            counter += 1
            t1 = time()

            if counter % 10 == 0:
                writer.add_scalar('train/loss', loss, counter)
                writer.add_scalar('train/l_fg', l_fg, counter)
                writer.add_scalar('train/l_ht', l_ht, counter)
                writer.add_scalar('train/lr', opt.param_groups[0]['lr'], counter)

        print(f'epoch-{epoch:3d}|{time() - t0:.3f}, '
              f'loss:{losses.avg:.4f}, fg:{fg_loss.avg:.4f}, '
              f'ht:{ht_loss.avg:.4f}')

        # LR plateaued

        scheduler.step(losses.avg)
        print('LR plateaued, hence is set to {}'.format(opt.param_groups[0]['lr']))

        if epoch % 5 == 0 and epoch > 0:
            # AverageMeter()
            # losses = AverageMeter()
            losses = AverageMeter()
            fg_loss = AverageMeter()
            ht_loss = AverageMeter()

            with torch.no_grad():
                for _, ret in enumerate(test_loader):
                    for k in ret:
                        ret[k] = ret[k].to(device)

                    with torch.no_grad():
                        depth = depthModel(ret['img'].to(device))[('disp', 0)][:, 0, ::2, ::2] * 75

                        main_msk, main_ht, radar_features, cam_features = PCGModel(
                            ret['img'].to(device),
                            ret['radarMap'].to(device),
                            ret['intrinsic_inv'].to(device),
                            ret['extrinsic_inv'].to(device),
                            depth,
                        )
                        difFeat, det = DetModel(
                            radar_features,
                            cam_features
                        )

                    msk, ht, det_msk = model(
                        difFeat,
                        det
                    )

                    l_fg = BevCrossEntropyLoss2d(msk, ret['lidarBev'].long(), det_msk)
                    l_ht = h_mse(ht[:, 0] * det_msk, ret['lidarHt'][:, :, :, 0] * det_msk) + \
                           h_mse(ht[:, 1] * det_msk, ret['lidarHt'][:, :, :, 1] * det_msk)

                    loss = l_fg + l_ht

                    # AverageMeter()
                    losses.update(loss.item(), ret['img'].size(0))
                    fg_loss.update(l_fg.item(), ret['img'].size(0))
                    ht_loss.update(l_ht.item(), ret['img'].size(0))

            print(f'epoch-{epoch:3d}| Val, '
                  f'loss:{losses.avg:.4f}, fg:{fg_loss.avg:.4f}, '
                  f'ht:{ht_loss.avg:.4f}')

        if epoch % 5 == 0 and epoch > 0:
            # model.eval()
            print(opt)
            mname = os.path.join(logdir, "model-{}.pth".format(epoch))
            print('saving', mname)
            if multi_gpu:
                checkpoint = {'model_state_dict': model.module.state_dict(),
                              'opt_state_dict': opt.state_dict(),
                              'epoch': epoch}
            else:
                checkpoint = {'model_state_dict': model.state_dict(),
                              'opt_state_dict': opt.state_dict(),
                              'epoch': epoch}
            torch.save(checkpoint, mname)
            # model.train()

        epoch = epoch + 1
