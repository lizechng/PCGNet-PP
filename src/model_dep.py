import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from efficientnet_pytorch import EfficientNet
from torchvision.models.resnet import resnet18, resnet34, resnet50
import numpy as np
from scipy.linalg import pinv, inv

from src.tools import gen_dx_bx, cumsum_trick, QuickCumsum
from src.affinity_propagate import Affinity_Propagate
from src.CenterDetModel import CenterDetNet, Bottleneck, BasicBlock
from src.CenterNetLoss import _transpose_and_gather_feat, _gather_feat
from src.ops.voxel_pooling import voxel_pooling


class SumAttention(nn.Module):

    def __init__(self, in_channels, c_m, c_n, reconstruct=True):
        super().__init__()
        self.in_channels = in_channels
        self.reconstruct = reconstruct
        self.c_m = c_m
        self.c_n = c_n
        self.convQ = nn.Conv2d(c_m, c_m, 1)
        self.convK = nn.Conv2d(c_n, c_n, 1)
        self.convC = nn.Conv2d(c_n, c_n, 1)
        self.convR = nn.Conv2d(c_m, c_m, 1)
        if self.reconstruct:
            self.conv_reconstruct = nn.Conv2d(c_m + c_n, in_channels, kernel_size=1)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, radar, imgs):
        b, _, h, w = radar.shape
        # assert c == self.in_channels
        Q = self.convQ(radar)
        K = self.convK(imgs)
        C = self.convC(imgs)
        R = self.convR(radar)

        tmpQ = Q.view(b, self.c_m, -1)
        tmpK = K.view(b, self.c_n, -1)
        attention_C = C.view(b, self.c_n, -1)
        attention_R = R.view(b, self.c_m, -1)

        global_descriptors = torch.bmm(tmpQ, tmpK.permute(0, 2, 1))
        global_C = F.softmax(global_descriptors, dim=1)
        global_R = F.softmax(global_descriptors.permute(0, 2, 1), dim=1)

        tmpZ = torch.cat((global_C.matmul(attention_C), global_R.matmul(attention_R)), dim=1)
        tmpZ = tmpZ.view(b, self.c_m + self.c_n, h, w)

        if self.reconstruct:
            tmpZ = self.conv_reconstruct(tmpZ)

        return tmpZ


class DifAttention(nn.Module):

    def __init__(self, in_channels, c_m, c_n, reconstruct=True):
        super().__init__()
        self.in_channels = in_channels
        self.reconstruct = reconstruct
        self.c_m = c_m
        self.c_n = c_n
        self.convQ = nn.Conv2d(c_m, c_m, 1)
        self.convK = nn.Conv2d(c_n, c_n, 1)
        self.convC = nn.Conv2d(c_n, c_n, 1)
        self.convR = nn.Conv2d(c_m, c_m, 1)
        if self.reconstruct:
            self.conv_reconstruct = nn.Conv2d(c_m + c_n, in_channels, kernel_size=1)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, radar, imgs):
        b, _, h, w = radar.shape
        # assert c == self.in_channels
        Q = self.convQ(radar)
        K = self.convK(imgs)
        C = self.convC(imgs)
        R = self.convR(radar)

        tmpQ = Q.view(b, self.c_m, -1)
        tmpK = K.view(b, self.c_n, -1)
        attention_C = C.view(b, self.c_n, -1)
        attention_R = R.view(b, self.c_m, -1)

        global_descriptors = torch.bmm(tmpQ, tmpK.permute(0, 2, 1))
        global_C = F.softmax(-global_descriptors, dim=1)
        global_R = F.softmax(-global_descriptors.permute(0, 2, 1), dim=1)

        tmpZ = torch.cat((global_C.matmul(attention_C), global_R.matmul(attention_R)), dim=1)
        tmpZ = tmpZ.view(b, self.c_m + self.c_n, h, w)

        if self.reconstruct:
            tmpZ = self.conv_reconstruct(tmpZ)

        return tmpZ


def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def _topk(scores, K=40):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind / K).int()
    topk_inds = _gather_feat(
        topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear',
                              align_corners=True)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = torch.cat([x2, x1], dim=1)
        return self.conv(x1)


class CamEncode(nn.Module):
    def __init__(self, C, downsample):
        super(CamEncode, self).__init__()
        self.C = C

        self.trunk = EfficientNet.from_pretrained("efficientnet-b0")
        # self.trunk = EfficientNet.from_name("efficientnet-b0")

        self.up1 = Up(320 + 112, 256)
        self.up2 = Up(256 + 40, 256)
        self.up3 = Up(256 + 24, 256)
        self.up4 = Up(256 + 16, 256)

        self.featurenet = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, self.C, kernel_size=1, padding=0)
        )

    def get_depth_dist(self, x, eps=1e-20):
        return x.softmax(dim=1)

    def get_depth_feat(self, x, depth):
        x = self.get_eff_depth(x)  # B, 3, 256, 960 -> B, 256, 128, 480

        feauture = self.featurenet(x)

        depth = self.get_depth_dist(depth)
        # [B, 1, 71, 128, 480] * [B, 32, 1, 128, 480] -> [B, 32, 71, 128, 480]
        new_x = depth.unsqueeze(1) * feauture.unsqueeze(2)

        return new_x

    def get_eff_depth(self, x):
        # adapted from https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py#L231
        endpoints = dict()
        # Stem
        x = self.trunk._swish(self.trunk._bn0(self.trunk._conv_stem(x)))
        prev_x = x  # shape: B, 32, 128, 480

        # Blocks
        for idx, block in enumerate(self.trunk._blocks):
            # print(idx, block)
            drop_connect_rate = self.trunk._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.trunk._blocks)  # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            # save the last feature of each downsample group
            if prev_x.size(2) > x.size(2):
                endpoints['reduction_{}'.format(len(endpoints) + 1)] = prev_x
            prev_x = x

        # Head
        # reduction_1: B, 16, 128, 480
        # reduction_2: B, 24, 64, 240
        # reduction_3: B, 40, 32, 120
        # reduction_4: 4, 112, 16, 60
        # reduction_5: 4, 320, 8, 30
        endpoints['reduction_{}'.format(len(endpoints) + 1)] = x
        x = self.up1(endpoints['reduction_5'], endpoints['reduction_4'])
        x = self.up2(x, endpoints['reduction_3'])
        x = self.up3(x, endpoints['reduction_2'])
        x = self.up4(x, endpoints['reduction_1'])
        return x

    def forward(self, x, depth):
        x = self.get_depth_feat(x, depth)

        return x


class BevEncode(nn.Module):
    def __init__(self, inC, outC):
        super(BevEncode, self).__init__()

        # trunk = resnet18(pretrained=False, zero_init_residual=True)
        trunk = resnet34(pretrained=False, zero_init_residual=True)
        self.conv1 = nn.Conv2d(inC, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = trunk.bn1
        self.relu = trunk.relu

        self.layer1 = trunk.layer1
        self.layer2 = trunk.layer2
        self.layer3 = trunk.layer3

        # self.up1 = Up(64 + 256, 256, scale_factor=4)
        self.up1 = Up(64 + 256, 256, scale_factor=(150 / 38, 4))
        self.up2_1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',
                        align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.up2_2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',
                        align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.ht_head = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, kernel_size=1, padding=0),
        )
        self.msk_head = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, kernel_size=1, padding=0),
        )

    def forward(self, x):
        x = self.conv1(x)  # B, 64, 150, 300
        x = self.bn1(x)
        x = self.relu(x)

        x1 = self.layer1(x)  # B, 64, 150, 300
        x = self.layer2(x1)  # B, 128, 75, 150
        x = self.layer3(x)  # B, 256, 38, 75

        x = self.up1(x, x1)  # B, 256, 150, 300

        lb = self.up2_1(x)
        lb = self.msk_head(lb)

        ht = self.up2_2(x)
        ht = self.ht_head(ht)

        return lb, ht


class RadarEncode(nn.Module):
    def __init__(self, inC, outC):
        super(RadarEncode, self).__init__()

        # trunk = resnet18(pretrained=False, zero_init_residual=True)
        trunk = resnet34(pretrained=False, zero_init_residual=True)
        self.conv1 = nn.Conv2d(inC, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = trunk.bn1
        self.relu = trunk.relu

        self.layer1 = trunk.layer1
        self.layer2 = trunk.layer2
        self.layer3 = trunk.layer3

        # self.up1 = Up(64 + 256, 256, scale_factor=4)
        self.up1 = Up(64 + 256, 256, scale_factor=(150 / 38, 4))
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',
                        align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, outC, kernel_size=1, padding=0),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x1 = self.layer1(x)
        x = self.layer2(x1)
        x = self.layer3(x)

        x = self.up1(x, x1)
        x = self.up2(x)

        return x[:, :32], x[:, 32:]


class DepthEncode(nn.Module):
    def __init__(self, inC, outC):
        super(DepthEncode, self).__init__()

        trunk = resnet50(pretrained=False, zero_init_residual=True)
        self.conv1 = nn.Conv2d(inC, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = trunk.bn1
        self.relu = trunk.relu

        self.layer1 = trunk.layer1
        self.layer2 = trunk.layer2
        self.layer3 = trunk.layer3
        self.layer4 = trunk.layer4

        # self.up1 = Up(64 + 256, 256, scale_factor=4)
        self.up1 = Up(1024 + 512, 256, scale_factor=(2, 2))
        self.up2 = Up(256 + 256, 256, scale_factor=(2, 2))
        self.head_conv = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, outC, kernel_size=1, padding=0),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)  # B, 64, 128, 480

        x1 = self.layer1(x)  # B, 256, 128, 480
        x2 = self.layer2(x1)  # B, 512, 64, 240
        x = self.layer3(x2)  # B, 1024, 32, 120

        x = self.up1(x, x2)
        x = self.up2(x, x1)

        x = self.head_conv(x)

        return x


class RadarPts(nn.Module):
    def __init__(self, grid_conf, data_aug_conf, outC):
        super(RadarPts, self).__init__()
        self.grid_conf = grid_conf
        self.data_aug_conf = data_aug_conf

        dx, bx, nx = gen_dx_bx(self.grid_conf['xbound'],
                               self.grid_conf['ybound'],
                               self.grid_conf['zbound'],
                               )
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)

        self.downsample = 2
        self.camC = 32
        self.radC = 32
        self.frustum = self.create_frustum()
        self.D, _, _, _ = self.frustum.shape
        self.CameraEncoder = CamEncode(self.camC, self.downsample)
        self.BevEncoder = BevEncode(inC=self.camC, outC=outC)
        # self.DepEncoder = DepthEncode(inC=3, outC=self.D)
        self.OccludedEncoder = BevEncode(inC=self.camC, outC=outC)
        self.RadarEncoder = RadarEncode(inC=2, outC=self.radC + 8)

        self.affinity_propagate = Affinity_Propagate(3, 3)

        # Feauture Alignment
        self.radar_align_main = nn.Conv2d(32, 32, kernel_size=1, padding=0)
        self.camera_align_main = nn.Conv2d(32, 32, kernel_size=1, padding=0)

        self.sumAmp = SumAttention(32, self.radC, self.camC, True)
        # self.difAmp = DifAttention(32, self.radC, self.camC, True)

    def create_frustum(self):
        # make grid in image plane
        ogfH, ogfW = self.data_aug_conf['final_dim']  # [256, 960]
        fH, fW = ogfH // self.downsample, ogfW // self.downsample  # [128, 480]

        ds = torch.arange(*self.grid_conf['dbound'], dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)

        D, _, _ = ds.shape  # [71, 128, 480]

        xs = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH,
                                                                                      fW)  # from 0 to 959, 480 bins
        ys = torch.linspace(0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH,
                                                                                      fW)  # from 0 to 255, 128 bins

        # D x H x W x 3
        frustum = torch.stack((xs, ys, ds), -1)  # each location is [u, v, d]
        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(self, imgs, intrinsic_inv, extrinsic_inv):
        """
        Determine the (x,y,z) locations (in the ego frame) of the points in the point cloud.
        Returns Batch x DepthBins x H/downsample x W/downsample x 3
        """
        B, _, h_, w_ = imgs.shape

        # B x D x H x W x 3
        D, H, W, _ = self.frustum.shape  # [71, 128, 480, 3]
        points = self.frustum.repeat(B, 1, 1, 1, 1)

        # cam_to_ego
        uvd = torch.cat((points[:, :, :, :, :1] * points[:, :, :, :, 2:3],
                         points[:, :, :, :, 1:2] * points[:, :, :, :, 2:3],
                         points[:, :, :, :, 2:3],
                         ), 4).unsqueeze(-1)
        K_inv = intrinsic_inv.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        RT_inv = extrinsic_inv.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        uvz = torch.matmul(K_inv, uvd).squeeze(-1)
        uvz1 = torch.cat((uvz[:, :, :, :, :3],
                          torch.ones_like(uvz[:, :, :, :, 2:3])
                          ), 4).unsqueeze(-1)
        xyz1 = torch.matmul(RT_inv, uvz1).squeeze(-1)

        return xyz1[..., :3]

    def get_cam_feats(self, x, depth):

        x = self.CameraEncoder(x, depth)
        x = x.permute(0, 2, 3, 4, 1)  # [B, D, H, W, C]

        return x

    def VoxelPooling(self, geom_feats, x):
        B, D, H, W, C = x.shape  # [B, 71, 128, 480, 32]

        x = x.reshape(B, -1, C).contiguous()

        geom_feats = ((geom_feats - (self.bx - self.dx / 2.)) / self.dx).int()

        geom_feats = geom_feats.reshape(B, -1, 3).contiguous()

        final = voxel_pooling(
            geom_feats, x,
            torch.tensor([600, 300, 1], dtype=torch.int, device=x.device))

        return final

    def get_voxels(self, imgs, intrinsic_inv, extrinsic_inv, ret_dep):
        geom = self.get_geometry(imgs, intrinsic_inv, extrinsic_inv)  # [B, 71, 128, 480, 3]

        # depth = self.DepEncoder(imgs)

        bins = torch.linspace(4, 74, 71).view(1, 71, 1, 1).to(ret_dep.device)
        ret_expanded = ret_dep.unsqueeze(1).long()

        ret_dis = -(ret_expanded - bins) ** 2 / 0.2

        x = self.get_cam_feats(imgs, ret_dis)  # [B, 71, 128, 480], [B, 71, 128, 480, 32]

        x = self.VoxelPooling(geom, x)

        return ret_dis, x

    def ctdet_post_process(self, dets, num_classes):
        # dets: batch x max_dets x dim
        # return 1-based class det dict
        ret = []
        for i in range(dets.shape[0]):
            top_preds = {}

            # dets[i, :, :2] = transform_preds(
            #     dets[i, :, 0:2], c[i], s[i], (w, h))
            # dets[i, :, 2:4] = transform_preds(
            #     dets[i, :, 2:4], c[i], s[i], (w, h))
            classes = dets[i, :, -1]
            for j in range(num_classes):
                inds = (classes == j)
                # top_preds[j + 1] = np.concatenate([
                #     dets[i, inds, :4].astype(np.float32),
                #     dets[i, inds, 4:5].astype(np.float32)], axis=1).tolist()
                top_preds[j + 1] = torch.cat([dets[i, inds, :4], dets[i, inds, 4:5]], 1)
            ret.append(top_preds)
        return ret

    def ctdet_decode(self, heat, wh, reg, K=100):
        batch, cat, height, width = heat.size()

        # heat = torch.sigmoid(heat)
        # perform nms on heatmaps
        heat = _nms(heat)

        scores, inds, clses, ys, xs = _topk(heat, K=K)
        if reg is not None:
            reg = _transpose_and_gather_feat(reg, inds)
            reg = reg.view(batch, K, 2)
            xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
            ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
        else:
            xs = xs.view(batch, K, 1) + 0.5
            ys = ys.view(batch, K, 1) + 0.5
        wh = _transpose_and_gather_feat(wh, inds)
        wh = wh.view(batch, K, 2)

        clses = clses.view(batch, K, 1).float()
        scores = scores.view(batch, K, 1)
        bboxes = torch.cat([xs - wh[..., 0:1] / 2,
                            ys - wh[..., 1:2] / 2,
                            xs + wh[..., 0:1] / 2,
                            ys + wh[..., 1:2] / 2], dim=2)
        detections = torch.cat([bboxes, scores, clses], dim=2)

        return detections

    def forward(self, imgs, radarMaps, intrinsic_inv, extrinsic_inv, ret_dep):
        depth, img_features = self.get_voxels(imgs, intrinsic_inv, extrinsic_inv, ret_dep)
        radar_features, affinix_matrix = self.RadarEncoder(radarMaps)  # [B, 40, 300, 600]

        cam_features = self.affinity_propagate(affinix_matrix, img_features)

        # Add [convolutional] alignment for radar features and camera features
        radar_features = self.radar_align_main(radar_features)
        cam_features = self.camera_align_main(cam_features)

        sumFeat = self.sumAmp(radar_features, cam_features)

        msk, ht = self.BevEncoder(sumFeat)  # [B, 4, 300, 600]

        return msk, ht, radar_features, cam_features


def PCGNet(grid_conf, data_aug_conf, outC):
    return RadarPts(grid_conf, data_aug_conf, outC)
