import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import json
import pickle
import torch
import math
from PIL import Image
import random
import torchvision.transforms.functional as F
from .utils import draw_msra_gaussian, draw_umich_gaussian, gaussian_radius, draw_dense_reg


def get_loader(args, dataset):
    # for k in dataset.train_paths:
    #     dataset.train_paths[k] = dataset.train_paths[k][:100]

    train_dataset = Dataset_loader(dataset.train_paths)
    test_dataset = Dataset_loader(dataset.test_paths)

    train_sampler = None
    val_sampler = None

    train_loader = DataLoader(
        train_dataset, batch_size=args['batch_size'], sampler=train_sampler,
        shuffle=False, num_workers=args['nworkers'],
        pin_memory=True, drop_last=True)

    test_loader = DataLoader(
        test_dataset, batch_size=int(args['val_batch_size']), sampler=val_sampler,
        shuffle=False, num_workers=args['nworkers_val'],
        pin_memory=True, drop_last=True)

    return train_loader, test_loader


class Dataset_loader(Dataset):
    """Dataset with labeled lanes"""
    num_classes = 3
    mean = np.asarray([0.2872368, 0.31290057, 0.31152891]).reshape(1, 1, 3)
    std = np.asarray([0.21084515, 0.23276329, 0.25607278]).reshape(1, 1, 3)

    def __init__(self, data_path):
        self.data_path = data_path

        self.totensor = transforms.ToTensor()
        # self.totensor = torch.from_numpy

        self.max_objs = 64
        self.class_name = ['__background__', 'car', 'truck', 'bus']
        self.label2id = {'car': 0, 'truck': 1, 'bus': 2}
        # self._valid_ids = [1, 2, 3]
        # self.cat_ids = {v: i for i, v in enumerate(self._valid_ids)}
        # self.voc_color = [(v // 32 * 64 + 64, (v // 8) % 4 * 64, v % 8 * 32) \
        #                   for v in range(1, self.num_classes + 1)]

    def __len__(self):

        return len(self.data_path['radarMap'])

    def __getitem__(self, idx):
        radarMap_filename = self.data_path['radarMap'][idx]
        lidarBev_filename = self.data_path['lidarBev'][idx]
        lidarHt_filename = self.data_path['lidarHt'][idx]
        fovMsk_filename = self.data_path['fovMsk'][idx]
        objMsk_filename = self.data_path['objMsk'][idx]
        camMsk_filename = self.data_path['camMsk'][idx]
        calib_filename = self.data_path['calib'][idx]
        depth_filename = self.data_path['depth'][idx]
        bevBox_filename = self.data_path['bevBox'][idx]
        img_filename = self.data_path['img'][idx]

        with open(calib_filename, 'rb') as f:
            calib_data = pickle.load(f)
            intrinsic_inv = calib_data['intrinsic_inv']
            extrinsic_inv = calib_data['extrinsic_inv']

        bevBox_data = pickle.load(open(bevBox_filename, 'rb'))

        img_data = Image.open(img_filename)
        img_data = np.asarray(img_data)

        fovMsk_data = np.load(fovMsk_filename)
        objMsk_data = np.load(objMsk_filename)
        camMsk_data = np.load(camMsk_filename)
        radarMap_data = np.load(radarMap_filename)
        lidarBev_data = np.load(lidarBev_filename)
        lidarHt_data = np.load(lidarHt_filename)
        # depth_data = np.load(depth_filename)[::2, ::2]
        depth_data = np.load(depth_filename)

        # Data Normalization
        img_data = img_data / 255.
        img_data = (img_data - self.mean) / self.std

        fovMsk_data = fovMsk_data / 255.
        objMsk_data = objMsk_data / 255.
        camMsk_data = camMsk_data / 255.
        lidarBev_data = lidarBev_data / 255.

        # CenterNet
        height, width = fovMsk_data.shape[0], fovMsk_data.shape[1]
        num_classes = self.num_classes
        num_objs = min(len(bevBox_data), self.max_objs)

        hm = np.zeros((num_classes, height, width), dtype=np.float32)
        wh = np.zeros((self.max_objs, 2), dtype=np.float32)
        dense_wh = np.zeros((2, height, width), dtype=np.float32)
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        ind = np.zeros((self.max_objs), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
        cat_spec_wh = np.zeros((self.max_objs, num_classes * 2), dtype=np.float32)
        cat_spec_mask = np.zeros((self.max_objs, num_classes * 2), dtype=np.uint8)

        # draw_gaussian = draw_msra_gaussian
        draw_gaussian = draw_umich_gaussian

        gt_det = []
        for k in range(num_objs):
            ann = bevBox_data[k]
            bbox = np.array([ann['x'], ann['y'], ann['x'] + ann['w'], ann['y'] + ann['h']],
                            dtype=np.float32)
            cls_id = self.label2id[ann['class']]

            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, width - 1)  # 1st item and 3rd item
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, height - 1)

            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if h > 0 and w > 0:
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))
                ct = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                ct_int = ct.astype(np.int32)
                draw_gaussian(hm[cls_id], ct_int, radius)
                wh[k] = 1. * w, 1. * h
                ind[k] = ct_int[1] * width + ct_int[0]
                reg[k] = ct - ct_int
                reg_mask[k] = 1
                cat_spec_wh[k, cls_id * 2: cls_id * 2 + 2] = wh[k]
                cat_spec_mask[k, cls_id * 2: cls_id * 2 + 2] = 1

                if False:
                    draw_dense_reg(dense_wh, hm.max(axis=0), ct_int, wh[k], radius)
                gt_det.append([ct[0] - w / 2, ct[1] - h / 2,
                               ct[0] + w / 2, ct[1] + h / 2, 1, cls_id])

        ret = {'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh}
        if False:
            hm_a = hm.max(axis=0, keepdims=True)
            dense_wh_mask = np.concatenate([hm_a, hm_a], axis=0)
            ret.update({'dense_wh': dense_wh, 'dense_wh_mask': dense_wh_mask})
            del ret['wh']
        elif False:
            ret.update({'cat_spec_wh': cat_spec_wh, 'cat_spec_mask': cat_spec_mask})
            del ret['wh']
        if True:
            ret.update({'reg': reg})
        if False:
            gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else \
                np.zeros((1, 6), dtype=np.float32)
            meta = {'c': c, 's': s, 'gt_det': gt_det, 'img_id': img_id}
            ret['meta'] = meta

        ret['intrinsic_inv'] = intrinsic_inv.astype(np.float32)
        ret['extrinsic_inv'] = extrinsic_inv.astype(np.float32)

        ret['img'] = self.totensor(img_data.astype(np.float32))
        ret['fovMsk'] = fovMsk_data.astype(np.float32)
        ret['objMsk'] = objMsk_data.astype(np.float32)
        ret['camMsk'] = camMsk_data.astype(np.float32)
        ret['radarMap'] = self.totensor(radarMap_data.astype(np.float32))
        ret['lidarBev'] = lidarBev_data.astype(np.float32)
        ret['lidarHt'] = lidarHt_data.astype(np.float32)
        ret['depth'] = depth_data.astype(np.float32)

        return ret
