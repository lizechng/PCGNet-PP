import os
import sys
import re
import numpy as np
from PIL import Image

sys.path.insert(1, os.path.join(sys.path[0], '..'))


class radar_preprocessing(object):
    def __init__(self, dataset_path, train_name, val_name):
        self.train_paths = {'img': [], 'lidarBev': [], 'lidarHt': [], 'radarMap': [],
                            'fovMsk': [], 'objMsk': [], 'calib': [], 'bevBox': []}
        self.test_paths = {'img': [], 'lidarBev': [], 'lidarHt': [], 'radarMap': [],
                           'fovMsk': [], 'objMsk': [], 'calib': [], 'bevBox': []}
        self.dataset_path = dataset_path
        self.task3_path = 'data'
        self.train_name = train_name
        self.val_name = val_name

    def get_paths(self):
        with open(f'{self.dataset_path}/{self.train_name}', 'r') as train_file:
            train_list = [line.strip() for line in train_file]

        with open(f'{self.dataset_path}/{self.val_name}', 'r') as val_file:
            val_list = [line.strip() for line in val_file]

        self.train_paths['img'] = [os.path.join(self.dataset_path, self.task3_path, 'image', f'{file}.png')
                                   for file in train_list]
        self.train_paths['radarMap'] = [os.path.join(self.dataset_path, self.task3_path, 'radarMap', f'{file}.npy')
                                        for file in train_list]
        self.train_paths['lidarBev'] = [os.path.join(self.dataset_path, self.task3_path, 'lidarBev', f'{file}.npy')
                                        for file in train_list]
        self.train_paths['lidarHt'] = [os.path.join(self.dataset_path, self.task3_path, 'lidarHt', f'{file}.npy')
                                       for file in train_list]
        self.train_paths['calib'] = [os.path.join(self.dataset_path, self.task3_path, 'calib', f'{file}.pickle')
                                     for file in train_list]
        self.train_paths['depth'] = [os.path.join(self.dataset_path, self.task3_path, 'depth', f'{file}.npy')
                                     for file in train_list]
        self.train_paths['fovMsk'] = [os.path.join(self.dataset_path, self.task3_path, 'fovMsk', f'{file}.npy')
                                      for file in train_list]
        self.train_paths['bevBox'] = [os.path.join(self.dataset_path, self.task3_path, 'bevBox', f'{file}.pickle')
                                      for file in train_list]
        self.train_paths['objMsk'] = [os.path.join(self.dataset_path, self.task3_path, 'objMsk', f'{file}.npy')
                                      for file in train_list]
        self.train_paths['camMsk'] = [os.path.join(self.dataset_path, self.task3_path, 'camMsk', f'{file}.npy')
                                      for file in train_list]


        self.test_paths['img'] = [os.path.join(self.dataset_path, self.task3_path, 'image', f'{file}.png')
                                  for file in val_list]
        self.test_paths['radarMap'] = [os.path.join(self.dataset_path, self.task3_path, 'radarMap', f'{file}.npy')
                                       for file in val_list]
        self.test_paths['lidarBev'] = [os.path.join(self.dataset_path, self.task3_path, 'lidarBev', f'{file}.npy')
                                       for file in val_list]
        self.test_paths['lidarHt'] = [os.path.join(self.dataset_path, self.task3_path, 'lidarHt', f'{file}.npy')
                                      for file in val_list]
        self.test_paths['calib'] = [os.path.join(self.dataset_path, self.task3_path, 'calib', f'{file}.pickle')
                                    for file in val_list]
        self.test_paths['depth'] = [os.path.join(self.dataset_path, self.task3_path, 'depth', f'{file}.npy')
                                    for file in val_list]
        self.test_paths['fovMsk'] = [os.path.join(self.dataset_path, self.task3_path, 'fovMsk', f'{file}.npy')
                                     for file in val_list]
        self.test_paths['bevBox'] = [os.path.join(self.dataset_path, self.task3_path, 'bevBox', f'{file}.pickle')
                                     for file in val_list]
        self.test_paths['objMsk'] = [os.path.join(self.dataset_path, self.task3_path, 'objMsk', f'{file}.npy')
                                     for file in val_list]
        self.test_paths['camMsk'] = [os.path.join(self.dataset_path, self.task3_path, 'camMsk', f'{file}.npy')
                                     for file in val_list]

    def prepare_dataset(self):
        self.get_paths()
        print('=' * 77)
        print('Train - [img]     : ', len(self.train_paths['img']))
        print('Train - [radarMap]: ', len(self.train_paths['radarMap']))
        print('Train - [lidarHt] : ', len(self.train_paths['lidarHt']))
        print('Train - [lidarBev]: ', len(self.train_paths['lidarBev']))
        print('Train - [fovMsk]  : ', len(self.train_paths['fovMsk']))
        print('Train - [objMsk]  : ', len(self.train_paths['objMsk']))
        print('Train - [bevBox]  : ', len(self.train_paths['bevBox']))
        print('Train - [calib]   : ', len(self.train_paths['calib']))
        print('Train - [depth]   : ', len(self.train_paths['depth']))
        print('=' * 77)
        print('Test - [img]     : ', len(self.test_paths['img']))
        print('Test - [radarMap]: ', len(self.test_paths['radarMap']))
        print('Test - [lidarHt] : ', len(self.test_paths['lidarHt']))
        print('Test - [lidarBev]: ', len(self.test_paths['lidarBev']))
        print('Test - [fovMsk]  : ', len(self.test_paths['fovMsk']))
        print('Test - [objMsk]  : ', len(self.test_paths['objMsk']))
        print('Test - [bevBox]  : ', len(self.test_paths['bevBox']))
        print('Test - [calib]   : ', len(self.test_paths['calib']))
        print('Test - [depth]   : ', len(self.test_paths['depth']))
        print('=' * 77)

    def compute_mean_std(self):
        mean = np.zeros(3)
        std = np.zeros(3)
        num_pixels = 0

        for raw_img_path in self.train_paths['img']:
            raw_img = Image.open(raw_img_path)
            vec = np.asarray(raw_img) / 255.
            num_pixels = num_pixels + vec.shape[0] * vec.shape[1]
            mean = mean + np.sum(vec, axis=(0, 1))
        mean = mean / num_pixels

        for raw_img_path in self.train_paths['img']:
            raw_img = Image.open(raw_img_path)
            vec = np.asarray(raw_img) / 255.
            std = std + np.sum((vec - mean) ** 2, axis=(0, 1))
        std = np.sqrt(std / num_pixels)

        return mean, std


if __name__ == '__main__':
    import os
    import argparse

    # arguments
    parser = argparse.ArgumentParser(description='Preprocess')
    parser.add_argument('--datapath', default='../')
    args = parser.parse_args()

    dataset = radar_preprocessing(args.datapath)
    dataset.prepare_dataset()
    mean, std = dataset.compute_mean_std()
    print(mean)
    print(std)
