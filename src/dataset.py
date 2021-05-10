from torch.utils.data import Dataset
from sklearn.model_selection import KFold
from common import get_annotation, get_binary_image, PATH_TO_TEST
from PIL import Image
import torch
import cv2
import numpy as np
import glob
import os


dict_target = {
    "normal": 0,
    "pneumonia": 1,
    "other": 2
}


class KaggleDataset(Dataset):
    def __init__(self, mode, transform=None):
        self.mode = mode
        if mode in {'train', 'val'}:
            df = get_annotation()
            samples = np.array([sample_id for sample_id in df['patientId']])
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            kf.split(samples).__next__()
            train_ind, val_ind = kf.split(samples).__next__()
            if mode == 'train':
                self.sample_ids = list(samples[train_ind])
                self.targets = list(np.array(df['target'])[train_ind])
            else:
                self.sample_ids = list(samples[val_ind])
                self.targets = list(np.array(df['target'])[val_ind])
        else:
            path = PATH_TO_TEST
            self.sample_ids = [os.path.join(path, sample) for sample in os.listdir(path)]

        self.transform = transform

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, index):
        output = dict()
        name_dcm = self.sample_ids[index]
        key = name_dcm.split('/')[-2]
        # img = get_binary_image(name_dcm, 'train')
        # img = torch.Tensor(img)
        img = cv2.imread(name_dcm)
        # img.unsqueeze_(0)
        # img = img.repeat(3, 1, 1)
        if self.transform is not None:
            img = self.transform(img)
        output['features'] = img
        output['targets'] = dict_target[key]
        # if self.mode != 'infer':
        #     output['targets'] = torch.tensor(self.targets[index])

        return output


class PneumoniaDataset(Dataset):
    def __init__(self, mode, path, transform=None):
        self.mode = mode
        samples = []
        labels = []
        for c in dict_target.keys():
            path_to_data = os.path.join(path, c)
            files = glob.glob(os.path.join(path_to_data, '*.png'))
            samples.extend(files)
            labels.extend([dict_target[c]] * len(files))
        if mode in {'train', 'val'}:
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            train_ind, val_ind = kf.split(samples).__next__()
            if mode == 'train':
                self.sample_ids = list(np.array(samples)[train_ind])
                self.targets = list(np.array(labels)[train_ind])
            else:
                self.sample_ids = list(np.array(samples)[val_ind])
                self.targets = list(np.array(labels)[val_ind])

        self.transform = transform

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, index):
        img_path = self.sample_ids[index]
        target = self.targets[index]
        # key = img_path.split('/')[-2]
        img = Image.open(img_path).convert('L')
        # img_bgr = cv2.imread(img_path, 1)
        # img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            img = self.transform(img)

        # if self.transform is not None:
        #     augmented = self.transform(image=img)
        #     img = augmented['image']

        out = {
            "features": img,
            "targets": torch.tensor(target, dtype=torch.long),
        }

        return out
