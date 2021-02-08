from torchvision import transforms
from torch.utils.data import Dataset
import imgaug.augmenters as iaa
from sklearn.model_selection import KFold
from common import get_annotation, get_binary_image, PATH_TO_TEST
import numpy as np
import torch
import os


train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(512),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.3, saturation=0.2),
    transforms.RandomAffine((-25, 25), translate=(0.07, 0.07)),
    transforms.RandomPerspective(distortion_scale=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(512),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class PneumoniaDataset(Dataset):
    def __init__(self, mode, transform=None):
        self.mode = mode
        if mode in {'train', 'val'}:
            df = get_annotation()
            samples = np.array([sample_id for sample_id in df['patientId']])
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
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
        img = get_binary_image(name_dcm, 'train')
        img = torch.Tensor(img)
        img.unsqueeze_(0)
        img = img.repeat(3, 1, 1)
        if self.transform is not None:
            img = self.transform(img)
        output['features'] = img
        if self.mode != 'infer':
            output['targets'] = torch.tensor(self.targets[index])

        return output
