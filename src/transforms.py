from torchvision.transforms import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


# def pre_transforms():
#     result = [
#         A.Resize(512, 512),
#     ]
#
#     return result
#
#
# def hard_transforms():
#     result = [
#         A.Resize(556, 556, p=1),
#         A.RandomResizedCrop(512, 512, p=1),
#         # A.GaussNoise(var_limit=3. / 255., p=0.33),
#         # A.CoarseDropout(max_holes=5, max_height=25, max_width=25, p=0.33),
#         # A.ElasticTransform(sigma=50, alpha=1, alpha_affine=10, p=0.2),
#         A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.4),
#         # A.IAAPiecewiseAffine(p=0.2),
#         A.ShiftScaleRotate(shift_limit=0.0625, rotate_limit=45, p=0.5),
#         A.HorizontalFlip(p=0.5),
#         # A.VerticalFlip(p=0.5)
#     ]
#
#     return result
#
#
# def post_transforms():
#     return [A.Normalize(), ToTensorV2()]
#
#
# def compose(transforms_to_compose):
#     result = A.Compose([
#         item for sublist in transforms_to_compose for item in sublist
#     ])
#     return result
#
#
# train_transforms = compose([
#     hard_transforms(),
#     post_transforms(),
# ])
#
# test_transforms = compose([pre_transforms(), post_transforms()])


train_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop((224, 224)),
    transforms.RandomHorizontalFlip(),
    # transforms.RandomVerticalFlip(),
    # transforms.ColorJitter(brightness=0.2, contrast=0.3, saturation=0.2),
    transforms.RandomAffine((-25, 25), translate=(0.07, 0.07)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transforms.Normalize(mean=[0.5], std=[0.5])
])

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
