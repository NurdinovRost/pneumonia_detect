import os
import pandas as pd
import pydicom


ROOT_RSNA = os.path.abspath(os.path.join('/ws', 'pneumonia-ws', 'data', 'rsna'))
PATH_TO_TRAIN = os.path.join(ROOT_RSNA, 'train')
PATH_TO_TEST = os.path.join(ROOT_RSNA, 'test')
PATH_TO_ANNOTATION = os.path.join(ROOT_RSNA, 'stage_2_train_labels.csv')


def data_handler(df):
    df = df.drop(['x', 'y', 'width', 'height'], axis=1)
    df = df.rename(columns={'Target': "target"})
    df = df.drop_duplicates()
    return df


def get_annotation():
    df = pd.read_csv(PATH_TO_ANNOTATION)
    df = data_handler(df)
    return df


def get_binary_image(file_name, mode):
    file_path = PATH_TO_TRAIN if mode in {'train', 'val'} else PATH_TO_TEST
    file_path = os.path.join(file_path, file_name + '.dcm')
    if os.path.isfile(file_path):
        img = pydicom.read_file(file_path).pixel_array
        return img
