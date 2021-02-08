import os
import shutil
import glob
from tqdm import tqdm


ROOT = '/Users/nurdinov/Desktop/'
DATA_PATH = os.path.join(ROOT, 'data_pneumonia')


pneumonia = ['01', '02', '04', '14', '15', '16', '17', '29', '30', '33', '34']
normal = ['03', '13', '19', '23', '24', '25', '26', '32', '35']
other = ['05', '08', '09', '10', '18', '20', '21', '22', '27', '28', '31', '36']

extension = '.png'


def concatenate_class():
    count = 0
    folders = glob.glob(os.path.join(ROOT, 'preproc', '*'))
    print(folders)
    for folder in tqdm(folders, desc='FOLDERS'):
        num = os.path.split(folder)[1][0:2]
        samples = glob.glob(os.path.join(folder, '*.png'))
        for sample in tqdm(samples, desc="SAMPLES"):
            if str(num) in pneumonia:
                shutil.move(sample, os.path.join(DATA_PATH, 'pneumonia', str(count) + extension))
                count += 1
            elif str(num) in normal:
                shutil.move(sample, os.path.join(DATA_PATH, 'normal', str(count) + extension))
                count += 1
            elif str(num) in other:
                shutil.move(sample, os.path.join(DATA_PATH, 'other', str(count) + extension))
                count += 1


concatenate_class()
