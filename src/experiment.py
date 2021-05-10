from collections import OrderedDict
from catalyst.dl import ConfigExperiment
from src.dataset import PneumoniaDataset, KaggleDataset
from src.transforms import train_transforms, test_transforms


class Experiment(ConfigExperiment):
    def get_datasets(self,
                     stage: str,
                     path_to_dataset: str = "./",
                     **kwargs):
        model_params = self._config['model_params']

        print('-' * 30)
        print('encoder_name:', model_params)
        print('-' * 30)

        datasets = OrderedDict()

        train_set = PneumoniaDataset(mode='train', path=path_to_dataset, transform=train_transforms)

        val_set = PneumoniaDataset(mode='val', path=path_to_dataset, transform=test_transforms)

        # infer_set = PneumoniaDataset(mode='infer', transform=test_transforms)

        datasets["train"] = train_set
        datasets["valid"] = val_set
        # datasets["infer"] = infer_set
        print(datasets)

        return datasets
