from catalyst.dl import SupervisedRunner as Runner
from .experiment import Experiment
from .model import PneumoniaNet
from catalyst.dl import registry


registry.Model(PneumoniaNet)
