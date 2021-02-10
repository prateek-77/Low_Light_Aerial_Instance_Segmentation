from .coco import CocoDataset
from .registry import DATASETS


@DATASETS.register_module
class iSAID_patches(CocoDataset):

    CLASSES = ('small vehicle', 'large vehicle', 'plane', 'storage tank', 'ship', 'swimming pool', 'harbor', 'tennis court', 'ground track field', 'soccer ball field', 'baseball diamond', 'bridge', 'basketball court', 'roundabout', 'helicopter')
