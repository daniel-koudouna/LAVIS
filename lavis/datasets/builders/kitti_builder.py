from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder

from lavis.common.registry import registry
from lavis.datasets.datasets.kitti_datasets import KittiDataset


@registry.register_builder("kitti")
class KITTIVQABuilder(BaseDatasetBuilder):
    train_dataset_cls = KittiDataset
    eval_dataset_cls = KittiDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/kitti/defaults.yaml",
        "eval": "configs/datasets/kitti/defaults.yaml",
    }
