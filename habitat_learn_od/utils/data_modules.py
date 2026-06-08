
import logging
import os
import pickle
import albumentations as A
import pytorch_lightning as pl # type: ignore
from torch.utils.data.dataloader import DataLoader, Dataset

import shutil
from detectron2.data import transforms as T  # type: ignore
from habitat_learn_od.utils.augmentations import get_transform
from habitat_learn_od.utils.detectron_utils import get_coco_item_dict
from habitat_learn_od.utils.pseudo_labeler import PseudoLabeler
from common.utils.dataset_utils import (
    SampleLoader,
    HabitatDataset,
    HabitatFullDataset,
    HabitatFullSequentialDataset,
    HabitatSequentialDataset,
)
from habitat_learn_od.utils.multi_stage_models import *
from habitat_learn_od.utils.train_helpers import (
    dict_helper_collate,
    list_helper_collate,
)

log = logging.getLogger(__name__)


class HabitatDataModule(pl.LightningDataModule):
    pseudo_labeler: PseudoLabeler

    def __init__(
        self, 
        pseudo_labeler, 
        collection_policy, 
        dataset_path,
        data_base_dir, 
        test_set, 
        transform_type='none', 
        batch_size=8, 
        consecutive_obs=1, 
        train_scenes=None,
        val_scenes=None,
        *args, 
        **kwargs
    ) -> None:
        
        super().__init__()
        self.pseudo_labeler = pseudo_labeler
        self.collection_policy = collection_policy
        self.dataset_path = dataset_path
        self.sampler = None
        self.labels = None
        self.batch_size=batch_size
        self.data_base_dir = data_base_dir
        self.test_set = test_set
        self.transform_type = transform_type
        self.num_workers = 0
        self.consecutive_obs = int(consecutive_obs)
        
        self.train_scenes = train_scenes
        self.val_scenes = val_scenes

    def _get_labels(self, sampler) -> dict:
        """Uses a pseudo_labeler and SinglecamEpisodeFullDataset to get consistent pseudo-labels for all samples in the dataset"""
        val_transform = A.Compose(
            get_transform("none"),
            bbox_params=A.BboxParams(
                format='pascal_voc',
                min_area=0,
                min_visibility=0,
                label_fields=['class_labels', 'infos'],
            ),
        )
        pseudolabel_dataset = SinglecamEpisodeFullDataset(
            None,
            sampler=sampler,
            transform=val_transform,
        )
        pseudolabel_loader = DataLoader(
            pseudolabel_dataset,
            sampler=None,
            pin_memory=False,
            persistent_workers=False,
            shuffle=False,
            batch_size=2 * self.batch_size,
            num_workers=self.num_workers,
            collate_fn=dict_helper_collate,
        )

        pseudolabel_trainer = pl.Trainer(gpus=1)
        self.pseudo_labeler.global_pcds = {}
        model_outs = pseudolabel_trainer.predict(
            self.pseudo_labeler, pseudolabel_loader
        )
        pseudo_labels = self.pseudo_labeler.get_pseudo_labels(
            model_outs, pseudolabel_loader
        )

        coco_pseudo_labels = get_coco_item_dict(pseudo_labels)
        return coco_pseudo_labels

    def prepare_data(self) -> None:
        if not os.path.exists(self.dataset_path):
            assert self.collection_policy is not None, "Collection policy must be provided to generate dataset"
            self.collection_policy.generate(self.dataset_path)

        sampler = self._get_sampler()
        coco_pseudo_labels = self._get_labels(sampler)

        with open("labels.pkl", "wb") as fp:
            pickle.dump(coco_pseudo_labels, fp)

    def _get_sampler(self) -> SampleLoader:
        if os.path.exists(self.dataset_path):
            sampler = SampleLoader(self.dataset_path)
        else:
            path = os.path.join(os.getcwd(), f"dataset")
            sampler = SampleLoader(path)
        return sampler

    def _get_train_dataset(self, sampler: SampleLoader, coco_pseudo_labels) -> Dataset:
        """
        Apply pseudo-labeler and return consistent pseudolabel dataset
        """
        train_transform = A.Compose(
            get_transform(self.transform_type),
            bbox_params=A.BboxParams(
                format='pascal_voc',
                min_area=0,
                label_fields=['class_labels', 'infos', 'gt_logits'],
            ),
        )
        assert len(coco_pseudo_labels) > 0, "No pseudo-labels provided"
        assert len(coco_pseudo_labels) == len(
            sampler
        ), f"Expected {len(sampler)} got {len(coco_pseudo_labels)}"

        train_dataset = PseudoFullDataset(
            dataset_path=None,
            sampler=sampler,
            transform=train_transform,
            pseudo_labels=coco_pseudo_labels,
            consecutive_obs=self.consecutive_obs
        )
        return train_dataset

    def _get_validation_dataset(self) -> Dataset:
        transform = A.Compose(
            [
                A.pytorch.ToTensorV2() # type: ignore
            ],
            bbox_params=A.BboxParams(
                format='pascal_voc',
                label_fields=['class_labels', 'infos'],
            ),
        )

        dataset = HabitatDataset(
            os.path.join(self.data_base_dir, self.test_set),
            transform=transform,
        )
        return dataset

    def setup(self, stage) -> None:
        sampler = self._get_sampler()

        with open('labels.pkl', 'rb') as handle:
            coco_pseudo_labels = pickle.load(handle)
    
        self.train_dataset = self._get_train_dataset(sampler, coco_pseudo_labels)
        self.test_dataset = self._get_validation_dataset()

    def train_dataloader(self) -> DataLoader:
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=list_helper_collate,
            pin_memory=False,
            persistent_workers=False,
            sampler=None,
        )
        return train_loader # type: ignore

    def val_dataloader(self) -> DataLoader:
        test_loader = DataLoader(
            self.test_dataset,
            batch_size = self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=dict_helper_collate,
            pin_memory=False,
            persistent_workers=False,
            sampler=None,
        )
        return test_loader

    def test_dataloader(self) -> DataLoader:
        test_loader = DataLoader(
            self.test_dataset,
            batch_size = self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=dict_helper_collate,
            pin_memory=False,
            persistent_workers=False,
            sampler=None,
        )
        return test_loader


class GTDataModule(HabitatDataModule):
    """Instead of using pseudo-labels, uses ground-truth labels for training the detector."""
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def prepare_data(self) -> None:
        if not os.path.exists(self.dataset_path):
            assert self.collection_policy is not None, "Collection policy must be provided to generate dataset"
            path = os.path.join(os.getcwd(), f"dataset")
            if os.path.exists(path):
                shutil.rmtree(path) # Empty dataset before generating new samples
            self.collection_policy.generate(path)

    def setup(self, stage) -> None:
        sampler = self._get_sampler()
        self.train_dataset = self._get_train_dataset(sampler,)
        self.test_dataset = self._get_validation_dataset()

    def _get_train_dataset(self, sampler: SampleLoader) -> Dataset:
        """Using ground-truth for detector training """
        train_transform = A.Compose(
            get_transform(self.transform_type),
            bbox_params=A.BboxParams(
                format='pascal_voc',
                min_area=0,
                label_fields=['class_labels', 'infos'],
            ),
        )

        inputs = sampler.get_episode_and_steps_dense_list()
        filter_empty_instances = []

        for ep, step in zip(inputs[0], inputs[1]):
            instances = sampler.get_sample(ep, 0, "bbsgt", step).get_bbs_as_gt()
            filter_empty_instances.append(len(instances) > 0)

        return HabitatFullSequentialDataset(
            dataset_path=None,
            sampler=sampler,
            index_mask=filter_empty_instances,
            transform=train_transform,
        )