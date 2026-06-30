
import logging
import os
import pickle
import pytorch_lightning as pl # type: ignore
from torch.utils.data.dataloader import DataLoader, Dataset

import shutil
from detectron2.data import transforms as T  # type: ignore
from habitat_learn_od.utils.detectron_utils import get_coco_item_dict
from habitat_learn_od.utils.pseudo_labeler import PseudoLabeler
from common.utils.dataset_utils import (
    SampleLoader,
    HabitatDataset,
    HabitatFullSequentialDataset,
)
from habitat_learn_od.utils.two_stage_models import *
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
        collected_dataset_path,
        test_dataset_path,
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
        self.collected_dataset_path = collected_dataset_path
        self.test_dataset_path = test_dataset_path
        self.sampler = None
        self.labels = None
        self.batch_size=batch_size
        self.num_workers = 0
        self.consecutive_obs = int(consecutive_obs)
        
        self.train_scenes = train_scenes
        self.val_scenes = val_scenes

    def _get_labels(self, sampler) -> dict:
        """Uses a pseudo_labeler and SinglecamEpisodeFullDataset to get consistent pseudo-labels for all samples in the dataset"""
        pseudolabel_dataset = SinglecamEpisodeFullDataset(
            None,
            sampler=sampler,
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
        sampler = self._get_sampler()
        coco_pseudo_labels = self._get_labels(sampler)

        with open("labels.pkl", "wb") as fp:
            pickle.dump(coco_pseudo_labels, fp)

    def _get_sampler(self) -> SampleLoader:
        if os.path.exists(self.collected_dataset_path):
            sampler = SampleLoader(self.collected_dataset_path)
        else:
            path = os.path.join(os.getcwd(), f"dataset")
            sampler = SampleLoader(path)
        return sampler

    def _get_train_dataset(self, sampler: SampleLoader, coco_pseudo_labels) -> Dataset:
        """
        Apply pseudo-labeler and return consistent pseudolabel dataset
        """

        assert len(coco_pseudo_labels) > 0, "No pseudo-labels provided"
        assert len(coco_pseudo_labels) == len(
            sampler
        ), f"Expected {len(sampler)} got {len(coco_pseudo_labels)}"

        train_dataset = PseudoFullDataset(
            dataset_path=None,
            sampler=sampler,
            pseudo_labels=coco_pseudo_labels,
            consecutive_obs=self.consecutive_obs
        )
        return train_dataset

    def _get_validation_dataset(self) -> Dataset:
        dataset = HabitatDataset(
            dataset_path=self.test_dataset_path,
            input_format=self.pseudo_labeler.model.cfg.INPUT.FORMAT,
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
        assert os.path.exists(self.collected_dataset_path), f"Collected dataset path does not exist: {self.collected_dataset_path}"

    def setup(self, stage) -> None:
        sampler = self._get_sampler()
        self.train_dataset = self._get_train_dataset(sampler,)
        self.test_dataset = self._get_validation_dataset()

    def _get_train_dataset(self, sampler: SampleLoader) -> Dataset:
        """Using ground-truth for detector training """
        inputs = sampler.get_episode_and_steps_dense_list()
        filter_empty_instances = []

        for ep, step in zip(inputs[0], inputs[1]):
            instances = sampler.get_sample(ep, 0, "bbsgt", step).get_bbs_as_gt()
            filter_empty_instances.append(len(instances) > 0)

        return HabitatFullSequentialDataset(
            dataset_path=None,
            sampler=sampler,
            index_mask=filter_empty_instances,
        )