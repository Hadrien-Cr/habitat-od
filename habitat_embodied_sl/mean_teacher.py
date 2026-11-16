"""Mean-Teacher self-training baseline for habitat_embodied_sl/main.py's embodied SL (Tarvainen and Valpola 2017, with
Unbiased Teacher's detection recipe): the same unlabeled training frames, none of the collection's labels.

Each step, a moving average of the student (teacher_ema, a copy of the source CustomFT_Detic) detects objects on a batch
of frames; its detections scored at least PSEUDO_LABEL_THRESH become the targets of the student's CustomFT_Detic loss on
a photometrically augmented copy of the frames (strong_augmentation, boxes unchanged). After the optimizer step,
teacher_ema moves toward the student by EMA_MOMENTUM. Evaluation and the reported scores are teacher_ema's.

Usage (defaults: procthor8, config/r50_procthor8_unsup_mean_teacher.yaml, the gt++ variant's frames):
  PYTHONPATH=. python habitat_embodied_sl/mean_teacher.py --config habitat_embodied_sl/config/procthor8_collection.yaml \
      --config-file habitat_embodied_sl/config/r50_procthor8_unsup_mean_teacher.yaml
"""
import copy
from typing import Iterable, Iterator, List

import torch
from detectron2.data import DatasetMapper, build_detection_train_loader, detection_utils  # type: ignore
from detectron2.engine import HookBase  # type: ignore
from detectron2.structures import Instances  # type: ignore
from torch import nn
from torchvision import transforms

from habitat_embodied_sl.main import ObjectViewsTrainer, main, train_argument_parser

PSEUDO_LABEL_THRESH = 0.5
EMA_MOMENTUM = 0.999
PHOTOMETRIC_AUGMENTATION = transforms.Compose([
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0))], p=0.5),
    transforms.RandomErasing(p=0.7, scale=(0.05, 0.2), ratio=(0.3, 3.3), value="random"),
])


def strong_augmentation(image: torch.Tensor) -> torch.Tensor:
    """image (uint8, CxHxW) color-jittered, grayed, blurred and erased at random, its geometry unchanged."""
    return (PHOTOMETRIC_AUGMENTATION(image.float() / 255).clamp(0, 1) * 255).to(torch.uint8)


@torch.no_grad()
def pseudo_labels(teacher_ema: nn.Module, batch: List[dict], score_threshold: float) -> List[Instances]:
    """teacher_ema's detections on batch's images scored at least score_threshold, as training targets in the images'
    own coordinates (no height/width given, so detections aren't rescaled to the original frames)."""
    detections = [output["instances"] for output in teacher_ema([{"image": frame["image"]} for frame in batch])]
    return [
        Instances(instances.image_size, gt_boxes=instances.pred_boxes, gt_classes=instances.pred_classes)
        for instances in (instances[instances.scores >= score_threshold] for instances in detections)
    ]


def pseudo_labelled_batches(batches: Iterable[List[dict]], teacher_ema: nn.Module) -> Iterator[List[dict]]:
    """batches' frames with at least one pseudo label, strongly augmented and with those labels as instances; batches
    with none are skipped, since a loss without targets has no gradient."""
    for batch in batches:
        labelled = [
            {**frame, "image": strong_augmentation(frame["image"].to(teacher_ema.device)), "instances": instances}
            for frame, instances in zip(batch, pseudo_labels(teacher_ema, batch, PSEUDO_LABEL_THRESH)) if len(instances)
        ]
        if labelled:
            yield labelled


class MovingAverageHook(HookBase):
    """After each step, teacher_ema ← momentum · teacher_ema + (1 − momentum) · student, buffers included."""

    def __init__(self, teacher_ema: nn.Module, momentum: float):
        self.teacher_ema = teacher_ema
        self.momentum = momentum

    @torch.no_grad()
    def after_step(self):
        student = self.trainer.model.state_dict()
        for name, value in self.teacher_ema.state_dict().items():
            if value.dtype.is_floating_point:
                value.mul_(self.momentum).add_(student[name], alpha=1 - self.momentum)


class MeanTeacherTrainer(ObjectViewsTrainer):
    """ObjectViewsTrainer training on teacher_ema's pseudo labels of the variant's frames, scoring teacher_ema. Its views
    still feed the logged class counts and class table."""

    def __init__(self, cfg, vocabulary, train_dataset):
        frame_fields = ("file_name", "image_id", "height", "width")
        self.frames = list({record["image_id"]: {key: record[key] for key in frame_fields} for record in train_dataset.records}.values())
        super().__init__(cfg, vocabulary, train_dataset)
        self.checkpointer.add_checkpointable("teacher_ema", self.teacher_ema)

    def build_model(self, cfg):
        model = super().build_model(cfg)
        self.teacher_ema = copy.deepcopy(model).requires_grad_(False).eval()
        return model

    def build_train_loader(self, cfg):
        mapper = DatasetMapper(is_train=True, augmentations=detection_utils.build_augmentation(cfg, is_train=True), image_format="RGB")
        frames = build_detection_train_loader(self.frames, mapper=mapper, total_batch_size=cfg.SOLVER.IMS_PER_BATCH, num_workers=cfg.DATALOADER.NUM_WORKERS)
        return pseudo_labelled_batches(frames, self.teacher_ema)

    def build_hooks(self):
        return [MovingAverageHook(self.teacher_ema, EMA_MOMENTUM), *super().build_hooks()]

    def test(self, cfg, model):
        """ObjectViewsTrainer.test of teacher_ema, the model Mean-Teacher reports, in place of the student the hooks pass."""
        return super().test(cfg, self.teacher_ema)


if __name__ == "__main__":
    parser = train_argument_parser()
    parser.set_defaults(config_file="habitat_embodied_sl/config/r50_procthor8_unsup_mean_teacher.yaml")
    main(parser.parse_args(), MeanTeacherTrainer)
