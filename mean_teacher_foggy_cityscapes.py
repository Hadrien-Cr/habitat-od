"""Mean Teacher object detection adaptation: Cityscapes -> Foggy Cityscapes (beta 0.02), 8 classes, mAP@0.5.

Unbiased Teacher recipe (Liu et al. 2021): a COCO-pretrained Faster R-CNN is fine-tuned on labelled clean images, then a
student keeps learning them plus, on unlabelled foggy images, the teacher's detections scoring at least SCORE_THRESHOLD,
on a photometrically augmented copy (same geometry, so the boxes stay valid; classification losses only, since pseudo
boxes are too noisy to regress). The teacher is an exponential moving average (EMA) of the student.

--data holds gtFine/, leftImg8bit/ and leftImg8bit_foggy/ from https://www.cityscapes-dataset.com/downloads/"""
import argparse
import copy
import glob
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from PIL import Image
from torchmetrics.detection import MeanAveragePrecision
from torchvision import transforms
from torchvision.io import read_image
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import masks_to_boxes
from torchvision.transforms import functional as TF

# Cityscapes label id -> class: person, rider, car, truck, bus, train, motorcycle, bicycle
CLASSES = {24: 1, 25: 2, 26: 3, 27: 4, 28: 5, 31: 6, 32: 7, 33: 8}
HEIGHT, WIDTH, BATCH, LR = 600, 1200, 2, 0.005
SOURCE_STEPS, ADAPT_STEPS, EVAL_EVERY, SCORE_THRESHOLD, EMA = 6000, 6000, 1000, 0.8, 0.999
STRONG_AUGMENTATION = transforms.Compose([
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomApply([transforms.GaussianBlur(9, sigma=(0.1, 2.0))], p=0.5),
    transforms.RandomErasing(p=0.7, scale=(0.05, 0.2), ratio=(0.3, 3.3), value="random"),
])

def load(image_path, instance_path):
    """Image (uint8) and its boxes, from the instance id map (label id * 1000 + instance), both resized."""
    image = TF.resize(read_image(image_path), [HEIGHT, WIDTH], antialias=True)
    instance_map = torch.from_numpy(np.array(Image.open(instance_path), dtype=np.int32))
    instance_map = TF.resize(instance_map[None], [HEIGHT, WIDTH], transforms.InterpolationMode.NEAREST)[0]
    instance_ids = [value for value in instance_map.unique().tolist() if value // 1000 in CLASSES]
    boxes = masks_to_boxes(instance_map == torch.tensor(instance_ids, dtype=torch.int32).view(-1, 1, 1))
    boxes[:, 2:] += 1  # masks_to_boxes' corners are inclusive pixels, a one-pixel object would get an empty box
    labels = torch.tensor([CLASSES[value // 1000] for value in instance_ids], dtype=torch.int64)
    return image, {"boxes": boxes, "labels": labels}

def load_split(root, split, foggy, part):
    instance_paths = sorted(glob.glob(f"{root}/gtFine/{split}/*/*_gtFine_instanceIds.png"))[part]
    folder, suffix = ("leftImg8bit_foggy", "_leftImg8bit_foggy_beta_0.02") if foggy else ("leftImg8bit", "_leftImg8bit")
    image_paths = [path.replace(f"{root}/gtFine", f"{root}/{folder}").replace("_gtFine_instanceIds", suffix)
                   for path in instance_paths]
    with ThreadPoolExecutor(16) as pool:
        return list(pool.map(load, image_paths, instance_paths))

def sgd(model):
    return torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=LR, momentum=0.9, weight_decay=1e-4)

def batch(samples, indices, device):
    images = [samples[index][0].to(device).float() / 255 for index in indices]
    return images, [{key: value.to(device) for key, value in samples[index][1].items()} for index in indices]

def random_flip(images, targets):
    """Horizontally flips a random half of images and their boxes."""
    for index in [index for index in range(len(images)) if torch.rand(()) < 0.5]:
        boxes = targets[index]["boxes"][:, [2, 1, 0, 3]] * torch.tensor([-1, 1, -1, 1], device=images[index].device)
        boxes[:, [0, 2]] += WIDTH
        images[index], targets[index] = images[index].flip(-1), {**targets[index], "boxes": boxes}
    return images, targets

def source_loss(model, source, device):
    """Faster R-CNN's losses on a random batch of labelled source images, randomly flipped."""
    return sum(model(*random_flip(*batch(source, torch.randint(len(source), (BATCH,)), device))).values())

@torch.no_grad()
def map50(model, samples, device):
    model.eval()
    metric = MeanAveragePrecision(iou_thresholds=[0.5])
    for indices in torch.arange(len(samples)).split(4):
        images, targets = batch(samples, indices, device)
        metric.update(model(images), targets)
    return metric.compute()["map_50"].item()

def train_source(model, source, device):
    optimizer = sgd(model)
    model.train()
    for step in range(1, SOURCE_STEPS + 1):
        loss = source_loss(model, source, device)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % EVAL_EVERY == 0:
            print(f"source step {step}: loss {loss.item():.3f}")

def mean_teacher(student, source, target_images, foggy_val, args):
    teacher = copy.deepcopy(student).eval()
    optimizer = sgd(student)
    student.train()
    pseudo_boxes = []
    for step in range(1, ADAPT_STEPS + 1):
        indices = torch.randint(len(target_images), (BATCH,))
        images = [target_images[index].to(args.device).float() / 255 for index in indices]
        with torch.no_grad():
            detections = teacher(images)
        pseudo_targets = [{key: detection[key][detection["scores"] >= SCORE_THRESHOLD] for key in ("boxes", "labels")}
                          for detection in detections]
        losses = student([STRONG_AUGMENTATION(image) for image in images], pseudo_targets)
        optimizer.zero_grad()
        (losses["loss_classifier"] + losses["loss_objectness"]).backward()  # one backward per domain saves memory
        if not args.source_free:
            source_loss(student, source, args.device).backward()
        optimizer.step()
        with torch.no_grad():  # teacher <- EMA * teacher + (1 - EMA) * student
            for teacher_weight, student_weight in zip(teacher.parameters(), student.parameters()):
                teacher_weight.lerp_(student_weight, 1 - EMA)
        pseudo_boxes += [len(target["boxes"]) for target in pseudo_targets]
        if step % EVAL_EVERY == 0:
            print(f"mean teacher step {step}: foggy val mAP50 {map50(teacher, foggy_val, args.device):.1%}  "
                  f"pseudo boxes per image {np.mean(pseudo_boxes):.1f}")
            pseudo_boxes = []

def main(args):
    source = load_split(args.data, "train", foggy=False, part=slice(0, None, 2))
    target_images = [image for image, _ in load_split(args.data, "train", foggy=True, part=slice(1, None, 2))]
    clean_val, foggy_val = (load_split(args.data, "val", foggy, slice(None)) for foggy in (False, True))
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT", min_size=HEIGHT, max_size=WIDTH)
    model.roi_heads.box_predictor = FastRCNNPredictor(1024, len(CLASSES) + 1)
    model.to(args.device)
    if os.path.exists(args.checkpoint):
        model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    else:
        train_source(model, source, args.device)
        torch.save(model.state_dict(), args.checkpoint)
    print(f"source only: clean val mAP50 {map50(model, clean_val, args.device):.1%}  "
          f"foggy val {map50(model, foggy_val, args.device):.1%}")
    mean_teacher(model, source, target_images, foggy_val, args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Cityscapes root")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint", default="source_faster_rcnn.pt", help="source model, trained if missing")
    parser.add_argument("--source-free", action="store_true", help="adapt without the source labels")
    main(parser.parse_args())
