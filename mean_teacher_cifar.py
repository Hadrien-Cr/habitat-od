"""Mean Teacher (French et al. 2018) adaptation: CIFAR-10 -> CIFAR-10 + Gaussian noise (CIFAR-10-C, severity 5).

A student learns labelled source images and, on unlabelled target images, the confident predictions of its exponential
moving average (the teacher) on another augmented view; the teacher's BN statistics only ever see the target."""
import argparse
import copy

import torch
import torch.nn.functional as F
import torchvision
from torch import nn

BATCH, SOURCE_EPOCHS, ADAPT_EPOCHS, ADAPT_LR = 250, 30, 10, 0.01
SIGMA, CONFIDENCE_THRESHOLD, EMA = 0.1, 0.9, 0.99

def load_cifar(train, device):
    dataset = torchvision.datasets.CIFAR10("data", train=train, download=True)
    images = torch.tensor(dataset.data, device=device).permute(0, 3, 1, 2).float() / 255
    return images, torch.tensor(dataset.targets, device=device)

def add_gaussian_noise(images, seed):
    generator = torch.Generator(images.device).manual_seed(seed)
    return (images + SIGMA * torch.randn(images.shape, generator=generator, device=images.device)).clamp(0, 1)

def augment(images):
    """Random horizontal flip per image and random shift of up to 4 pixels per batch."""
    flipped = torch.where(torch.rand(len(images), 1, 1, 1, device=images.device) < 0.5, images.flip(3), images)
    dx, dy = torch.randint(0, 9, (2,)).tolist()
    return F.pad(flipped, (4, 4, 4, 4), mode="reflect")[:, :, dy:dy + 32, dx:dx + 32]

@torch.no_grad()
def accuracy(model, images, labels):
    model.eval()
    return (torch.cat([model(chunk).argmax(1) for chunk in images.split(1000)]) == labels).float().mean().item()

def train_source(model, images, labels):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 0.1, total_steps=SOURCE_EPOCHS * len(images) // BATCH)
    model.train()
    for epoch in range(SOURCE_EPOCHS):
        for index in torch.randperm(len(images), device=images.device).view(-1, BATCH):
            loss = F.cross_entropy(model(augment(images[index])), labels[index])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        print(f"source epoch {epoch}: loss {loss.item():.3f}")

def consistency_loss(student, teacher, images):
    """Student cross-entropy on one augmented view against the teacher's confident predictions on another."""
    with torch.no_grad():
        confidence, pseudo_labels = teacher(augment(images)).softmax(1).max(1)
    mask = (confidence >= CONFIDENCE_THRESHOLD).float()
    return (F.cross_entropy(student(augment(images)), pseudo_labels, reduction="none") * mask).mean(), mask.mean()

def mean_teacher(teacher, source, target_images, noisy_test, source_free):
    student = copy.deepcopy(teacher)
    optimizer = torch.optim.SGD(student.parameters(), lr=ADAPT_LR, momentum=0.9, weight_decay=5e-4)
    for epoch in range(ADAPT_EPOCHS):
        student.train()
        teacher.train()  # its BN running statistics track the target batches it labels
        confident = []
        batches = (torch.randperm(len(target_images), device=target_images.device).view(-1, BATCH) for _ in range(2))
        for target_index, source_index in zip(*batches):
            optimizer.zero_grad()
            loss, confident_fraction = consistency_loss(student, teacher, target_images[target_index])
            loss.backward()  # one forward/backward per domain: each is normalized by its own batch statistics
            if not source_free:
                F.cross_entropy(student(augment(source[0][source_index])), source[1][source_index]).backward()
            optimizer.step()
            with torch.no_grad():  # teacher <- EMA * teacher + (1 - EMA) * student
                for teacher_weight, student_weight in zip(teacher.parameters(), student.parameters()):
                    teacher_weight.lerp_(student_weight, 1 - EMA)
            confident.append(confident_fraction)
        print(f"mean teacher epoch {epoch}: noisy test {accuracy(teacher, *noisy_test):.1%}  "
              f"confident target {torch.stack(confident).mean():.0%}")

def main(args):
    (images, labels), (test_images, test_labels) = (load_cifar(train, args.device) for train in (True, False))
    source = images[:25000], labels[:25000]
    target_images = add_gaussian_noise(images[25000:], seed=1)  # disjoint from source; its labels are never used
    noisy_test = add_gaussian_noise(test_images, seed=2), test_labels
    model = torchvision.models.resnet18(num_classes=10)
    model.conv1, model.maxpool = nn.Conv2d(3, 64, 3, padding=1, bias=False), nn.Identity()  # for 32x32 images
    train_source(model.to(args.device), *source)
    print(f"source only: clean test {accuracy(model, test_images, test_labels):.1%}  "
          f"noisy test {accuracy(model, *noisy_test):.1%}")
    teacher = copy.deepcopy(model).train()
    with torch.no_grad():  # AdaBN (Li et al. 2016): BN running statistics re-estimated on the target
        for chunk in target_images.split(BATCH):
            teacher(chunk)
    print(f"AdaBN: noisy test {accuracy(teacher, *noisy_test):.1%}")
    mean_teacher(teacher, source, target_images, noisy_test, args.source_free)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--source-free", action="store_true", help="adapt without the source labels")
    main(parser.parse_args())
