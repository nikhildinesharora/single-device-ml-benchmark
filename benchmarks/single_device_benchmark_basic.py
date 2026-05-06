from __future__ import annotations

import argparse
import csv
import json
import platform
import shutil
import socket
import time
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

TINY_IMAGENET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
IMAGE_SIZE = 224
MODEL_CHOICES = ("resnet50", "resnet101", "vit_b_16")
DATASET_CHOICES = ("cifar100", "tiny-imagenet-200")


@dataclass(frozen=True)
class BenchmarkResult:
    hostname: str
    os: str
    machine: str
    device: str
    model: str
    dataset: str
    classes: int
    image_size: int
    batch_size: int
    dataset_samples: int
    measured_batches: int
    warmup_batches: int
    samples: int
    seconds: float
    samples_per_second: float
    seconds_per_batch: float
    estimated_epoch_seconds: float
    estimated_100_epoch_hours: float
    loss: float
    amp: bool


def choose_device(requested: str) -> torch.device:
    requested = requested.lower()
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        mps = getattr(torch, "mps", None)
        if mps is not None and hasattr(mps, "synchronize"):
            torch.mps.synchronize()


def train_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


def maybe_download_tiny_imagenet(data_dir: Path) -> Path:
    root = data_dir / "tiny-imagenet-200"
    train_dir = root / "train"
    if train_dir.exists():
        return root

    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / "tiny-imagenet-200.zip"
    if not archive.exists():
        print(f"[bench] downloading Tiny ImageNet: {TINY_IMAGENET_URL}")
        with urllib.request.urlopen(TINY_IMAGENET_URL, timeout=60) as response:
            with archive.open("wb") as handle:
                shutil.copyfileobj(response, handle)

    print(f"[bench] extracting {archive}")
    with zipfile.ZipFile(archive) as zipped:
        zipped.extractall(data_dir)
    if not train_dir.exists():
        raise FileNotFoundError(f"Tiny ImageNet train directory not found: {train_dir}")
    return root


def load_dataset(name: str, data_dir: Path, download: bool, max_samples: int) -> tuple[torch.utils.data.Dataset, int]:
    transform = train_transform()
    if name == "cifar100":
        dataset = datasets.CIFAR100(
            root=str(data_dir),
            train=True,
            download=download,
            transform=transform,
        )
        classes = 100
    elif name == "tiny-imagenet-200":
        root = maybe_download_tiny_imagenet(data_dir) if download else data_dir / "tiny-imagenet-200"
        train_dir = root / "train"
        if not train_dir.exists():
            raise FileNotFoundError(
                f"Tiny ImageNet is missing at {train_dir}. "
                "Run with --download once, or place tiny-imagenet-200 under the data directory."
            )
        dataset = datasets.ImageFolder(str(train_dir), transform=transform)
        classes = 200
    else:
        raise ValueError(f"unsupported dataset: {name}")

    if max_samples > 0:
        dataset = Subset(dataset, range(min(max_samples, len(dataset))))
    return dataset, classes


def build_model(name: str, classes: int) -> nn.Module:
    if name == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, classes)
        return model
    if name == "resnet101":
        model = models.resnet101(weights=None)
        model.fc = nn.Linear(model.fc.in_features, classes)
        return model
    if name == "vit_b_16":
        return models.vit_b_16(weights=None, num_classes=classes)
    raise ValueError(f"unsupported model: {name}")


def autocast_context(device: torch.device, enabled: bool):
    if not enabled or device.type not in {"cuda", "mps"}:
        return torch.autocast(device_type="cpu", enabled=False)
    return torch.autocast(device_type=device.type, dtype=torch.float16)


def run_one(
    model_name: str,
    dataset_name: str,
    data_dir: Path,
    output_dir: Path,
    device: torch.device,
    batch_size: int,
    batches: int,
    warmup_batches: int,
    max_samples: int,
    num_workers: int,
    download: bool,
    amp: bool,
) -> BenchmarkResult:
    dataset, classes = load_dataset(dataset_name, data_dir, download=download, max_samples=max_samples)
    dataset_samples = len(dataset)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    model = build_model(model_name, classes).to(device)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    print(
        f"[bench] {dataset_name}/{model_name}: "
        f"device={device}, batch={batch_size}, warmup={warmup_batches}, measured={batches}"
    )
    total_loss = 0.0
    total_samples = 0
    measured = 0
    started = 0.0

    iterator = iter(loader)
    target_steps = warmup_batches + batches
    for step in range(target_steps):
        try:
            images, labels = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            images, labels = next(iterator)

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if step == warmup_batches:
            synchronize_device(device)
            started = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, amp):
            outputs = model(images)
            loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if step >= warmup_batches:
            batch_samples = int(images.shape[0])
            total_samples += batch_samples
            total_loss += float(loss.detach().cpu()) * batch_samples
            measured += 1

    synchronize_device(device)
    seconds = max(time.perf_counter() - started, 1e-9)
    result = BenchmarkResult(
        hostname=socket.gethostname(),
        os=platform.system() or "Unknown",
        machine=platform.machine() or "unknown",
        device=str(device),
        model=model_name,
        dataset=dataset_name,
        classes=classes,
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
        dataset_samples=dataset_samples,
        measured_batches=measured,
        warmup_batches=warmup_batches,
        samples=total_samples,
        seconds=seconds,
        samples_per_second=total_samples / seconds,
        seconds_per_batch=seconds / max(1, measured),
        estimated_epoch_seconds=dataset_samples / max(total_samples / seconds, 1e-9),
        estimated_100_epoch_hours=(dataset_samples / max(total_samples / seconds, 1e-9)) * 100 / 3600,
        loss=total_loss / max(1, total_samples),
        amp=amp,
    )
    print(
        f"[bench] done {dataset_name}/{model_name}: "
        f"{result.samples_per_second:.2f} samples/s, "
        f"{result.seconds_per_batch:.3f}s/batch, "
        f"est_epoch={result.estimated_epoch_seconds / 60:.1f}min, "
        f"est_100ep={result.estimated_100_epoch_hours:.1f}h, "
        f"loss={result.loss:.4f}"
    )
    write_result_files(result, output_dir)
    return result


def write_result_files(result: BenchmarkResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    base = (
        f"{stamp}-{result.hostname}-{result.dataset}-{result.model}-"
        f"{result.device.replace(':', '')}"
    ).replace("/", "-").replace("\\", "-")
    json_path = output_dir / f"{base}.json"
    csv_path = output_dir / "single_device_results.csv"
    row = asdict(result)

    json_path.write_text(json.dumps(row, indent=2, sort_keys=True), encoding="utf-8")
    needs_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if needs_header:
            writer.writeheader()
        writer.writerow(row)
    print(f"[bench] saved {json_path}")
    print(f"[bench] appended {csv_path}")


def expand_choice(value: str, choices: Iterable[str]) -> list[str]:
    if value == "all":
        return list(choices)
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single-device training benchmark for model/dataset combinations."
    )
    parser.add_argument("--model", default="all", help="all, or comma-list: resnet50,resnet101,vit_b_16")
    parser.add_argument("--dataset", default="all", help="all, or comma-list: cifar100,tiny-imagenet-200")
    parser.add_argument("--data-dir", default=".data")
    parser.add_argument("--output-dir", default="runs/single_device")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, or mps")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--batches", type=int, default=20)
    parser.add_argument("--warmup-batches", type=int, default=3)
    parser.add_argument("--max-samples", type=int, default=0, help="Limit dataset samples; 0 uses all.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--download", action="store_true", help="Download missing datasets.")
    parser.add_argument("--amp", action="store_true", help="Use autocast mixed precision on CUDA/MPS.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models_to_run = expand_choice(args.model, MODEL_CHOICES)
    datasets_to_run = expand_choice(args.dataset, DATASET_CHOICES)

    unknown_models = sorted(set(models_to_run) - set(MODEL_CHOICES))
    unknown_datasets = sorted(set(datasets_to_run) - set(DATASET_CHOICES))
    if unknown_models:
        raise SystemExit(f"unknown model(s): {', '.join(unknown_models)}")
    if unknown_datasets:
        raise SystemExit(f"unknown dataset(s): {', '.join(unknown_datasets)}")

    device = choose_device(args.device)
    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    print(f"[bench] host={socket.gethostname()} os={platform.system()} machine={platform.machine()}")
    print(f"[bench] torch={torch.__version__} device={device}")

    for dataset_name in datasets_to_run:
        for model_name in models_to_run:
            run_one(
                model_name=model_name,
                dataset_name=dataset_name,
                data_dir=data_dir,
                output_dir=output_dir,
                device=device,
                batch_size=max(1, args.batch_size),
                batches=max(1, args.batches),
                warmup_batches=max(0, args.warmup_batches),
                max_samples=max(0, args.max_samples),
                num_workers=max(0, args.num_workers),
                download=args.download,
                amp=args.amp,
            )


if __name__ == "__main__":
    main()
