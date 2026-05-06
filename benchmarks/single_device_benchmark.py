from __future__ import annotations

import argparse
import csv
import json
import platform
import shutil
import socket
import subprocess
import threading
import time
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from math import ceil
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

TINY_IMAGENET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
IMAGE_SIZE = 224
MODEL_CHOICES = ("resnet50", "resnet101", "vit_b_16")
DATASET_CHOICES = ("cifar100", "tiny-imagenet-200")
DEFAULT_CONFIG: dict[str, Any] = {
    "model": "all",
    "dataset": "all",
    "data_dir": ".data",
    "output_dir": "runs/single_device",
    "device": "auto",
    "batch_size": 8,
    "batches": 20,
    "epochs": 0,
    "batches_per_epoch": 0,
    "warmup_batches": 3,
    "log_every": 100,
    "power_sample_interval": 2.0,
    "max_samples": 0,
    "num_workers": 0,
    "lr": 0.01,
    "momentum": 0.9,
    "weight_decay": 0.0,
    "download": False,
    "amp": False,
}


@dataclass(frozen=True)
class BenchmarkResult:
    run_id: str
    hostname: str
    os: str
    machine: str
    device: str
    world_size: int
    participant_count: int
    model: str
    dataset: str
    classes: int
    image_size: int
    batch_size: int
    epochs: int
    batches_per_epoch: int
    lr: float
    momentum: float
    weight_decay: float
    dataset_samples: int
    measured_batches: int
    warmup_batches: int
    samples: int
    seconds: float
    samples_per_second: float
    throughput: float
    seconds_per_batch: float
    speedup: float
    efficiency: float
    worker_score: float
    worker_scores: str
    estimated_epoch_seconds: float
    estimated_100_epoch_hours: float
    loss: float
    final_batch_loss: float
    avg_power_watts: float | None
    max_power_watts: float | None
    energy_joules: float | None
    power_source: str
    amp: bool


@dataclass(frozen=True)
class ProgressRow:
    run_id: str
    hostname: str
    device: str
    world_size: int
    model: str
    dataset: str
    batch_size: int
    epoch: int
    epoch_batch: int
    measured_batch: int
    samples: int
    interval_seconds: float
    total_seconds: float
    interval_samples_per_second: float
    cumulative_samples_per_second: float
    interval_loss: float
    cumulative_loss: float
    avg_power_watts: float | None
    max_power_watts: float | None
    energy_joules: float | None
    power_source: str


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


class PowerSampler:
    def __init__(self, device: torch.device, interval_seconds: float) -> None:
        self.device = device
        self.interval_seconds = max(0.25, interval_seconds)
        self.values: list[float] = []
        self.source = "not_available"
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self.device.type == "cuda" and shutil.which("nvidia-smi"):
            self.source = "nvidia-smi"
            self._thread = threading.Thread(target=self._run_nvidia, daemon=True)
            self._thread.start()
        elif self.device.type == "mps":
            self.source = "not_available_mps"
        elif self.device.type == "cpu":
            self.source = "not_available_cpu"

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def stats(self, seconds: float) -> tuple[float | None, float | None, float | None, str]:
        if not self.values:
            return None, None, None, self.source
        avg_power = sum(self.values) / len(self.values)
        return avg_power, max(self.values), avg_power * max(0.0, seconds), self.source

    def _run_nvidia(self) -> None:
        while not self._stop.is_set():
            value = self._read_nvidia_power()
            if value is not None:
                self.values.append(value)
            self._stop.wait(self.interval_seconds)

    @staticmethod
    def _read_nvidia_power() -> float | None:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=power.draw",
                    "--format=csv,noheader,nounits",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=3,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        if result.returncode != 0:
            return None
        line = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
        try:
            return float(line.strip())
        except ValueError:
            return None


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
    epochs: int,
    batches_per_epoch: int,
    warmup_batches: int,
    lr: float,
    momentum: float,
    weight_decay: float,
    max_samples: int,
    num_workers: int,
    download: bool,
    amp: bool,
    log_every: int,
    power_sample_interval: float,
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
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    measured_batches = (
        max(1, epochs) * max(1, batches_per_epoch)
        if epochs > 0 and batches_per_epoch > 0
        else batches
    )
    effective_batches_per_epoch = (
        max(1, batches_per_epoch)
        if epochs > 0 and batches_per_epoch > 0
        else max(1, batches)
    )

    run_id = (
        f"{time.strftime('%Y%m%d-%H%M%S')}-"
        f"{socket.gethostname()}-{dataset_name}-{model_name}"
    ).replace("/", "-").replace("\\", "-")
    print(
        f"[bench] {dataset_name}/{model_name}: "
        f"device={device}, batch={batch_size}, warmup={warmup_batches}, "
        f"measured={measured_batches}, epochs={epochs}, "
        f"batches/epoch={batches_per_epoch}, lr={lr}"
    )
    total_loss = 0.0
    total_samples = 0
    measured = 0
    started = 0.0
    final_batch_loss = 0.0
    interval_loss = 0.0
    interval_samples = 0
    interval_started = 0.0
    power = PowerSampler(device, power_sample_interval)

    iterator = iter(loader)
    target_steps = warmup_batches + measured_batches
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
            power.start()
            started = time.perf_counter()
            interval_started = started

        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, amp):
            outputs = model(images)
            loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if step >= warmup_batches:
            batch_samples = int(images.shape[0])
            batch_loss = float(loss.detach().cpu())
            total_samples += batch_samples
            total_loss += batch_loss * batch_samples
            interval_samples += batch_samples
            interval_loss += batch_loss * batch_samples
            final_batch_loss = batch_loss
            measured += 1
            current_epoch = ceil(measured / effective_batches_per_epoch)
            current_epoch_batch = ((measured - 1) % effective_batches_per_epoch) + 1
            if log_every > 0 and (measured % log_every == 0 or measured == measured_batches):
                synchronize_device(device)
                now = time.perf_counter()
                interval_seconds = max(now - interval_started, 1e-9)
                total_seconds = max(now - started, 1e-9)
                avg_power, max_power, energy, source = power.stats(total_seconds)
                progress = ProgressRow(
                    run_id=run_id,
                    hostname=socket.gethostname(),
                    device=str(device),
                    world_size=1,
                    model=model_name,
                    dataset=dataset_name,
                    batch_size=batch_size,
                    epoch=current_epoch,
                    epoch_batch=current_epoch_batch,
                    measured_batch=measured,
                    samples=total_samples,
                    interval_seconds=interval_seconds,
                    total_seconds=total_seconds,
                    interval_samples_per_second=interval_samples / interval_seconds,
                    cumulative_samples_per_second=total_samples / total_seconds,
                    interval_loss=interval_loss / max(1, interval_samples),
                    cumulative_loss=total_loss / max(1, total_samples),
                    avg_power_watts=avg_power,
                    max_power_watts=max_power,
                    energy_joules=energy,
                    power_source=source,
                )
                write_progress_row(progress, output_dir)
                print(
                    f"[bench] progress {dataset_name}/{model_name}: "
                    f"batch={measured}/{measured_batches}, "
                    f"epoch={current_epoch}, "
                    f"throughput={progress.cumulative_samples_per_second:.2f} samples/s, "
                    f"loss={progress.cumulative_loss:.4f}"
                )
                interval_started = now
                interval_loss = 0.0
                interval_samples = 0

    synchronize_device(device)
    power.stop()
    seconds = max(time.perf_counter() - started, 1e-9)
    avg_power, max_power, energy, source = power.stats(seconds)
    throughput = total_samples / seconds
    result = BenchmarkResult(
        run_id=run_id,
        hostname=socket.gethostname(),
        os=platform.system() or "Unknown",
        machine=platform.machine() or "unknown",
        device=str(device),
        world_size=1,
        participant_count=1,
        model=model_name,
        dataset=dataset_name,
        classes=classes,
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
        epochs=epochs,
        batches_per_epoch=batches_per_epoch,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        dataset_samples=dataset_samples,
        measured_batches=measured,
        warmup_batches=warmup_batches,
        samples=total_samples,
        seconds=seconds,
        samples_per_second=throughput,
        throughput=throughput,
        seconds_per_batch=seconds / max(1, measured),
        speedup=1.0,
        efficiency=1.0,
        worker_score=throughput,
        worker_scores=json.dumps({socket.gethostname(): throughput}, sort_keys=True),
        estimated_epoch_seconds=dataset_samples / max(total_samples / seconds, 1e-9),
        estimated_100_epoch_hours=(dataset_samples / max(total_samples / seconds, 1e-9)) * 100 / 3600,
        loss=total_loss / max(1, total_samples),
        final_batch_loss=final_batch_loss,
        avg_power_watts=avg_power,
        max_power_watts=max_power,
        energy_joules=energy,
        power_source=source,
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


def write_progress_row(progress: ProgressRow, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "single_device_progress.csv"
    row = asdict(progress)
    needs_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if needs_header:
            writer.writeheader()
        writer.writerow(row)


def expand_choice(value: str, choices: Iterable[str]) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if value == "all":
        return list(choices)
    return [item.strip() for item in value.split(",") if item.strip()]


def load_config(path: str) -> dict[str, Any]:
    if not path:
        return {}
    config_path = Path(path).resolve()
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid JSON config {config_path}: {exc}") from exc
    if not isinstance(config, dict):
        raise SystemExit(f"config must be a JSON object: {config_path}")
    return config


def merged_config(args: argparse.Namespace) -> dict[str, Any]:
    config = dict(DEFAULT_CONFIG)
    config.update(load_config(args.config))
    for key, value in vars(args).items():
        if key == "config":
            continue
        if value is not None:
            config[key] = value
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single-device training benchmark for model/dataset combinations."
    )
    parser.add_argument("--config", default="", help="Path to a JSON benchmark config.")
    parser.add_argument("--model", default=None, help="all, or comma-list: resnet50,resnet101,vit_b_16")
    parser.add_argument("--dataset", default=None, help="all, or comma-list: cifar100,tiny-imagenet-200")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None, help="auto, cpu, cuda, cuda:0, or mps")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--batches", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batches-per-epoch", type=int, default=None)
    parser.add_argument("--warmup-batches", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=None, help="Write progress every N measured batches; 0 disables.")
    parser.add_argument("--power-sample-interval", type=float, default=None)
    parser.add_argument("--max-samples", type=int, default=None, help="Limit dataset samples; 0 uses all.")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--momentum", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=None, help="Download missing datasets.")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None, help="Use autocast mixed precision on CUDA/MPS.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = merged_config(args)
    models_to_run = expand_choice(config["model"], MODEL_CHOICES)
    datasets_to_run = expand_choice(config["dataset"], DATASET_CHOICES)

    unknown_models = sorted(set(models_to_run) - set(MODEL_CHOICES))
    unknown_datasets = sorted(set(datasets_to_run) - set(DATASET_CHOICES))
    if unknown_models:
        raise SystemExit(f"unknown model(s): {', '.join(unknown_models)}")
    if unknown_datasets:
        raise SystemExit(f"unknown dataset(s): {', '.join(unknown_datasets)}")

    device = choose_device(str(config["device"]))
    data_dir = Path(str(config["data_dir"])).resolve()
    output_dir = Path(str(config["output_dir"])).resolve()
    print(f"[bench] host={socket.gethostname()} os={platform.system()} machine={platform.machine()}")
    print(f"[bench] torch={torch.__version__} device={device}")
    print("[bench] config:", json.dumps(config, sort_keys=True))

    for dataset_name in datasets_to_run:
        for model_name in models_to_run:
            run_one(
                model_name=model_name,
                dataset_name=dataset_name,
                data_dir=data_dir,
                output_dir=output_dir,
                device=device,
                batch_size=max(1, int(config["batch_size"])),
                batches=max(1, int(config["batches"])),
                epochs=max(0, int(config["epochs"])),
                batches_per_epoch=max(0, int(config["batches_per_epoch"])),
                warmup_batches=max(0, int(config["warmup_batches"])),
                lr=float(config["lr"]),
                momentum=float(config["momentum"]),
                weight_decay=float(config["weight_decay"]),
                max_samples=max(0, int(config["max_samples"])),
                num_workers=max(0, int(config["num_workers"])),
                download=bool(config["download"]),
                amp=bool(config["amp"]),
                log_every=max(0, int(config["log_every"])),
                power_sample_interval=max(0.25, float(config["power_sample_interval"])),
            )


if __name__ == "__main__":
    main()
