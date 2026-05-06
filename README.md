# Single-Device ML Benchmark

Benchmark individual laptops before distributed training.

This repo measures local training throughput for:

- `resnet50`
- `resnet101`
- `vit_b_16`

on:

- `cifar100`
- `tiny-imagenet-200`

Results are saved to:

```text
runs/single_device/single_device_results.csv
runs/single_device/single_device_progress.csv
runs/single_device/*.json
```

This is a throughput benchmark, not a full convergence run. Each model/dataset
pair runs a few warmup batches and then a fixed number of measured training
batches. Use the same batch count on every device for fair comparison.

You can run either fixed-batch mode or epoch-style mode:

- Fixed-batch mode: set `Batches`; `Epochs=0`.
- Epoch-style mode: set `Epochs` and `BatchesPerEpoch`; total measured batches are `Epochs * BatchesPerEpoch`.

The CSV also estimates:

- `estimated_epoch_seconds`
- `estimated_100_epoch_hours`

These estimates help decide which model/dataset pair is large enough to justify
distributed training before spending hours on full local training.

`single_device_results.csv` is the final per-run summary. `single_device_progress.csv`
contains intermediate rows every `LogEvery` / `LOG_EVERY` measured batches so you
can inspect loss and throughput trends during the run.

## Windows

Setup:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1
```

Smoke test:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_single_device_benchmarks.ps1 -Model "resnet50" -Dataset "cifar100" -BatchSize 4 -Batches 5 -Download
```

Full benchmark:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_single_device_benchmarks.ps1 -Model "resnet50,resnet101,vit_b_16" -Dataset "cifar100,tiny-imagenet-200" -BatchSize 8 -Batches 20 -LogEvery 5 -Download
```

Longer benchmark:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_single_device_benchmarks.ps1 -Model "resnet50,resnet101,vit_b_16" -Dataset "cifar100,tiny-imagenet-200" -BatchSize 8 -Batches 100 -LogEvery 10 -Download
```

Epoch-style benchmark from config:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_single_device_benchmarks.ps1 -Config "configs\all6_epoch_benchmark.json"
```

Focused ResNet-101/CIFAR-100 epoch-style benchmark:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_single_device_benchmarks.ps1 -Config "configs\resnet101_cifar100_epoch_benchmark.json"
```

## macOS

Setup:

```bash
chmod +x setup_macos.sh run_single_device_benchmarks.sh
./setup_macos.sh
```

Smoke test:

```bash
DOWNLOAD=1 MODEL="resnet50" DATASET="cifar100" BATCH_SIZE=4 BATCHES=5 ./run_single_device_benchmarks.sh
```

Full benchmark:

```bash
DOWNLOAD=1 MODEL="resnet50,resnet101,vit_b_16" DATASET="cifar100,tiny-imagenet-200" BATCH_SIZE=8 BATCHES=20 LOG_EVERY=5 ./run_single_device_benchmarks.sh
```

Longer benchmark:

```bash
DOWNLOAD=1 MODEL="resnet50,resnet101,vit_b_16" DATASET="cifar100,tiny-imagenet-200" BATCH_SIZE=8 BATCHES=100 LOG_EVERY=10 ./run_single_device_benchmarks.sh
```

Epoch-style benchmark from config:

```bash
CONFIG=configs/all6_epoch_benchmark.json ./run_single_device_benchmarks.sh
```

Focused ResNet-101/CIFAR-100 epoch-style benchmark:

```bash
CONFIG=configs/resnet101_cifar100_epoch_benchmark.json ./run_single_device_benchmarks.sh
```

## Notes

- If a model runs out of memory, reduce `BatchSize` to `4`, `2`, or `1`.
- Use the same `BatchSize`, `Batches`, and model/dataset list on every device for fair comparison.
- `Batches` controls measured training batches per combination. It is not an epoch count.
- In epoch-style mode, `Epochs * BatchesPerEpoch` controls measured training batches per combination.
- `LogEvery` / `LOG_EVERY` controls intermediate progress logging.
- The all-6 config uses `lr=0.001` to reduce ViT instability risk. The focused ResNet-101 config uses `lr=0.01`.
- Power is best-effort. NVIDIA CUDA devices use `nvidia-smi`; Apple MPS/CPU power is recorded as unavailable unless OS-level tooling is added.
- Tiny ImageNet is downloaded from Stanford's CS231n public dataset URL.
- Send back `runs/single_device/single_device_results.csv` after the run.
