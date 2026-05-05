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
runs/single_device/*.json
```

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
powershell -ExecutionPolicy Bypass -File .\run_single_device_benchmarks.ps1 -Model "resnet50,resnet101,vit_b_16" -Dataset "cifar100,tiny-imagenet-200" -BatchSize 8 -Batches 20 -Download
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
DOWNLOAD=1 MODEL="resnet50,resnet101,vit_b_16" DATASET="cifar100,tiny-imagenet-200" BATCH_SIZE=8 BATCHES=20 ./run_single_device_benchmarks.sh
```

## Notes

- If a model runs out of memory, reduce `BatchSize` to `4`, `2`, or `1`.
- Use the same `BatchSize`, `Batches`, and model/dataset list on every device for fair comparison.
- Tiny ImageNet is downloaded from Stanford's CS231n public dataset URL.
- Send back `runs/single_device/single_device_results.csv` after the run.

