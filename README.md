# WavesFM-Lite

A standalone pure-Python research prototype investigating Layer-2 early exiting and symmetric INT8 weight, input, and activation quantization on a Vision Transformer (ViT) over 2D OFDM I/Q resource grids.

This repository isolates architectural footprint, arithmetic operations (FLOPs), I/Q phase distortion, and FP32-vs-INT8 classification agreement on synthetic wireless channels using untrained deterministic weights (`seed=101, 202, 404, 505`).

---

## Architecture & Module Design

```
[1. dataset.py] Synthetic 2 x 64 x 64 OFDM Grid (In-Phase I, Quadrature Q)
  - 4 synthetic classes ("BPSK", "QPSK", "8PSK", "16QAM" via Doppler shifts)
  - Rayleigh fading + multipath delay (1..8) + AWGN (15 dB SNR)
       |
       v
[2. model.py: PatchEmbedding] ([P1] 6G WavesFM ViT-style tokenization)
  - 16 non-overlapping 16x16 patches (512 elements each -> D = 128) + positional embeddings
       |
       v
[3. model.py: Transformer Encoder] (4 layers, 4 heads, FFN dim = 256, GELU)
  |-- Layer 1 -> Layer 2 --+--> [Early-Exit Head]  (5.4 M FLOPs, -44.3%)
                           |
                           +--> Layer 3 -> Layer 4 -> [Final Exit Head] (9.7 M FLOPs)
       |
       v
[4. quantizer.py + benchmark.py] Symmetric INT8 Quantization & Profiling
  - Symmetric scale: S = max(|W|) / 127.0, zero-point = 0, clamp [-128, 127]
  - Quantizes weights, biases, I/Q inputs, and inter-layer activations
  - Measures circular phase error: mean |atan2(Q, I) - atan2(Q_hat, I_hat)| (wrapped at pi)
```

---

## Verified Benchmark Results (`python3 run_benchmark.py`)

Measured across `10` synthetic `2 x 64 x 64` grids at `15 dB` SNR (untrained random weights):

| Configuration | Precision | Active Params | Weights (MB) | FLOPs (M) | Python CPU Latency | Raw Acc (Untrained) | Mean Phase Error |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Full ViT (Layer 4) | FP32 | 592,516 | 2.26 MB | 9.7 M | 950.68 ms | 40.0% | 0.0 deg |
| Early-Exit (Layer 2) | FP32 | 330,372 | 1.26 MB | 5.4 M | 578.98 ms | 10.0% | 0.0 deg |
| Early-Exit + INT8 | INT8 | 330,372 | 0.32 MB | 5.4 M | 649.4 ms | 10.0% | 0.16 deg |

Key verified metrics:
* **Weight memory cut (`-85.8%`):** `592,516` FP32 parameters (`2.26 MB`, 4 bytes/param) down to `330,372` INT8 parameters (`0.32 MB`, 1 byte/param).
* **Arithmetic cut (`-44.3%`):** `9.7 M` FLOPs across 4 layers down to `5.4 M` FLOPs across 2 layers.
* **Measured Python CPU latency:** Skipping Layers 3–4 speeds up FP32 inference by **`1.6x`** (`950.68 ms -> 578.98 ms`). Pure Python has no INT8 SIMD or NPU kernels, so simulated pure-Python quantize/dequantize loops add Python overhead (`649.4 ms`) rather than hardware speedup.
* **Quantization fidelity:** **`100.0%`** Top-1 prediction agreement (`10 / 10` grids) between Layer-2 FP32 and INT8 weights + inputs + activations, with **`0.16 deg`** mean circular I/Q phase distortion across all 10 grids.

---

## Quickstart & Tests

Requires Python 3.8+ (uses the Python standard library only; packages in `requirements.txt` are not imported at runtime).

```bash
python3 run_benchmark.py
python3 -m unittest discover -s tests
```

All `5` unit tests verify dataset shapes, Layer-2/Layer-4 probability outputs, symmetric INT8 integer bounds (`[-128, 127]`), `ModelProfiler` parameter/memory reductions, and `100.0%` FP32-vs-INT8 Top-1 agreement with bounded phase error.

---

## Project Structure

```
wavesfm_lite/
├── __init__.py          Package exports
├── dataset.py           Synthetic 2D OFDM Rayleigh-fading I/Q grid generator
├── model.py             Patch embedding and 4-layer dual-exit ViT backbone
├── quantizer.py         Symmetric INT8 quantizer, circular phase error, and FLOP/memory profiler
├── benchmark.py         Benchmark runner comparing FP32 full, FP32 early-exit, and INT8 early-exit
├── run_benchmark.py     CLI entry point for the benchmark table
├── requirements.txt     Optional environment file (core benchmark uses Python stdlib only)
└── tests/
    └── test_pipeline.py 5 unit tests covering shapes, quantization bounds, phase error, and agreement
```

---

## Research References

1. **6G WavesFM [P1]:** Ahmed Aboulfotouh, Elsayed Mohammed, and Hatem Abou-Zeid, *"6G WavesFM: A Foundation Model for Sensing, Communication, and Localization,"* IEEE Open Journal of the Communications Society, vol. 6, pp. 6792–6807, 2025.
2. **Early-Exits [P2]:** Omar Mashaal and Hatem Abou-Zeid, *"Fast Wireless Foundation Models with Early-Exits,"* arXiv:2606.29640, June 2026.
3. **RRM-GPT [P3]:** Ahmed Aboulfotouh, Akram Bin Sediq, Koosha Pourtahmasi Roshandeh, Omar Mashaal, Ahmad M. Nagib, Jale Sadreddini, and Hatem Abou-Zeid, *"RRM-GPT: A Framework and Vision for Radio Resource Management Foundation Models,"* arXiv:2610.04296, October 2026.

---

## Author

**Hadiza Mohammed**  
B.Sc. Computer Science (First Class Honours, 4.53 / 5.00 CGPA), Ahmadu Bello University  
Global 1st Place, Huawei ICT Competition Finals (Shenzhen, 2025): *CropDiseaseDetector* ([cdd_electron](https://github.com/Hardeezah/cdd_electron))  
Email: `mohammedhardeeza@gmail.com` | GitHub: [github.com/Hardeezah](https://github.com/Hardeezah)
