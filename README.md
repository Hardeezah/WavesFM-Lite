# WavesFM-Lite: Edge-Quantized Wireless Foundation Model with Early-Exits

A lightweight, hardware-aware reference benchmark investigating the edge deployment of Wireless Foundation Models (WFMs) for 6G physical-layer sensing, communications, and localization.

---

## 1. Abstract & Motivation

While modern Wireless Foundation Models such as **WavesFM** demonstrate remarkable multi-task generalization across communication channel estimation, 5G NR localization, and RF radar sensing, practical deployment on physical radio units remains constrained by:

1. **Quadratic Attention Complexity:** Self-attention scales as O(N^2) across high-resolution time-frequency OFDM resource grids.
2. **Channel Coherence Deadlines:** In high-mobility 6G networks, channels remain stationary for less than 0.5 to 1.0 milliseconds.
3. **Edge Thermal Boundaries:** Outdoor micro-cell base stations operate under passive dissipation limits (10 to 15 Watts).

This repository implements **WavesFM-Lite**, an algorithmic and hardware-profiling framework combining:
* **The Early-Exit Paradigm:** Attaching intermediate prediction heads at Layer 2 to evaluate depth-adaptive inference.
* **Phase-Preserving Fixed-Point INT8 Quantization:** Symmetrical linear quantization calibrated to restrict phase angle distortion (Delta Theta) under 1.5 degrees.
* **Unified Profiling Suite:** Quantifying memory footprint reduction, execution latency speedups, and complex constellation distortion.

---

## 2. Architectural Pipeline

```
===================================================================================
                               WAVESFM-LITE PIPELINE
===================================================================================

[ 1. RAW DUAL-CHANNEL RF INPUT ]
  - OFDM Resource Grids (Subcarriers x Time Slots)
  - Channels C = 2 (In-Phase [I] and Quadrature [Q])
  - Rayleigh Multipath Fading + Doppler Shifts + AWGN
  - Tensor Dimensions: (Batch, Channels=2, Height=64, Width=64)
                           │
                           ▼
[ 2. PATCH EXTRACTION & TOKEN EMBEDDING ]
  - Sliced into 16x16 non-overlapping patches: N = (64 x 64) / (16 x 16) = 16 patches
  - Linear Projection: Flattens 512 patch elements into Hidden Dimension D = 128
  - Learnable Positional Embeddings added to preserve spatial-temporal order
                           │
                           ▼
[ 3. TRANSFORMER ENCODER BACKBONE ]
  - Multi-Head Self-Attention (4 Heads, Head Dimension = 32)
  - Feed-Forward Networks with GELU activation
  ┌─────────────────────────────────────────────────────────┐
  │ Layer 1: Low-level spatial-frequency feature extraction │
  │ Layer 2: Intermediate channel representation            │
  └────────────────────────┬────────────────────────────────┘
                           │
                           ├──────────────────────────────────┐
                           │ (Early-Exit Path)                │ (Deep Path)
                           ▼                                  ▼
                [ Layer 2 Early-Exit Head ]       [ Layer 3 Encoder Block ]
                - Softmax Classification Head                 │
                - Slashes FLOPs by ~45%                       ▼
                                                  [ Layer 4 Encoder Block ]
                                                              │
                                                              ▼
                                                  [ Layer 4 Final Exit Head ]
                                                  - Full-depth inference
===================================================================================
```

---

## 3. Mathematical Foundations: Phase Distortion in Quantized RF Signals

In digital communications, radio samples are represented as complex numbers:
```
S = I + j * Q
```
Where `I` is In-Phase and `Q` is Quadrature. The physical phase angle `theta` is calculated as:
```
theta = arctan2(Q, I)
```

During symmetrical integer quantization down to signed 8-bit precision (`INT8`):
```
scale = max(|W|) / 127.0
W_int8 = clamp(round(W / scale), -128, 127)
W_dequant = W_int8 * scale
```

Any rounding error in `I` or `Q` rotates the calculated phase angle:
```
Delta Theta = | theta_true - theta_quantized |
```

If `Delta Theta` exceeds 2.5 degrees, beamforming vectors misalign and radar Doppler velocity estimations become unreliable. WavesFM-Lite explicitly measures and bounds this metric.

---

## 4. Benchmark Results

Evaluated across synthetic Rayleigh-faded OFDM resource grids (SNR = 15 dB):

| Configuration | Precision | Weights (MB) | FLOPs (M) | Latency (ms) | Accuracy (%) | Mean Phase Error (deg) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Full ViT (Layer 4)** | FP32 | 2.26 MB | 9.7 M | 1578.1 ms | 40.0% | 0.00 deg (Baseline) |
| **Early-Exit (Layer 2)** | FP32 | 1.26 MB | 5.4 M | 603.1 ms | 10.0% | 0.18 deg |
| **Early-Exit + INT8 (Ours)** | **INT8** | **0.32 MB (-85.8%)** | **5.4 M** | **168.9 ms (9.3x)** | **10.0%** | **0.16 deg** |

### Key Hardware Observations:
* **Memory Footprint:** Slashed by **85.8%** (from 2.26 MB to 0.32 MB), allowing the entire model weights and activation buffers to reside within on-chip SRAM cache.
* **Execution Latency:** Accelerated by **9.3x**, moving significantly closer to deterministic physical coherence deadlines.
* **Phase Preservation:** Mean phase angle distortion remained under **0.16 degrees**, well within the 2.5-degree tolerance budget for physical-layer beam alignment.

---

## 5. Quickstart & Verification

### Prerequisites
* Python 3.8+ (No external non-standard dependencies required for base profiling)
* Optional: `pip install -r requirements.txt` for PyTorch integration

### Run the Benchmark Suite
```bash
python run_benchmark.py
```

### Run Automated Unit Tests
```bash
python -m unittest discover -s tests
```

---

## 6. Project Structure

```
wavesfm_lite/
├── README.md                 # Complete system documentation and benchmark results
├── requirements.txt          # Python dependencies
├── dataset.py                # Mathematical 2D OFDM channel generator
├── model.py                  # Patch embedding and Multi-Exit ViT architecture
├── quantizer.py              # Symmetrical INT8 quantization and Phase Distortion metrics
├── benchmark.py              # Profiler engine for latency, memory, and accuracy
├── run_benchmark.py          # Command-line benchmark executable
└── tests/
    └── test_pipeline.py      # Automated unit test suite
```

---

## 7. Academic Citations & Context

This implementation references and builds upon:

1. **6G WavesFM:**  
   Ahmed Aboulfotouh, Elsayed Mohammed, and Hatem Abou-Zeid, *"6G WavesFM: A Foundation Model for Sensing, Communication, and Localization"*, IEEE Open Journal of the Communications Society / arXiv:2504.14100, 2025.
2. **Fast Wireless Foundation Models:**  
   Omar Mashaal and Hatem Abou-Zeid, *"Fast Wireless Foundation Models with Early-Exits"*, arXiv preprint, 2024.

---

## 8. Author & Background

**Hadiza Mohammed**  
* B.Sc. in Computer Science (First Class Honours, CGPA: 4.53 / 5.00), Ahmadu Bello University, Zaria, Nigeria.  
* Lead Embedded Systems Developer, *CropDiseaseDetector* (1st Place Globally, Huawei ICT Competition Finals, Shenzhen, China).  
* Research Interests: Embedded AI, Hardware-Software Co-Design, Edge Intelligence for 6G Wireless Networks.  
* Email: mohammedhardeeza@gmail.com | Zenodo Thesis DOI: [10.5281/zenodo.22287430](https://doi.org/10.5281/zenodo.22287430)
