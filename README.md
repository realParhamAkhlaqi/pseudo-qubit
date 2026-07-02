# pseudo-qubit

# Pseudo-Qubit: Classical Simulator of Quantum Superposition

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![C++](https://img.shields.io/badge/C++-17-blue.svg)](https://isocpp.org/)
[![AVX2](https://img.shields.io/badge/AVX2-Optimized-green.svg)](https://en.wikipedia.org/wiki/AVX2)

---

## 📖 The Logic Behind This Project

**This is not just a code — it's a conceptual architecture.**

This repository demonstrates how a **qubit's superposition** can be simulated on classical hardware using:
- **Complex numbers** (for probability amplitudes)
- **SIMD instructions (AVX2)** (for parallel processing)

We explore three fundamental levels of computation:

---

### 1️⃣ Classical Bit
- A simple **switch**: either `0` or `1`
- Like a light switch — on or off
- To check 8 paths, it must try them **one by one** (8 time units)

---

### 2️⃣ Real Qubit (Quantum)
- A **rotating vector** in 2D complex space
- Exists in **superposition**: both `0` and `1` simultaneously with different weights
- Mathematical representation:
  
$$
|\psi\rangle = \alpha|0\rangle + \beta|1\rangle
$$
  
$$
|\psi\rangle = \alpha|0\rangle + \beta|1\rangle
\quad \text{where} \quad
\alpha, \beta \in \mathbb{C}
\quad \text{and} \quad
|\alpha|^2 + |\beta|^2 = 1
$$

- Lives on the **Bloch Sphere** — can point anywhere, not just north/south
- Can explore **all paths at once** (1 time unit for 8 paths)

---

### 3️⃣ Classical Simulator (This Project)
- A **numerical copy** of the qubit's state vector
- Uses **two floating-point numbers** per amplitude (real + imaginary)
- Memory per qubit: **16 bytes** (vs 1 bit in classical computing)

---

## 🧮 The Mathematics

For **N qubits**, we need \(2^N\) complex amplitudes:

$$
|\psi\rangle = \sum_{i=0}^{2^N-1} c_i |i\rangle
$$

where each \(c_i\) is a complex number.

**Examples:**
- 1 qubit → 2 amplitudes
- 10 qubits → 1024 amplitudes
- 22 qubits → 4,194,304 amplitudes (~64 MB memory)
- 28 qubits → 268,435,456 amplitudes (~4 GB memory)

---

## 🔬 How It Works

1. **State vector** is stored as an array of complex numbers
2. **Quantum gates** are applied as matrix multiplications
3. **SIMD (AVX2)** processes 4 complex numbers simultaneously
4. **Benchmarking** compares scalar vs vectorized performance

---

## 📊 Comparison Table

| Feature | Classical Bit | Real Qubit | This Simulator |
|---------|---------------|------------|----------------|
| **State Space** | 2 values (0/1) | 2 amplitudes (complex) | 2 amplitudes (floats) |
| **Representation** | Single number | Vector in Hilbert space | Array of complex numbers |
| **Parallelism** | None (sequential) | Yes (superposition) | Yes (all amplitudes in memory) |
| **Physical Nature** | Voltage | Electron spin/photon polarization | Numbers in RAM |
| **Limitation** | Must iterate | Decoherence/noise | Memory (30+ qubits = RAM limit) |

---

## ⚡ Performance (Benchmark Results)

**Hardware:** Intel i5-6200U (laptop processor)

| Qubits | Size | Scalar (ns) | AVX2 (ns) | Speedup |
|--------|------|-------------|-----------|---------|
| 4 | 16 | 1,063 | 578 | **1.84x** |
| 10 | 1024 | 10,900 | 11,400 | **2.64x** |
| 18 | 262k | 13,470,000 | 13,645,000 | 0.99x |
| 22 | 4.19M | 332,969,374 | 303,555,565 | **1.10x** |
| 24 | 16.7M | 1,121,987,664 | 1,247,747,216 | 0.90x |

> **Peak AVX2 speedup:** **2.64x** observed in best-case scenarios

---

## 🚀 Getting Started

### Prerequisites
- C++17 compiler (GCC 9+ / Clang 10+)
- CPU with AVX2 support
- Linux/macOS/Windows (with WSL)

### Build & Run
```bash
git clone https://github.com/realParhamAkhlaqi/pseudo-qubit.git
cd pseudo-qubit
g++ -O3 -march=native -mavx2 main.cpp -o pseudo_qubit
./pseudo_qubit
```

### Changing Parameters
Edit `main.cpp`:
```cpp
const int n_qubits = 10;        // Number of qubits (1-28)
const int iterations = 500;     // Benchmark iterations
const double test_angle = 0.7853981633974483; // 45 degrees
```

---

## 📂 Repository Structure
```
pseudo-qubit/
├── main.cpp        # Version 1: Original AVX2 implementation (step==1 scalar bug)
├── main_v2.cpp        # Version 2: Fixed AVX2 for ALL layers (+17% avg speedup)
├── README.md          # This file
├── LICENSE.txt        # GPL-3.0 license
├── pq-test.txt     # Benchmark results for v1 (scalar vs AVX2)
└── pq-test-v2.txt     # Benchmark results for v2 (fixed AVX2 Max:3.25x Average:~2.10x)
```

---

## 📜 License
This project is released under the **GNU General Public License v3.0**.
See the [LICENSE](LICENSE) file for full details.

**Why GPL3?**
- Protects freedom to share and modify
- Prevents proprietary forks
- Ensures logic remains open for all

---

## 🤝 Contributing
Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with clear description

---

## 📧 Contact
[Parham Akhlaqi] — [parhamakhlaqi59@gmail.com]

---

**Built with ❤️ for the open-source community.**

*"The logic of a qubit, simulated classically."*
