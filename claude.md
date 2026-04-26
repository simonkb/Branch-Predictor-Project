# Branch Prediction Simplification for AI Workloads

## Project Overview

This is a computer architecture research project for Khalifa University (COSC course). The central research question: **Can a lightweight, loop-aware branch predictor (LABP) achieve competitive accuracy on AI workloads relative to conventional predictors (BiModal, Gshare, TAGE), while reducing complexity and storage?**

The project uses **gem5** (RISC-V, SE mode, O3CPU) to simulate and compare branch predictors on AI-style microbenchmarks.

### Current Status

Preliminary results on `tinymlp` show LABP **underperforms** even the simple baselines (BiModeBP, GshareBP). TAGE dominates. The next phase requires refining LABP's confidence gating/arbitration and adding new benchmarks that isolate counted-loop behavior from data-dependent branching.

## Repository Structure

```
├── gem5/                          # git submodule → github.com/gem5/gem5.git (DO NOT MODIFY unless adding LABP source)
├── configs/
│   └── se_riscv_bp.py             # Project-owned gem5 SE runner with extended --bp-type support
├── benchmarks/
│   ├── hello/                     # Trivial hello-world sanity check (RISC-V static binary)
│   └── tinymlp/
│       ├── tinymlp.cpp            # MLP inference microbenchmark with sparsity-driven data-dependent branches
│       └── Makefile               # Cross-compile: riscv64-linux-gnu-g++ -O3 -static -march=rv64gc
├── scripts/
│   └── run_benchmarks.sh          # Runs gem5 for all predictor×workload combinations
├── analysis/
│   └── branch_predictor_analysis.ipynb  # Jupyter notebook: parses stats.txt + config.ini → comparison DataFrame + plots
├── results/
│   ├── tinymlp_predictor_comparison.csv # Summary CSV of all predictor metrics
│   └── tinymlp/{BiModeBP,GshareBP,LAP,TAGE}/
│       ├── stats.txt              # gem5 simulation statistics
│       ├── config.ini             # gem5 configuration dump
│       └── config.json            # gem5 configuration (JSON format)
├── documentation.txt              # Setup instructions (deps, gem5 build, running experiments)
└── requirements.txt               # Python deps for analysis (pandas, matplotlib, seaborn, jupyter)
```

## Key Technical Details

### gem5 Configuration

- **ISA:** RISC-V (rv64gc)
- **CPU:** O3CPU (out-of-order) with caches (L1 + L2)
- **Mode:** SE (Syscall Emulation), not full-system
- **Instruction limit:** 10M committed instructions per run
- **gem5 binary path:** `./gem5/build/RISCV/gem5.opt` (built via `scons build/RISCV/gem5.opt -j$(nproc)`)

### Branch Predictors Under Study

| Name | gem5 Class | Type | Role |
|------|-----------|------|------|
| BiModeBP | `BiModeBP` | ConditionalPredictor | Low-complexity baseline |
| GshareBP | `GshareBP` | BPredUnit (wraps LocalBP as conditional) | Correlation-aware baseline |
| TAGE | `TAGE` | ConditionalPredictor | High-complexity reference (performance ceiling) |
| LABP/LAP | `LAPBP` | ConditionalPredictor (bimodal base + LoopPredictor child) | Our proposed simplified predictor |

### LABP Architecture (gem5 class: `LAPBP`)

- **Base predictor:** 2048-entry bimodal table, 2-bit saturating counters
- **Loop predictor:** gem5's `LoopPredictor` — 256 sets × 4 ways, 14-bit tags, 14-bit iteration counter, 2-bit confidence, 8-bit age
- **Override policy:** Loop prediction overrides base only when valid and confident
- **Config flag:** `--bp-type=LAP` (aliased to `LAPBP` in `se_riscv_bp.py`)

### Running Experiments

```bash
# Single run
./gem5/build/RISCV/gem5.opt --outdir=results/tinymlp/BiModeBP \
  configs/se_riscv_bp.py \
  --cmd=./benchmarks/tinymlp/tinymlp_rv_static \
  --cpu-type=O3CPU \
  --bp-type=BiModeBP \
  --caches --l2cache \
  -I 10000000

# All predictors at once
bash scripts/run_benchmarks.sh
```

### Building Benchmarks

```bash
cd benchmarks/tinymlp
make                    # requires riscv64-linux-gnu-g++
file tinymlp_rv_static  # should show RISC-V ELF, statically linked
```

### Key Metrics (from stats.txt)

- `system.cpu.ipc` — Instructions Per Cycle
- `system.cpu.cpi` — Cycles Per Instruction
- `system.cpu.branchPred.lookups_0::total` — total branch predictor lookups
- `system.cpu.branchPred.mispredicted_0::total` — total mispredictions
- Misprediction rate = mispredictions / lookups
- Conditional predictor storage is estimated from `config.ini` parameters

### Preliminary Results (tinymlp, 10M instructions)

| Predictor | IPC | Mispred Rate | Cond. BP Storage |
|-----------|-----|-------------|-----------------|
| BiModeBP | 1.919 | 3.86% | 4.00 KB |
| GshareBP | 1.007 | 4.40% | 0.625 KB |
| TAGE | 3.661 | 0.18% | 6.69 KB |
| LABP | 0.569 | 12.16% | 5.50 KB |

## Code Conventions

- Python config scripts follow gem5 conventions (m5.objects, SimObject hierarchy)
- LABP C++ source lives inside `gem5/src/cpu/pred/` (within the submodule) — files include `lap_bp.hh`, `lap_bp.cc`, and the Python SimObject definition
- Benchmark C++ uses `-O3 -static -march=rv64gc -mabi=lp64d -std=c++17`
- Analysis uses pandas + matplotlib/seaborn in Jupyter notebooks
- Results follow directory convention: `results/<workload>/<predictor>/`

## What Needs Work Next

1. **LABP confidence gating:** The loop predictor overrides too aggressively. Need conservative arbitration — only override when loop confidence is high and the branch pattern is truly a stable counted loop.
2. **New benchmarks:** Add workloads that separate clean loop-exit branches from data-dependent inner-loop branches (e.g., a pure matmul kernel, a convolution loop, a workload with minimal sparsity). The `tinymlp` benchmark mixes both, making it hard to isolate the value of loop-awareness.
3. **Storage normalization:** Fair comparison requires normalizing performance against conditional predictor storage (IPC/KB or accuracy/KB).
4. **Analysis expansion:** More detailed plots (misprediction rate vs. storage, IPC breakdown by branch type if gem5 stats support it).
5. **Final report:** Extend the progress report into a complete paper with refined results.

## Dependencies

- **gem5 build:** `build-essential git m4 scons zlib1g-dev libprotobuf-dev protobuf-compiler libgoogle-perftools-dev python3-dev libboost-all-dev pkg-config`
- **RISC-V cross-compiler:** `gcc-riscv64-linux-gnu g++-riscv64-linux-gnu`
- **Python analysis:** see `requirements.txt` (pandas, matplotlib, seaborn, numpy, jupyter)
