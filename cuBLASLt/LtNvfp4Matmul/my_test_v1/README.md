# cuBLASLt NVFP4 dense GEMM benchmark (reproduction of nvfp4_dense_gemm.csv)

Reproduces `<repo>/nvfp4_dense_gemm.csv` (cuBLASLt NVFP4 dense GEMM, 49 shapes).

## Method
- **Timing:** one CUDA graph capturing **10 `cublasLtMatmul` back-to-back**, replayed once,
  timed with one cudaEvent pair → `us = elapsed / 10`. Warmup = 5 matmuls + 2 graph replays.
  (Batching 10 matmuls inside the graph avoids per-launch CPU gaps that otherwise inflate tiny
  shapes; the window stays short so the clock holds 2062.)
- **cuBLASLt setup (== LtNvfp4Matmul sample, but bf16 output):** A/B = `CUDA_R_4F_E2M1` with VEC16
  UE4M3 block scales, transa=T/transb=N, heuristic algo, 32MB workspace; output **D = bf16**
  (`CUDA_R_16BF`), `alpha=1, beta=0` (the reference's gbps counts a bf16 output, and the raw
  sample's fp4-output+requant epilogue is slower). Operand/scale alloc reuses `TestBench`
  (Common/helpers.h).
- **gbps** = ((m·k + n·k)·0.5 + m·n·2) / s / 1e9   (A,B packed fp4 + bf16 output).
- **sm_mhz / power_w:** NVML over a short (~40ms) sustained-replay window (sm=2062; power
  under-reads since NVML `nvmlDeviceGetPowerUsage` is a ~1s rolling average).

## Match vs reference
TFLOPS: **median ratio 1.007, 45/49 within 10%** (per-shape ±10% = cuBLASLt heuristic / run-to-run
variance); 16384³ to ~0.7%; **sm_mhz = 2062** everywhere.

## Files
- `bench_nvfp4.cu` — the benchmark; `shapes.txt` — the 49 shapes.
- `bench_nvfp4_capN.cu` — variant: capture N matmuls/graph (argv[3], default 100), replay once.
- output: `../result/nvfp4_dense_gemm_repro.csv`, `../result/compare_vs_reference.csv`.

## Build / run
    export PATH=/usr/local/cuda/bin:$PATH
    nvcc -O3 -std=c++17 -arch=sm_100a bench_nvfp4.cu -I../../Common -lcublasLt -lnvidia-ml -o bench_nvfp4
    ./bench_nvfp4 shapes.txt ../result/nvfp4_dense_gemm_repro.csv
