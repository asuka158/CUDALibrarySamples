# NVFP4 × NVFP4 → NVFP4 dense GEMM — graph-sustained steady-state benchmark

cuBLASLt NVFP4×NVFP4 dense GEMM (D = A·B, no C/bias) with **NVFP4 (e2m1) output**, over the 64 shapes
in `DeepGEMM/tests/my_test/shape.txt`. Measures the *sustained, throttled* steady-state — the same
methodology as the Python `python/{nvfp4,mxfp4}_v2` suites.

## Method (shared `../nvfp4_sustained_bench.cuh`)
- **Graph-sustained:** per shape, capture G `cublasLtMatmul` calls into a CUDA graph (G auto-sized so
  one segment ≈ 20 ms, ≤512), replay back-to-back for ~3 s with **no host sync** inside the timed
  region (the SM never drains), per-segment `cudaEvent`s as in-stream timestamps + an async NVML
  sampler (1 ms). Report only the **steady-state tail (last 1 s)**: median us / TFLOPS / GB/s /
  sm_mhz / power_w.
- **cuBLASLt recipe (== sample `main.cpp`):** A,B = `CUDA_R_4F_E2M1` with VEC16 UE4M3 block scales,
  D = `CUDA_R_4F_E2M1` with a scalar (32F) D-scale **and** a VEC16 UE4M3 block out-scale that the
  kernel computes; C = bf16 (unused, beta=0); compute = 32F; transa=T/transb=N; 32 MB workspace.
- **Data (close to torch.randn, NOT `float(i%5)`):** A,B are overwritten with N(0,1) samples
  quantized to NVFP4 on-device (curand `curand_normal2` × scale → hardware float2→fp4x2 converter).
  Low-entropy `i%5` data otherwise inflates the sustained clock under the power cap; randn-like data
  gives realistic switching activity → realistic power/clock. Scale via `NVFP4_RANDN_SCALE` (default 2.5).
- **gbps** = (m·k·0.5 + n·k·0.5 + m·n·0.5) / s / 1e9   (A,B,D all packed fp4; scales negligible).

## Build / run
```
export PATH=/usr/local/cuda/bin:$PATH
nvcc -O3 -std=c++17 -arch=sm_100a bench_nvfp4_nvfp4.cu -I../../Common -I.. \
     -lcublasLt -lnvidia-ml -lcurand -o bench_nvfp4_nvfp4
./bench_nvfp4_nvfp4 [shapes.txt] [out.csv]
# defaults: ../../../../DeepGEMM/tests/my_test/shape.txt  ->  result/nvfp4_nvfp4.csv
```
Run alone (no other GPU work) — concurrent kernels skew the clock/timing.

## Output
`result/nvfp4_nvfp4.csv`, columns `m,n,k,us,tflops,gbps,sm_mhz,power_w,backend` (same schema as the
Python v2 suites). 16384³ ≈ 6250 TFLOPS @ ~1310 MHz; 32768³ ≈ 6070 @ ~1240 MHz (power at the ~1180 W cap).

## Sibling FP8 experiment
NVFP4×NVFP4 → FP8 was dropped — cuBLASLt has no such path. See [`../nvfp4_fp8/README.md`](../nvfp4_fp8).
