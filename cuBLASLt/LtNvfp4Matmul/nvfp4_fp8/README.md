# NVFP4 × NVFP4 → FP8 — not supported by cuBLASLt (dropped)

This experiment (NVFP4×NVFP4 dense GEMM with an **FP8** output, graph-sustained steady-state, the
sibling of [`../nvfp4_nvfp4`](../nvfp4_nvfp4)) **cannot be built on cuBLASLt**: cuBLASLt has no
NVFP4-inputs → FP8-output path. It was dropped after verification (see below).

## What was tried
A,B = NVFP4 (`CUDA_R_4F_E2M1`, VEC16 UE4M3 block scales), transa=T/transb=N, compute = 32F, and
`cublasLtMatmulAlgoGetHeuristic` requested for **D = FP8 e4m3** across the full cross product of:

- **C type:** bf16, fp32, fp16, fp8_e4m3
- **D scale mode** (`D_SCALE_*`): none, scalar 32F, VEC128 32F
- **D out scale mode** (`D_OUT_SCALE_*`): none, scalar 32F, VEC16 UE4M3, VEC32 UE8M0, VEC128 32F,
  BLK128x128 32F, OUTER_VEC 32F
- **AMAX_D pointer:** set / unset

**Every** one of these (168 configurations, plus earlier e5m2 variants) returns
`CUBLAS_STATUS_NOT_SUPPORTED` (15) at the heuristic stage. A control in the same harness —
NVFP4 → bf16 — succeeds, as does NVFP4 → NVFP4 (`../nvfp4_nvfp4`) and NVFP4 → fp32. So this is a
hard cuBLASLt limitation for the FP8 *output* type with FP4 inputs, not a misconfiguration.

> The cuBLAS 12.9 blog line "compute scaling factors for the D tensor when the output is FP4 or FP8"
> refers to FP8 **inputs** (fp8×fp8 with a computed D scale), not NVFP4 inputs → FP8 output.

## Reproduce / re-check on a future cuBLAS
```
export PATH=/usr/local/cuda/bin:$PATH
nvcc -O3 -std=c++17 -arch=sm_100a check_unsupported.cu -I../../Common -lcublasLt -o check_unsupported
./check_unsupported
```
Observed (GB200 / sm_100, cuBLASLt 13.4.0.1):
```
cuBLASLt version 130400
CONTROL  nvfp4 -> bf16 : got=1 (expect >=1)
SWEEP    nvfp4 -> fp8_e4m3 : 0/168 configurations supported
=> NVFP4 x NVFP4 -> FP8 is NOT supported by this cuBLASLt.
```
If a future cuBLAS reports a supported fp8 config, revive the benchmark: it is identical to
`../nvfp4_nvfp4/bench_nvfp4_nvfp4.cu` but with `Dd = CUDA_R_8F_E4M3` and the working D-scale recipe
(and a `fillData()` specialization for the `…, __nv_fp8_e4m3, …` TestBench type, since helpers.h
only ships nvfp4→{nvfp4,bf16} specializations).

## Supported NVFP4-input outputs (for reference)
`CUDA_R_4F_E2M1` (see `../nvfp4_nvfp4`), `CUDA_R_16F`, `CUDA_R_16BF`, `CUDA_R_32F`. Not FP8.
