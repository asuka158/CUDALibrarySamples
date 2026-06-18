# Why the C++ cuBLASLt bench (~7250 TFLOPS) > Python torch._scaled_mm_v2 (~5500–6400), 16384³ NVFP4→fp32

Both call the **same** cuBLASLt NVFP4 dense GEMM. Investigated the gap end-to-end.

## 1. Output dtype
Both compared at **D = fp32** (C++ `bench_smmhz.cu` and `python/nvfp4/bench_nvfp4_fp32.py`). bf16 is
not the issue; we only compare fp32 here.

## 2/3/4. Same kernel, same algo, measurement method is NOT the cause
- **Identical kernel** (nsys + cuBLASLt API log, both processes):
  `nvjet_sm100_oosss_128x256_256x5_2x1_2cta_v_bz_Avec16UE4M3_Bvec16UE4M3_TNT`,
  `algoId=66, tile=128x256, customOption=1, clusterShape=2x1x1`, 32 MB workspace, `transa=OP_T`,
  `aScaleMode=bScaleMode=VEC16_UE4M3`. The cuBLASLt descriptors are byte-identical except C++ ran
  `outOfPlace=0` (C==D) vs torch `outOfPlace=1` (C!=D) — tested, makes no difference.
- **CUPTI vs CUDA-event+graph doesn't matter (Q3):** Python eager (1308 µs), Python CUDA-graph
  (1311 µs), and CUPTI `bench_kineto` all agree. Graph vs eager and the timer are not the cause.
- Ruled out by direct C++ A/B tests: output-buffer reuse (`SMMHZ_NOUT`), workspace size
  (`SMMHZ_WS_MB` 32/1/0), in/out-of-place (`SMMHZ_OOP`) — none moved the number.

## The actual cause: data-dependent power → SM clock (Nsight Compute proof)
ncu on one kernel instance in each process (clock-locked), same grid `(8192,1,1)x(256,1,1)`:

| | duration | sm clock | tensor-pipe util | DRAM | cycles |
|---|---|---|---|---|---|
| C++  | 1.07 ms | 1.82 GHz | 91% | 27% | ~1.95 Gcyc |
| Python | 1.32 ms | 1.43 GHz | 94% | 21% | ~1.89 Gcyc |

**Same cycle count, same ~92% tensor utilization** — identical work. The only difference is the SM
**clock**, and the clock is set by the **1200 W power cap**. FP4 tensor-core power is
**data-dependent** (operand switching activity), so the operand values decide the sustained clock:

| C++ operand data (`SMMHZ_RANDOM`) | cold µs | steady TFLOPS | steady sm MHz | power |
|---|---|---|---|---|
| `float(i%5)` paired (i%5,i%5+1) — TestBench default, values {0..5}, ~10% zeros, low-entropy all-positive | 1.04 | **7180–7260** | 1567–1582 | 1177 W |
| uniform-random fp4 (rough proxy for real data, NOT exact randn) | 1.21 | **5815–5889** | 1245–1260 | 1180 W |
| Python `torch.randn` (reference) | 1.31 | **~5500** | ~1190 | 1182 W |

The driver is **low entropy / no sign-bit toggling**, not the zero fraction (only ~10% zeros).

All three pinned at the **same ~1180 W cap**; only the clock differs. Switching C++ from the trivial
`i%5` fill ([helpers.h:779](../../Common/helpers.h#L779-L783)) to random data drops it from
1582→1250 MHz and 7250→5850 TFLOPS — almost all the way to Python. Root cause: **the cuBLASLt-sample
`TestBench::fillData` uses low-entropy operands that under-load the FP4 multipliers, so the bench
holds a higher clock under the power cap and OVER-REPORTS throughput.** The Python benches quantize
`torch.randn`, which is realistic.

## 5. Sustained (≥1.5 s) Python steady-state
`diag_sustained.py` (eager) and `diag_graph_sustained.py` (CUDA graph), 16384³ fp32, >3 s
continuous, NVML-sampled, single final sync:
- **Steady-state ≈ 5500 TFLOPS @ 1177–1190 MHz, 1180 W** (eager 5562, graph 5495 — agree).
- The suite CSV value (`bench_kineto`, flush_l2=False) is **6402 TFLOPS** — higher, because its window
  is only ~82 ms (2×30 kernels), so it is **partially un-throttled** and over-reads the true sustained
  number by ~15%.

## Takeaways (keep the two gaps separate)
1. **C++ (7250, TestBench data, sustained) vs Python (~5500, randn, sustained):** ≈24% gap, **almost
   entirely operand data entropy** (C++ with random data drops to ~5850, closing most of it).
2. **Python suite `bench_kineto` (6402) vs Python true sustained (~5500):** ≈10–15%, a *separate*
   effect — the suite's ~82 ms window is too short to reach throttle steady state, so the suite
   over-reports its own number. (This is about the Python suite, not the C++ bench, which runs ≥1.5 s.)
3. For a fair NVFP4 throughput number use **realistic random operands** AND a **sustained ≥1.5 s**
   window. Representative 16384³ NVFP4→fp32 ≈ **~5500 TFLOPS @ ~1180 MHz**; the C++ `i%5` number
   (~8400 peak / 7250 sustained) is optimistic.

## Repro flags added to `bench_smmhz.cu`
`SMMHZ_RANDOM=1` (random fp4 operands), `SMMHZ_NOUT=N` (rotate N output buffers), `SMMHZ_OOP=1`
(out-of-place C≠D), `SMMHZ_WS_MB=N` (workspace MB). Python diagnostics in `python/nvfp4/diag_*.py`.
