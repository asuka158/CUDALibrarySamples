# SM frequency-droop vs sustained-runtime experiment (NVFP4 dense GEMM)

Quantifies how the GB200 SM clock droops as a single kernel is **replayed continuously**, to test
the hypothesis that the droop is governed mainly by how long the SM has run without stopping.

Fixed shape **16384×16384×16384**, A,B = NVFP4 (VEC16 UE4M3 scales), **D = fp32**, C = void (beta=0),
transa=T/transb=N, heuristic algo — same cuBLASLt setup as `../my_test_v1/bench_nvfp4_fp32.cu`.

## How the "no mid-run sync" requirement is met (the feasibility question)
Replays must never stop or the SM idles, yet we want per-segment data. So measurement is fully
decoupled from execution:
- **Zero host sync inside the timed region.** It issues `cudaGraphLaunch` back-to-back into one
  stream and only `cudaDeviceSynchronize` once, at the very end.
- **Per-segment GPU time = CUDA events as in-stream timestamps.** `cudaEventRecord` only enqueues a
  marker; it does not block the host or stop the SMs. Event deltas are read *after* the final sync.
- **SM MHz + power = async NVML sampler thread** (1 ms cadence, host-timestamped); never touches GPU
  execution.
- **Alignment:** since the SM runs continuously from the start anchor, `host_time(gpu_t) =
  t_host_start + gpu_t`. Each event segment's GPU window is mapped to host time and the median NVML
  clock/power inside it is taken. Validated by comparing host wall time vs summed GPU time —
  observed **drift = 0.00%**, confirming the SM never stalled and the alignment is exact.

Before timing, the GPU is idled 600 ms so clock/power recover to the un-throttled peak, capturing the
full transient from 2062 MHz down into throttle.

## Build / run
    export PATH=/usr/local/cuda/bin:$PATH
    nvcc -O3 -std=c++17 -arch=sm_100a bench_smmhz.cu -I../../Common -lcublasLt -lnvidia-ml -o bench_smmhz
    ./bench_smmhz                                 # 16384^3, 2.5s, ~25ms/seg -> smmhz_16384.csv
    ./bench_smmhz 16384 16384 16384 8.0 25 out.csv  # args: m n k total_sec seg_ms out.csv

The segment (one CSV row) is **auto-sized** to ~`seg_ms` from a per-GEMM calibration, so it works
from 1K (single GEMM ~4 us) to 32K (single GEMM ~10 ms): it picks G (matmuls captured per graph,
≤512) and graphs-per-segment so one segment ≈ seg_ms and ≥1 GEMM.

## CSV columns
`seg, gemm_start, gemm_end, cum_run_ms, seg_ms, gemm_us, tflops, sm_mhz, power_w`
`cum_run_ms` is the continuous-run time (the x-axis); `gemm_us` is per single GEMM.

## Result 1 — droop transient & true steady state (16384³, 9 s run)
| cum_run | sm_mhz | tflops | power_w |
|--------:|-------:|-------:|--------:|
| ~0.03 s | 2062   | 8413   | 329 (power ramping) |
| ~0.5 s  | 1620   | 7441   | 671 |
| ~1.0 s  | 1530   | 7219   | 1124 (power hits cap, clock undershoots) |
| ~1.5 s  | 1567   | 7207   | 1176 |
| 1.5→9.25 s | **1582 ±15** | **~7260 ±1%** | **1177–1178** |

The power loop ramps to the 1200 W cap over ~1 s (with a slight clock undershoot near 1.0 s), then
**locks to a true steady state by ~1.5 s and stays flat through 9.25 s** (sm 1582 ±15 MHz, power
1177–1178 W). So yes — beyond ~1.5 s it is genuinely stable; longer runtime brings no further droop.

## Result 2 — shape sweep (`result/shape_sweep_summary.csv`)
| shape | steady sm | droop | power | peak→steady TFLOPS | throttled |
|------:|----------:|------:|------:|-------------------:|:---------:|
| 1024³ | 2062 | 0%   | 393 W  | 665→664   | no |
| 2048³ | 2062 | 0%   | 710 W  | 2751→2730 | no |
| 4096³ | 2062 | 0%   | 1129 W | 5584→5534 | no (just under cap) |
| 8192³ | 1642 | 20%  | 1178 W | 7356→6605 | **yes** |
| 16384³| 1582 | 23%  | 1178 W | 8544→7248 | **yes** |
| 32768³| 1455 | 29%  | 1181 W | 8224→6974 | **yes** |
| 16384²×65536 | 1477 | 28% | 1180 W | 8565→7139 | **yes** |

**The throttle is power-cap-driven, not runtime-driven per se.** Small/low-power shapes (≤4096³)
never reach the 1200 W cap and hold 2062 MHz indefinitely — no droop no matter how long they run.
Only shapes that saturate the cap (here ≥8192³) droop, and the droop deepens with intensity
(20%→29%). "Continuous-run time" matters only because it takes ~1 s of sustained load for power to
ramp to the cap; the steady clock is then set by how far over the cap the shape would push.

Limit: square shapes >32768 (49152³, 65536³) fail — the **TestBench harness** overflows 32-bit
element counts once a dimension product exceeds INT_MAX (√2³¹ ≈ 46340), so 32768³ is the largest
power-of-two square. Not a GPU/NVFP4 limit (the k-heavy 16384²×65536 = 1.07e9 elems runs fine).
