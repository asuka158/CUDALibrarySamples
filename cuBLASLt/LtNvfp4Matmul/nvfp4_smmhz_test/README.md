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
    ./bench_smmhz                                   # 16384^3, 1 replay/seg, 2.5s -> smmhz_16384.csv
    ./bench_smmhz 16384 16384 16384 1 2.5 out.csv   # explicit args: m n k graphs_per_seg target_s out

`graphs_per_seg` = replays between event markers (coarser granularity if >1). One replay = 10 GEMMs.

## CSV columns
`seg, replay_start, replay_end, cum_run_ms, seg_ms, replay_us, tflops, sm_mhz, power_w`
`cum_run_ms` is the continuous-run time (the x-axis); `replay_us` is per single GEMM.

## Result (this run)
Droop is real and tracks the 1200 W power cap, not a fixed time:
| cum_run | sm_mhz | tflops | power_w |
|--------:|-------:|-------:|--------:|
| ~50 ms  | 2062   | 8543   | 269 (power ramping) |
| ~1.5 s  | ~1550  | ~7120  | ~1184 (at cap) |
| ~2.9 s  | 1582   | 7265   | 1177 |

SM holds 2062 MHz while power climbs, then droops to ~1480–1580 MHz once power saturates the 1200 W
cap; TFLOPS falls 8543 → ~7260 (~15%). I.e. the droop is power-cap driven and sets in after the
power-management loop ramps to the cap (~hundreds of ms of continuous run), not instantly.
