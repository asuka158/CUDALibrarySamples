// Shared machinery for the graph-sustained cuBLASLt NVFP4 GEMM benchmarks.
// Used by nvfp4_nvfp4/ (NVFP4 x NVFP4 -> NVFP4). The sibling FP8-output experiment was dropped:
// cuBLASLt has no NVFP4-inputs -> FP8-output path (see nvfp4_fp8/README.md).
// Mirrors the Python python/{nvfp4,mxfp4}_v2 methodology:
//
//   * capture G cublasLtMatmul calls into a CUDA graph, replay back-to-back for ~SUSTAIN_S with
//     NO host sync inside the timed region (SM never drains), per-segment cudaEvents as in-stream
//     timestamps + an async NVML sampler, single final cudaDeviceSynchronize.
//   * report ONLY the steady-state tail (last TAIL_S): the throttled, sustained number.
//
// Data generation (per user request): NOT TestBench's low-entropy float(i%5) fill. Instead A,B are
// overwritten with N(0,1) samples quantized to NVFP4 e2m1 on-device (curand + the hardware float2->
// fp4x2 converter), to reproduce torch.randn's fp4 code distribution -> realistic switching activity
// -> realistic sustained power/clock (i%5's low entropy otherwise inflates the clock under the cap).
#pragma once
#include <cuda_runtime.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <curand_kernel.h>
#include <nvml.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include "helpers.h"  // checkCudaStatus, checkCublasStatus

// ---------------- methodology constants (match python v2) ----------------
static constexpr double SUSTAIN_S = 3.0;   // back-to-back replay duration per shape
static constexpr double SEG_MS    = 20.0;  // target wall time per event segment
static constexpr double TAIL_S    = 1.0;   // only the last TAIL_S is the steady-state record

// ---------------- async NVML sampler (host side, never blocks the GPU) ----------------
static nvmlDevice_t g_dev;
struct Samp { double t; unsigned clk; unsigned pw; };  // host time s, SM MHz, power W
static std::vector<Samp> g_samp;
static volatile bool g_stop = false;

static inline double nowS() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
static void samplerLoop() {
    g_samp.reserve(1 << 20);  // reserved once -> no realloc -> safe to index while thread appends
    while (!g_stop) {
        unsigned clk = 0, pw = 0;
        nvmlDeviceGetClockInfo(g_dev, NVML_CLOCK_SM, &clk);
        nvmlDeviceGetPowerUsage(g_dev, &pw);  // mW
        g_samp.push_back({nowS(), clk, pw / 1000});
        std::this_thread::sleep_for(std::chrono::microseconds(1000));
    }
}
template <typename Sel>
static unsigned medianIn(size_t start, double t0, double t1, Sel sel) {
    std::vector<unsigned> v;
    for (size_t i = start; i < g_samp.size(); i++)
        if (g_samp[i].t >= t0 && g_samp[i].t <= t1) v.push_back(sel(g_samp[i]));
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// ---------------- shapes ----------------
static std::vector<std::array<int, 3>> readShapes(const std::string &path) {
    std::vector<std::array<int, 3>> shapes;
    std::ifstream f(path);
    if (!f) { printf("cannot open shapes file %s\n", path.c_str()); exit(1); }
    std::string line;
    while (std::getline(f, line)) {
        size_t i = line.find_first_not_of(" \t\r\n");
        if (i == std::string::npos || !(line[i] >= '0' && line[i] <= '9')) continue;  // skip header/blank
        int m, n, k;
        if (sscanf(line.c_str(), "%d %d %d", &m, &n, &k) == 3) shapes.push_back({m, n, k});
    }
    return shapes;
}

// ---------------- N(0,1) -> NVFP4 fill (mimics torch.randn quantized to nvfp4) ----------------
__global__ void fillRandnFp4Kernel(__nv_fp4x2_e2m1 *out, size_t nPairs,
                                   unsigned long long seed, float scale) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nPairs) return;
    curandStatePhilox4_32_10_t st;
    curand_init(seed, i, 0, &st);   // Philox init is O(1)
    float2 g = curand_normal2(&st); // two N(0,1) samples
    g.x *= scale; g.y *= scale;     // spread across the e2m1 range like block-scaled randn (amax->6)
    out[i] = __nv_fp4x2_e2m1(g);    // round-to-nearest fp4, hardware converter
}
static void fillRandnFp4(void *dev, size_t nElems, unsigned long long seed, float scale,
                         cudaStream_t s) {
    size_t nPairs = (nElems + 1) / 2;
    int tpb = 256;
    size_t blocks = (nPairs + tpb - 1) / tpb;
    fillRandnFp4Kernel<<<(unsigned)blocks, tpb, 0, s>>>(
        reinterpret_cast<__nv_fp4x2_e2m1 *>(dev), nPairs, seed, scale);
    checkCudaStatus(cudaGetLastError());
}

// ---------------- graph-sustained steady-state measurement ----------------
struct SustainResult { double us, tflops; unsigned sm_mhz, power_w; double driftPct; };

// matmul(j): enqueue one cublasLtMatmul on `stream` (j is a free index, e.g. for buffer rotation).
template <typename MatmulFn>
static SustainResult runSustained(MatmulFn matmul, int m, int n, int k, cudaStream_t stream) {
    const double flop = 2.0 * (double)m * n * k;

    auto buildGraph = [&](int gN, cudaGraphExec_t &exec, cudaGraph_t &graph) {
        checkCudaStatus(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        for (int j = 0; j < gN; j++) matmul(j);
        checkCudaStatus(cudaStreamEndCapture(stream, &graph));
        checkCudaStatus(cudaGraphInstantiate(&exec, graph, 0));
    };

    // --- calibrate per-GEMM time (sync allowed here, outside the timed region) ---
    for (int i = 0; i < 5; i++) matmul(i);
    checkCudaStatus(cudaStreamSynchronize(stream));
    const int Gc = 8;
    cudaGraph_t cg; cudaGraphExec_t cexec; buildGraph(Gc, cexec, cg);
    for (int i = 0; i < 2; i++) checkCudaStatus(cudaGraphLaunch(cexec, stream));
    checkCudaStatus(cudaStreamSynchronize(stream));
    cudaEvent_t cs, ce; cudaEventCreate(&cs); cudaEventCreate(&ce);
    checkCudaStatus(cudaEventRecord(cs, stream));
    checkCudaStatus(cudaGraphLaunch(cexec, stream));
    checkCudaStatus(cudaEventRecord(ce, stream));
    checkCudaStatus(cudaEventSynchronize(ce));
    float calMs = 0; cudaEventElapsedTime(&calMs, cs, ce);
    cudaEventDestroy(cs); cudaEventDestroy(ce);
    cudaGraphExecDestroy(cexec); cudaGraphDestroy(cg);
    double perGemmMs = std::max((double)calMs / Gc, 1e-4);

    // --- size G (matmuls/graph, <=512) and graphsPerSeg so one segment ~ SEG_MS ---
    long gemmsPerSeg = std::max(1L, lround(SEG_MS / perGemmMs));
    int G = (int)std::min(gemmsPerSeg, 512L);
    int graphsPerSeg = (int)std::max(1L, lround((double)gemmsPerSeg / G));
    gemmsPerSeg = (long)G * graphsPerSeg;
    double actualSegMs = gemmsPerSeg * perGemmMs;
    long nSeg = std::max(2L, lround(SUSTAIN_S * 1000.0 / actualSegMs));

    cudaGraph_t graph; cudaGraphExec_t exec; buildGraph(G, exec, graph);
    for (int i = 0; i < 2; i++) checkCudaStatus(cudaGraphLaunch(exec, stream));
    checkCudaStatus(cudaStreamSynchronize(stream));

    std::vector<cudaEvent_t> ev(nSeg + 1);
    for (auto &e : ev) checkCudaStatus(cudaEventCreate(&e));

    // idle so clock/power recover -> the loop captures the transient down to steady state
    checkCudaStatus(cudaDeviceSynchronize());
    std::this_thread::sleep_for(std::chrono::milliseconds(300));

    // ===== continuous timed region: NO host sync inside =====
    size_t sampStart = g_samp.size();
    double tHostStart = nowS();
    checkCudaStatus(cudaEventRecord(ev[0], stream));
    for (long s = 0; s < nSeg; s++) {
        for (int r = 0; r < graphsPerSeg; r++) checkCudaStatus(cudaGraphLaunch(exec, stream));
        checkCudaStatus(cudaEventRecord(ev[s + 1], stream));
    }
    checkCudaStatus(cudaDeviceSynchronize());  // the ONLY sync
    double tHostEnd = nowS();
    // ========================================================

    float totalGpuMs = 0; cudaEventElapsedTime(&totalGpuMs, ev[0], ev[nSeg]);
    double hostWallMs = (tHostEnd - tHostStart) * 1000.0;
    double drift = 100.0 * (hostWallMs - totalGpuMs) / totalGpuMs;

    double cumMs = 0;
    std::vector<unsigned> tSm, tPw; std::vector<double> tUs, tTf;
    for (long s = 0; s < nSeg; s++) {
        float sMs = 0; cudaEventElapsedTime(&sMs, ev[s], ev[s + 1]);
        double h0 = tHostStart + cumMs / 1000.0, h1 = tHostStart + (cumMs + sMs) / 1000.0;
        unsigned sm = medianIn(sampStart, h0, h1, [](const Samp &p) { return p.clk; });
        unsigned pw = medianIn(sampStart, h0, h1, [](const Samp &p) { return p.pw; });
        cumMs += sMs;
        if (cumMs >= totalGpuMs - TAIL_S * 1000.0) {
            tUs.push_back((double)sMs * 1000.0 / gemmsPerSeg);
            tTf.push_back(gemmsPerSeg * flop / ((double)sMs / 1000.0) / 1e12);
            tSm.push_back(sm); tPw.push_back(pw);
        }
    }
    for (auto &e : ev) cudaEventDestroy(e);
    cudaGraphExecDestroy(exec); cudaGraphDestroy(graph);

    auto medD = [](std::vector<double> v) { std::sort(v.begin(), v.end()); return v.empty() ? 0.0 : v[v.size() / 2]; };
    auto medU = [](std::vector<unsigned> v) { std::sort(v.begin(), v.end()); return v.empty() ? 0u : v[v.size() / 2]; };
    return SustainResult{medD(tUs), medD(tTf), medU(tSm), medU(tPw), drift};
}
