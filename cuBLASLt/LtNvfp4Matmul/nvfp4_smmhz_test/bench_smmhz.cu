// SM frequency-throttling-vs-sustained-runtime experiment for cuBLASLt NVFP4 dense GEMM.
//
// Hypothesis under test: whether the SM clock droops is governed mainly by how long the SM
// has been running *continuously*. So we run one fixed shape (A,B = NVFP4, D = fp32, C = void
// with beta=0) and keep replaying a CUDA graph back-to-back for several seconds WITHOUT ever
// stopping the SM, recording how replay time / TFLOPS / SM MHz / power evolve as the continuous
// run time grows -- and whether they reach a true steady state.
//
// KEY DESIGN (no host sync inside the timed region -> SM is never drained):
//   * Whole timed region issues graph replays back-to-back into one stream; ZERO host sync until
//     the single final cudaDeviceSynchronize.
//   * Per-segment GPU time = CUDA events as IN-STREAM timestamps (cudaEventRecord only enqueues a
//     marker; it neither blocks the host nor stops the SMs). Event deltas read after final sync.
//   * SM clock + power = async host NVML sampler thread (1 ms), host-timestamped.
//   * Alignment: SM runs continuously from the start anchor, so host_time(gpu_t)=t_host_start+gpu_t;
//     map each segment's GPU window to host time, take median NVML inside it. Validated by
//     comparing host wall time vs summed GPU time (drift should be ~0%).
//
// AUTO-SIZING: a segment (the unit between two event markers) is auto-sized to ~seg_ms so it works
// across shapes whose single-GEMM time spans microseconds (1K) to tens of ms (64K). We calibrate
// the per-GEMM time, then pick G (matmuls captured per graph, <=512) and graphs-per-segment so that
// one segment ~ seg_ms and is also large enough (>=1 GEMM) for big shapes.
//
// Build:
//   export PATH=/usr/local/cuda/bin:$PATH
//   nvcc -O3 -std=c++17 -arch=sm_100a bench_smmhz.cu -I../../Common -lcublasLt -lnvidia-ml -o bench_smmhz
// Run:
//   ./bench_smmhz [m n k] [total_sec] [seg_ms] [out.csv]
//   default: 16384 16384 16384  2.5  25  smmhz_16384.csv
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <nvml.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include "helpers.h"  // TestBench, checkCublasStatus, checkCudaStatus

using TB = TestBench<__nv_fp4_e2m1, __nv_bfloat16, float, __nv_fp8_e4m3, float, __nv_bfloat16>;

// ---- async NVML sampler (host side, never blocks the GPU) ----
static nvmlDevice_t g_dev;
struct Samp { double t; unsigned clk; unsigned pw; };  // (host time s, SM MHz, power W)
static std::vector<Samp> g_samp;
static volatile bool g_stop = false;
static double nowS() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
static void sampler() {
    g_samp.reserve(1 << 20);
    while (!g_stop) {
        unsigned clk = 0, pw = 0;
        nvmlDeviceGetClockInfo(g_dev, NVML_CLOCK_SM, &clk);
        nvmlDeviceGetPowerUsage(g_dev, &pw);  // mW
        g_samp.push_back({nowS(), clk, pw / 1000});
        std::this_thread::sleep_for(std::chrono::microseconds(1000));
    }
}
template <typename Sel> static unsigned medianIn(double t0, double t1, Sel sel) {
    std::vector<unsigned> v;
    for (auto &s : g_samp) if (s.t >= t0 && s.t <= t1) v.push_back(sel(s));
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

int main(int argc, char **argv) {
    int m = 16384, n = 16384, k = 16384;
    double totalSec = 2.5;   // continuous-run duration
    double segMs = 25.0;     // target wall-time per event segment (CSV row)
    std::string outFile = "smmhz_16384.csv";
    int a = 1;
    if (argc > 3 && argv[1][0] >= '0' && argv[1][0] <= '9') { m = atoi(argv[1]); n = atoi(argv[2]); k = atoi(argv[3]); a = 4; }
    if (argc > a) totalSec = atof(argv[a++]);
    if (argc > a) segMs = atof(argv[a++]);
    if (argc > a) outFile = argv[a++];

    nvmlInit();
    nvmlDeviceGetHandleByIndex(0, &g_dev);
    std::thread th(sampler);

    const auto MA = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
    const auto SC = CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
    size_t WS = 32ULL * 1024 * 1024;
    if (const char *e = getenv("SMMHZ_WS_MB")) WS = (size_t)std::max(0, atoi(e)) * 1024 * 1024;
    const float alpha = 1.0f, beta = 0.0f;

    TB props(CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, alpha, beta, WS, 1, MA, MA, SC, SC, MA);
    props.copyDataToDevice();
    props.streamSynchronize();
    cudaStream_t stream = props.stream;
    cublasLtHandle_t lt = props.ltHandle;

    // fp32 output D (beta=0 -> reuse as C and D). SMMHZ_NOUT lets us rotate among N distinct output
    // buffers (default 1 = reuse one buffer, like the original bench). Rotating buffers mimics
    // torch._scaled_mm_v2 allocating a fresh output each call -> more cold HBM writes (the 1GB fp32
    // output dominates traffic), to test the eager-Python vs C++ throughput gap.
    int NOUT = 1;
    if (const char *e = getenv("SMMHZ_NOUT")) NOUT = std::max(1, atoi(e));
    std::vector<float *> Douts(NOUT, nullptr);
    for (auto &d : Douts) checkCudaStatus(cudaMalloc(reinterpret_cast<void **>(&d), (size_t)m * n * sizeof(float)));
    float *Dfp32 = Douts[0];
    // SMMHZ_OOP=1 -> out-of-place (separate C buffer, C != D) like torch._scaled_mm_v2; default in-place (C==D)
    int OOP = 0; if (const char *e = getenv("SMMHZ_OOP")) OOP = atoi(e);
    float *Cbuf = nullptr;
    if (OOP) checkCudaStatus(cudaMalloc(reinterpret_cast<void **>(&Cbuf), (size_t)m * n * sizeof(float)));

    cublasLtMatmulDesc_t op = nullptr;
    cublasLtMatrixLayout_t Ad = nullptr, Bd = nullptr, Cd = nullptr, Dd = nullptr;
    cublasLtMatmulPreference_t pref = nullptr;
    cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
    checkCublasStatus(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    checkCublasStatus(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)));
    checkCublasStatus(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)));
    checkCublasStatus(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &props.AScaleMode, sizeof(MA)));
    checkCublasStatus(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &props.BScaleMode, sizeof(MA)));
    checkCublasStatus(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &props.AscaleDev, sizeof(void *)));
    checkCublasStatus(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &props.BscaleDev, sizeof(void *)));
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Ad, CUDA_R_4F_E2M1, k, m, props.lda));
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Bd, CUDA_R_4F_E2M1, k, n, props.ldb));
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Cd, CUDA_R_32F, m, n, props.ldc));
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Dd, CUDA_R_32F, m, n, props.ldd));
    checkCublasStatus(cublasLtMatmulPreferenceCreate(&pref));
    checkCublasStatus(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &WS, sizeof(WS)));
    int got = 0;
    cublasLtMatmulHeuristicResult_t heur = {};
    checkCublasStatus(cublasLtMatmulAlgoGetHeuristic(lt, op, Ad, Bd, Cd, Dd, pref, 1, &heur, &got));
    if (got == 0) { printf("no algo for %dx%dx%d\n", m, n, k); return 1; }

    auto matmul = [&](int j) {
        float *D = Douts[j % NOUT];
        float *C = OOP ? Cbuf : D;   // C!=D -> outOfPlace=1 (torch); C==D -> outOfPlace=0 (in-place)
        checkCublasStatus(cublasLtMatmul(lt, op, &alpha, props.Adev, Ad, props.Bdev, Bd, &beta,
                                         C, Cd, D, Dd, &heur.algo, props.workspace, WS, stream));
    };
    auto buildGraph = [&](int gN, cudaGraphExec_t &exec, cudaGraph_t &graph) {
        checkCudaStatus(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        for (int j = 0; j < gN; j++) matmul(j);
        checkCudaStatus(cudaStreamEndCapture(stream, &graph));
        checkCudaStatus(cudaGraphInstantiate(&exec, graph, 0));
    };
    if (NOUT > 1) printf("[SMMHZ_NOUT=%d: rotating %d fp32 output buffers]\n", NOUT, NOUT);

    // SMMHZ_RANDOM=1 -> overwrite A,B with full-entropy random fp4 bytes (every byte = 2 fp4 codes,
    // all 16 e2m1 codes are finite, so random bytes are valid). TestBench's default fill is the very
    // low-entropy pattern float(i%5) in {0..4} (lots of zeros) -> low tensor-core switching activity
    // -> low power/cycle -> higher sustained clock under the power cap. Random data mimics torch.randn.
    if (const char *e = getenv("SMMHZ_RANDOM"); e && atoi(e)) {
        auto fillRand = [](void *dev, size_t bytes) {
            std::vector<uint8_t> h(bytes);
            uint32_t s = 0x9e3779b9u;
            for (size_t i = 0; i < bytes; i++) { s ^= s << 13; s ^= s >> 17; s ^= s << 5; h[i] = (uint8_t)(s >> 24); }
            checkCudaStatus(cudaMemcpy(dev, h.data(), bytes, cudaMemcpyHostToDevice));
        };
        fillRand(props.Adev, (size_t)m * k / 2);   // fp4 packed: 2 codes/byte
        fillRand(props.Bdev, (size_t)n * k / 2);
        checkCudaStatus(cudaDeviceSynchronize());
        printf("[SMMHZ_RANDOM=1: A,B filled with random fp4 (high switching activity)]\n");
    }

    // ---- calibrate per-GEMM time (before timed region; sync allowed) ----
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
    double perGemmMs = (double)calMs / Gc;

    // ---- choose G (<=512) and graphsPerSeg so one segment ~ segMs (>=1 GEMM) ----
    long gemmsPerSeg = std::max(1L, lround(segMs / perGemmMs));
    int G = (int)std::min(gemmsPerSeg, 512L);
    int graphsPerSeg = (int)std::max(1L, lround((double)gemmsPerSeg / G));
    gemmsPerSeg = (long)G * graphsPerSeg;
    double actualSegMs = gemmsPerSeg * perGemmMs;
    long nSeg = std::max(1L, lround(totalSec * 1000.0 / actualSegMs));
    printf("shape %dx%dx%d | per-GEMM ~%.4f ms | seg=%ld GEMMs (G=%d x %d launches, ~%.1f ms) | %.1fs -> %ld segments\n",
           m, n, k, perGemmMs, gemmsPerSeg, G, graphsPerSeg, actualSegMs, totalSec, nSeg);

    cudaGraph_t graph; cudaGraphExec_t exec; buildGraph(G, exec, graph);
    for (int i = 0; i < 2; i++) checkCudaStatus(cudaGraphLaunch(exec, stream));
    checkCudaStatus(cudaStreamSynchronize(stream));

    std::vector<cudaEvent_t> ev(nSeg + 1);
    for (auto &e : ev) checkCudaStatus(cudaEventCreate(&e));

    // idle so clock/power recover to the un-throttled peak -> capture the transient from cold
    std::this_thread::sleep_for(std::chrono::milliseconds(600));

    // ================= continuous timed region: NO host sync inside =================
    double tHostStart = nowS();
    checkCudaStatus(cudaEventRecord(ev[0], stream));
    for (long s = 0; s < nSeg; s++) {
        for (int r = 0; r < graphsPerSeg; r++) checkCudaStatus(cudaGraphLaunch(exec, stream));
        checkCudaStatus(cudaEventRecord(ev[s + 1], stream));
    }
    checkCudaStatus(cudaDeviceSynchronize());   // the ONLY sync; SM ran continuously up to here
    double tHostEnd = nowS();
    // ================================================================================

    float totalGpuMs = 0; cudaEventElapsedTime(&totalGpuMs, ev[0], ev[nSeg]);
    double hostWallMs = (tHostEnd - tHostStart) * 1000.0;
    double drift = hostWallMs - (double)totalGpuMs;
    printf("GPU continuous run = %.1f ms ; host wall = %.1f ms ; drift = %.2f%% %s\n",
           totalGpuMs, hostWallMs, 100.0 * drift / totalGpuMs,
           (fabs(drift) / totalGpuMs < 0.05 ? "[OK alignment]" : "[WARN drift]"));

    std::ofstream out(outFile);
    out << "seg,gemm_start,gemm_end,cum_run_ms,seg_ms,gemm_us,tflops,sm_mhz,power_w\n";
    double cumMs = 0;
    std::vector<unsigned> tailSm, tailPw; std::vector<double> tailTf;  // last 1s for steady-state stats
    for (long s = 0; s < nSeg; s++) {
        float sMs = 0; cudaEventElapsedTime(&sMs, ev[s], ev[s + 1]);
        double h0 = tHostStart + cumMs / 1000.0, h1 = tHostStart + (cumMs + sMs) / 1000.0;
        unsigned smMhz = medianIn(h0, h1, [](const Samp &p) { return p.clk; });
        unsigned pwW = medianIn(h0, h1, [](const Samp &p) { return p.pw; });
        double gemmUs = (double)sMs * 1000.0 / gemmsPerSeg;
        double tflops = gemmsPerSeg * 2.0 * m * n * k / ((double)sMs / 1000.0) / 1e12;
        cumMs += sMs;
        out << s << "," << s * gemmsPerSeg << "," << (s + 1) * gemmsPerSeg << "," << cumMs << ","
            << sMs << "," << gemmUs << "," << tflops << "," << smMhz << "," << pwW << "\n";
        if (cumMs >= totalGpuMs - 1000.0) { tailSm.push_back(smMhz); tailPw.push_back(pwW); tailTf.push_back(tflops); }
    }
    out.flush();

    // steady-state summary over the last ~1 s
    auto mm = [](std::vector<unsigned> &v) { std::sort(v.begin(), v.end()); return std::pair<unsigned,unsigned>{v.front(), v.back()}; };
    if (!tailSm.empty()) {
        auto [smn, smx] = mm(tailSm); auto [pmn, pmx] = mm(tailPw);
        std::sort(tailTf.begin(), tailTf.end());
        printf("steady-state (last 1s, %zu segs): sm_mhz %u-%u (spread %u) | power_w %u-%u | tflops %.0f-%.0f\n",
               tailSm.size(), smn, smx, smx - smn, pmn, pmx, tailTf.front(), tailTf.back());
    }
    printf("wrote %ld rows -> %s\n", nSeg, outFile.c_str());

    g_stop = true; th.join(); nvmlShutdown();
    for (auto &e : ev) cudaEventDestroy(e);
    cudaGraphExecDestroy(exec); cudaGraphDestroy(graph);
    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(Dd); cublasLtMatrixLayoutDestroy(Cd);
    cublasLtMatrixLayoutDestroy(Bd); cublasLtMatrixLayoutDestroy(Ad);
    cublasLtMatmulDescDestroy(op);
    for (auto &d : Douts) cudaFree(d);
    return 0;
}
