// SM frequency-throttling-vs-sustained-runtime experiment for cuBLASLt NVFP4 dense GEMM.
//
// Hypothesis under test: whether the SM clock droops is governed mainly by how long the SM
// has been running *continuously*. So we run one fixed shape (16384^3, A,B = NVFP4, D = fp32,
// C = void/unused with beta=0) and keep replaying a CUDA graph back-to-back for >2s WITHOUT
// ever stopping the SM, while recording how replay time / TFLOPS / SM MHz / power evolve as the
// continuous-run time grows.
//
// KEY DESIGN (answers "do I need to sync mid-run, which would stall the SM?"  -> NO):
//   * The whole timed region issues graph replays back-to-back into one stream and does ZERO
//     host synchronization until the very end. The SM is therefore never drained mid-run.
//   * Per-segment GPU time is taken from CUDA events used as IN-STREAM timestamps. cudaEventRecord
//     only enqueues a marker; it does not block the host nor serialize/stop the SMs. We read the
//     event deltas (cudaEventElapsedTime) only AFTER the single final cudaDeviceSynchronize.
//   * SM clock + power come from an asynchronous host-side NVML sampler thread that never touches
//     the GPU's execution; each sample is host-timestamped.
//   * Alignment: because the SM runs continuously from the start anchor, GPU time advances at the
//     same rate as host wall time, so host_time(gpu_t) = t_host_start + gpu_t. We map each event
//     segment's GPU window into host time and take the median NVML clock/power inside it. We then
//     VALIDATE this by comparing (t_host_end - t_host_start) against the summed GPU time.
//
// Build:
//   export PATH=/usr/local/cuda/bin:$PATH
//   nvcc -O3 -std=c++17 -arch=sm_100a bench_smmhz.cu -I../../Common -lcublasLt -lnvidia-ml -o bench_smmhz
// Run:
//   ./bench_smmhz [m n k] [graphs_per_seg] [target_seconds] [out.csv]
//   default: 16384 16384 16384  1  2.5  smmhz_16384.csv
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <nvml.h>

#include <algorithm>
#include <chrono>
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
// median of selected field over host-time window [t0,t1]
template <typename Sel> static unsigned medianIn(double t0, double t1, Sel sel) {
    std::vector<unsigned> v;
    for (auto &s : g_samp) if (s.t >= t0 && s.t <= t1) v.push_back(sel(s));
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

int main(int argc, char **argv) {
    int m = 16384, n = 16384, k = 16384;
    int graphsPerSeg = 1;       // replays (graph launches) between consecutive event markers
    double targetSec = 2.5;     // continuous-run target
    std::string outFile = "smmhz_16384.csv";
    int a = 1;
    if (argc > a + 2 && argv[1][0] >= '0' && argv[1][0] <= '9') { m = atoi(argv[1]); n = atoi(argv[2]); k = atoi(argv[3]); a = 4; }
    if (argc > a) graphsPerSeg = atoi(argv[a++]);
    if (argc > a) targetSec = atof(argv[a++]);
    if (argc > a) outFile = argv[a++];
    if (graphsPerSeg < 1) graphsPerSeg = 1;

    const int G = 10;  // cublasLtMatmul calls captured per graph (one replay == G GEMMs)

    nvmlInit();
    nvmlDeviceGetHandleByIndex(0, &g_dev);
    std::thread th(sampler);

    const auto MA = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
    const auto SC = CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
    const size_t WS = 32ULL * 1024 * 1024;
    const float alpha = 1.0f, beta = 0.0f;

    TB props(CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, alpha, beta, WS, 1, MA, MA, SC, SC, MA);
    props.copyDataToDevice();
    props.streamSynchronize();
    cudaStream_t stream = props.stream;
    cublasLtHandle_t lt = props.ltHandle;

    // fp32 output D (== DeepGEMM fp32 D). beta=0 -> C unread -> reuse buffer for both C and D.
    float *Dfp32 = nullptr;
    checkCudaStatus(cudaMalloc(reinterpret_cast<void **>(&Dfp32), (size_t)m * n * sizeof(float)));

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

    auto matmul = [&]() {
        checkCublasStatus(cublasLtMatmul(lt, op, &alpha, props.Adev, Ad, props.Bdev, Bd, &beta,
                                         Dfp32, Cd, Dfp32, Dd, &heur.algo, props.workspace, WS, stream));
    };
    // build a graph of G back-to-back matmuls
    for (int i = 0; i < 5; i++) matmul();
    checkCudaStatus(cudaStreamSynchronize(stream));
    cudaGraph_t graph; cudaGraphExec_t exec;
    checkCudaStatus(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    for (int j = 0; j < G; j++) matmul();
    checkCudaStatus(cudaStreamEndCapture(stream, &graph));
    checkCudaStatus(cudaGraphInstantiate(&exec, graph, 0));

    // --- pre-measure ONE replay (before the timed region; sync allowed here) to size the run ---
    for (int i = 0; i < 2; i++) checkCudaStatus(cudaGraphLaunch(exec, stream));
    checkCudaStatus(cudaStreamSynchronize(stream));
    cudaEvent_t ps, pe; cudaEventCreate(&ps); cudaEventCreate(&pe);
    checkCudaStatus(cudaEventRecord(ps, stream));
    checkCudaStatus(cudaGraphLaunch(exec, stream));
    checkCudaStatus(cudaEventRecord(pe, stream));
    checkCudaStatus(cudaEventSynchronize(pe));
    float replayMs = 0; cudaEventElapsedTime(&replayMs, ps, pe);
    cudaEventDestroy(ps); cudaEventDestroy(pe);

    long totalReplays = (long)(targetSec * 1000.0 / replayMs) + 1;
    long nSeg = totalReplays / graphsPerSeg + 1;
    totalReplays = nSeg * graphsPerSeg;
    printf("shape %dx%dx%d | replay ~%.3f ms (%d GEMMs) | target %.2fs -> %ld replays, %ld segments (%d replays/seg)\n",
           m, n, k, replayMs, G, targetSec, totalReplays, nSeg, graphsPerSeg);

    // pre-create all event markers (nSeg+1)
    std::vector<cudaEvent_t> ev(nSeg + 1);
    for (auto &e : ev) checkCudaStatus(cudaEventCreate(&e));

    // let the GPU go idle so the clock/power recover to the un-throttled peak; capture the transient
    checkCudaStatus(cudaStreamSynchronize(stream));
    std::this_thread::sleep_for(std::chrono::milliseconds(600));

    // ====================== continuous timed region: NO host sync inside ======================
    double tHostStart = nowS();
    checkCudaStatus(cudaEventRecord(ev[0], stream));
    for (long s = 0; s < nSeg; s++) {
        for (int r = 0; r < graphsPerSeg; r++) checkCudaStatus(cudaGraphLaunch(exec, stream));
        checkCudaStatus(cudaEventRecord(ev[s + 1], stream));
    }
    checkCudaStatus(cudaDeviceSynchronize());   // the ONLY sync; SM ran continuously up to here
    double tHostEnd = nowS();
    // =========================================================================================

    // GPU run was continuous from tHostStart, so host_time(gpu_t) = tHostStart + gpu_t.
    float totalGpuMs = 0; cudaEventElapsedTime(&totalGpuMs, ev[0], ev[nSeg]);
    double hostWallMs = (tHostEnd - tHostStart) * 1000.0;
    double drift = hostWallMs - (double)totalGpuMs;
    printf("GPU continuous run = %.1f ms ; host wall = %.1f ms ; drift = %.1f ms (%.2f%%) %s\n",
           totalGpuMs, hostWallMs, drift, 100.0 * drift / totalGpuMs,
           (drift / totalGpuMs < 0.05 ? "[OK: clock/power alignment valid]" : "[WARN: large drift, alignment loose]"));

    std::ofstream out(outFile);
    out << "seg,replay_start,replay_end,cum_run_ms,seg_ms,replay_us,tflops,sm_mhz,power_w\n";
    double cumMs = 0;
    for (long s = 0; s < nSeg; s++) {
        float segMs = 0; cudaEventElapsedTime(&segMs, ev[s], ev[s + 1]);
        double segStartHost = tHostStart + cumMs / 1000.0;
        double segEndHost = tHostStart + (cumMs + segMs) / 1000.0;
        unsigned smMhz = medianIn(segStartHost, segEndHost, [](const Samp &p) { return p.clk; });
        unsigned pwW = medianIn(segStartHost, segEndHost, [](const Samp &p) { return p.pw; });
        double replayUs = (double)segMs * 1000.0 / graphsPerSeg / G;  // per single GEMM
        double secs = (double)segMs / 1000.0;
        double gemms = (double)graphsPerSeg * G;
        double tflops = gemms * 2.0 * m * n * k / secs / 1e12;
        cumMs += segMs;
        out << s << "," << s * graphsPerSeg << "," << (s + 1) * graphsPerSeg << ","
            << cumMs << "," << segMs << "," << replayUs << "," << tflops << ","
            << smMhz << "," << pwW << "\n";
    }
    out.flush();
    printf("wrote %ld rows -> %s\n", nSeg, outFile.c_str());

    g_stop = true; th.join(); nvmlShutdown();
    for (auto &e : ev) cudaEventDestroy(e);
    cudaGraphExecDestroy(exec); cudaGraphDestroy(graph);
    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(Dd); cublasLtMatrixLayoutDestroy(Cd);
    cublasLtMatrixLayoutDestroy(Bd); cublasLtMatrixLayoutDestroy(Ad);
    cublasLtMatmulDescDestroy(op);
    cudaFree(Dfp32);
    return 0;
}
