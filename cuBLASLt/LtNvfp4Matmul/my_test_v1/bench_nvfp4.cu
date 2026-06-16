// Benchmark cuBLASLt NVFP4 dense GEMM: CUDA-graph replay x10 timed with cudaEvents.
// Reproduces nvfp4_dense_gemm.csv. Matmul config is exactly the LtNvfp4Matmul sample
// (A,B = CUDA_R_4F_E2M1 with VEC16 UE4M3 block scales, C = bf16, D = fp4, transa=T transb=N,
// alpha=2 beta=1, heuristic-selected algo); TestBench (helpers.h) does the operand/scale alloc.
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <nvml.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include "helpers.h"  // TestBench, StorageType, checkCublasStatus, checkCudaStatus

// nvfp4 A/B -> bf16 output D (no output requant) — matches the reference (gbps counts a bf16 D).
using TB = TestBench<__nv_fp4_e2m1, __nv_bfloat16, float, __nv_fp8_e4m3, float, __nv_bfloat16>;

// ---- NVML background sampler ----
static nvmlDevice_t g_dev;
static std::vector<std::pair<double, std::pair<unsigned, unsigned>>> g_samp;  // (t, (clk, pw_w))
static volatile bool g_stop = false;
static double nowS() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
static void sampler() {
    while (!g_stop) {
        unsigned clk = 0, pw = 0;
        nvmlDeviceGetClockInfo(g_dev, NVML_CLOCK_SM, &clk);
        nvmlDeviceGetPowerUsage(g_dev, &pw);  // mW
        g_samp.push_back({nowS(), {clk, pw / 1000}});
        std::this_thread::sleep_for(std::chrono::microseconds(1500));
    }
}
template <typename Sel> static unsigned medianIn(double t0, double t1, Sel sel) {
    std::vector<unsigned> v;
    for (auto &s : g_samp) if (s.first >= t0 && s.first <= t1) v.push_back(sel(s.second));
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

int main(int argc, char **argv) {
    const char *shapeFile = argc > 1 ? argv[1] : "shapes.txt";
    const char *outFile = argc > 2 ? argv[2] : "../result/nvfp4_dense_gemm_repro.csv";
    nvmlInit();
    nvmlDeviceGetHandleByIndex(0, &g_dev);
    std::thread th(sampler);

    std::ifstream in(shapeFile);
    std::ofstream out(outFile);
    out << "m,n,k,us,tflops,gbps,sm_mhz,power_w,backend\n";
    printf("%6s%7s%7s | %9s %8s %9s | %6s %7s\n", "m", "n", "k", "us", "TFLOPS", "GB/s", "sm", "pw");

    const auto MA = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
    const auto SC = CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
    const size_t WS = 32ULL * 1024 * 1024;
    const float alpha = 1.0f, beta = 0.0f;

    int m, n, k;
    while (in >> m >> n >> k) {
        try {
            TB props(CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, alpha, beta, WS, 1, MA, MA, SC, SC, MA);
            props.copyDataToDevice();
            props.streamSynchronize();
            cudaStream_t stream = props.stream;
            cublasLtHandle_t lt = props.ltHandle;

            // ---- build matmul descriptors once (== LtNvfp4Matmul, but kept for graph capture) ----
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
            checkCublasStatus(cublasLtMatrixLayoutCreate(&Cd, CUDA_R_16BF, m, n, props.ldc));
            checkCublasStatus(cublasLtMatrixLayoutCreate(&Dd, CUDA_R_16BF, m, n, props.ldd));
            checkCublasStatus(cublasLtMatmulPreferenceCreate(&pref));
            checkCublasStatus(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &WS, sizeof(WS)));
            int got = 0;
            cublasLtMatmulHeuristicResult_t heur = {};
            checkCublasStatus(cublasLtMatmulAlgoGetHeuristic(lt, op, Ad, Bd, Cd, Dd, pref, 1, &heur, &got));
            if (got == 0) {
                printf("%6d%7d%7d | no algo\n", m, n, k);
            } else {
                auto matmul = [&]() {
                    checkCublasStatus(cublasLtMatmul(lt, op, &alpha, props.Adev, Ad, props.Bdev, Bd, &beta,
                                                     props.Cdev, Cd, props.Ddev, Dd, &heur.algo, props.workspace, WS, stream));
                };
                // warmup, then capture the matmul into a CUDA graph
                for (int i = 0; i < 5; i++) matmul();
                checkCudaStatus(cudaStreamSynchronize(stream));
                cudaGraph_t graph; cudaGraphExec_t exec;
                checkCudaStatus(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
                for (int j = 0; j < 10; j++) matmul();  // 10 matmuls per graph (back-to-back, no per-launch CPU gaps)
                checkCudaStatus(cudaStreamEndCapture(stream, &graph));
                checkCudaStatus(cudaGraphInstantiate(&exec, graph, 0));

                for (int i = 0; i < 2; i++) checkCudaStatus(cudaGraphLaunch(exec, stream));  // warmup replays
                checkCudaStatus(cudaStreamSynchronize(stream));

                // time: ONE event pair around ONE replay of the 10-matmul graph -> us = elapsed/10
                cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
                checkCudaStatus(cudaStreamSynchronize(stream));
                checkCudaStatus(cudaEventRecord(s, stream));
                checkCudaStatus(cudaGraphLaunch(exec, stream));
                checkCudaStatus(cudaEventRecord(e, stream));
                checkCudaStatus(cudaEventSynchronize(e));
                float ms = 0; cudaEventElapsedTime(&ms, s, e);
                double us = ms * 1000.0 / 10.0;

                // telemetry: short sustained replay (~40ms) keeps clock at 2062 (matches ref);
                // NOTE power here is a partial rolling-avg (NVML power lags ~1s), so it under-reads.
                double tt0 = nowS();
                while (nowS() - tt0 < 0.040) {
                    for (int i = 0; i < 8; i++) checkCudaStatus(cudaGraphLaunch(exec, stream));
                    checkCudaStatus(cudaStreamSynchronize(stream));
                }
                double tt1 = nowS();
                unsigned smMhz = medianIn(tt0, tt1, [](std::pair<unsigned, unsigned> p) { return p.first; });
                unsigned pwW = medianIn(tt0, tt1, [](std::pair<unsigned, unsigned> p) { return p.second; });

                double secs = us * 1e-6;
                double tflops = 2.0 * m * n * k / secs / 1e12;
                double gbps = ((double)m * k * 0.5 + (double)n * k * 0.5 + (double)m * n * 2.0) / secs / 1e9;
                out << m << "," << n << "," << k << "," << us << "," << tflops << "," << gbps << ","
                    << smMhz << "," << pwW << ",cublaslt_nvfp4\n"; out.flush();
                printf("%6d%7d%7d | %9.2f %8.1f %9.1f | %6u %7u\n", m, n, k, us, tflops, gbps, smMhz, pwW);

                cudaEventDestroy(s); cudaEventDestroy(e);
                cudaGraphExecDestroy(exec); cudaGraphDestroy(graph);
            }
            if (pref) cublasLtMatmulPreferenceDestroy(pref);
            if (Dd) cublasLtMatrixLayoutDestroy(Dd);
            if (Cd) cublasLtMatrixLayoutDestroy(Cd);
            if (Bd) cublasLtMatrixLayoutDestroy(Bd);
            if (Ad) cublasLtMatrixLayoutDestroy(Ad);
            if (op) cublasLtMatmulDescDestroy(op);
        } catch (const std::exception &ex) {
            printf("%6d%7d%7d | ERROR %s\n", m, n, k, ex.what());
        }
    }
    g_stop = true; th.join(); nvmlShutdown();
    printf("\nDone -> %s\n", outFile);
    return 0;
}
