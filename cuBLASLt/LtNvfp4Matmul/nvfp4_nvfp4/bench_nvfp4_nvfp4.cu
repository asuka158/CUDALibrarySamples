// cuBLASLt NVFP4 x NVFP4 -> NVFP4 (e2m1) dense GEMM, graph-sustained steady-state benchmark.
//
//   D = alpha * (A @ B),  A,B = NVFP4 (CUDA_R_4F_E2M1) with VEC16 UE4M3 block scales,
//   D = NVFP4 with a scalar (32F) D-scale AND a VEC16 UE4M3 block out-scale (kernel-computed),
//   C = bf16 (unused, beta=0), compute = 32F.  (Same recipe as the LtNvfp4Matmul sample main.cpp.)
//   A,B filled with N(0,1)-quantized NVFP4 (close to torch.randn), NOT TestBench's float(i%5).
//   Per shape: capture G matmuls in a CUDA graph, replay ~3s back-to-back, record steady-state tail.
//
// Build (from this dir):
//   export PATH=/usr/local/cuda/bin:$PATH
//   nvcc -O3 -std=c++17 -arch=sm_100a bench_nvfp4_nvfp4.cu -I../../Common -I.. \
//        -lcublasLt -lnvidia-ml -lcurand -o bench_nvfp4_nvfp4
// Run:
//   ./bench_nvfp4_nvfp4 [shapes.txt] [out.csv]
//   default shapes: ../../../../DeepGEMM/tests/my_test/shape.txt ; default out: result/nvfp4_nvfp4.csv
#include <cublasLt.h>

#include "helpers.h"
#include "nvfp4_sustained_bench.cuh"

using TB = TestBench<__nv_fp4_e2m1, __nv_fp4_e2m1, float, __nv_fp8_e4m3, float, __nv_bfloat16>;

static const char *DEF_SHAPES = "../../../../DeepGEMM/tests/my_test/shape.txt";
static const char *DEF_OUT    = "result/nvfp4_nvfp4.csv";
static const char *BACKEND    = "cublaslt_nvfp4_nvfp4";

int main(int argc, char **argv) {
    std::string shapesPath = argc > 1 ? argv[1] : DEF_SHAPES;
    std::string outPath    = argc > 2 ? argv[2] : DEF_OUT;
    float randnScale = 2.5f;  // N(0,1)*scale -> fp4 (spreads codes over e2m1 range, ~block-scaled randn)
    if (const char *e = getenv("NVFP4_RANDN_SCALE")) randnScale = atof(e);

    auto shapes = readShapes(shapesPath);
    const auto MA = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;  // A,B and D-out block scales
    const auto SC = CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;   // C,D scalar scales
    const size_t WS = 32ULL * 1024 * 1024;
    const float alpha = 1.0f, beta = 0.0f;

    nvmlInit();
    nvmlDeviceGetHandleByIndex(0, &g_dev);
    std::thread th(samplerLoop);

    char devName[256] = {0}; { cudaDeviceProp p; cudaGetDeviceProperties(&p, 0); snprintf(devName, sizeof(devName), "%s", p.name); }
    printf("Device: %s\n", devName);
    printf("Backend: %s | A,B=nvfp4 (VEC16 UE4M3) | D=nvfp4 (scalar D-scale + VEC16 UE4M3 out-scale) | compute=32F\n", BACKEND);
    printf("Graph-sustained: replay ~%.0fs back-to-back, steady-state tail (last %.0fs) | randn_scale=%.2f\n",
           SUSTAIN_S, TAIL_S, randnScale);
    printf("Benchmarking %zu shapes\n\n", shapes.size());
    printf("%6s %6s %6s | %9s %8s %8s | %5s %6s\n", "m", "n", "k", "us", "TFLOPS", "GB/s", "sm", "pw");

    std::ofstream out(outPath);
    if (!out) { printf("cannot open out %s\n", outPath.c_str()); return 1; }
    out << "m,n,k,us,tflops,gbps,sm_mhz,power_w,backend\n";

    for (auto &shp : shapes) {
        int m = shp[0], n = shp[1], k = shp[2];

        TB props(CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, alpha, beta, WS, 1, MA, MA, SC, SC, MA);
        props.copyDataToDevice();
        props.streamSynchronize();
        cudaStream_t stream = props.stream;
        cublasLtHandle_t lt = props.ltHandle;

        // overwrite A,B with N(0,1)-quantized NVFP4 (different seeds so A != B)
        fillRandnFp4(props.Adev, (size_t)m * k, 0x1234ull + (unsigned)m, randnScale, stream);
        fillRandnFp4(props.Bdev, (size_t)n * k, 0xABCDull + (unsigned)n, randnScale, stream);
        checkCudaStatus(cudaStreamSynchronize(stream));

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
        // nvfp4 output: scalar D-scale + per-block (VEC16 UE4M3) out-scale that the kernel computes
        checkCublasStatus(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_D_SCALE_MODE, &props.DScaleMode, sizeof(SC)));
        checkCublasStatus(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_D_SCALE_POINTER, &props.DscaleDev, sizeof(void *)));
        checkCublasStatus(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_D_OUT_SCALE_MODE, &props.DOutScaleMode, sizeof(MA)));
        checkCublasStatus(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_D_OUT_SCALE_POINTER, &props.DOutscaleDev, sizeof(void *)));

        checkCublasStatus(cublasLtMatrixLayoutCreate(&Ad, CUDA_R_4F_E2M1, k, m, props.lda));
        checkCublasStatus(cublasLtMatrixLayoutCreate(&Bd, CUDA_R_4F_E2M1, k, n, props.ldb));
        checkCublasStatus(cublasLtMatrixLayoutCreate(&Cd, CUDA_R_16BF, m, n, props.ldc));
        checkCublasStatus(cublasLtMatrixLayoutCreate(&Dd, CUDA_R_4F_E2M1, m, n, props.ldd));
        checkCublasStatus(cublasLtMatmulPreferenceCreate(&pref));
        checkCublasStatus(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &WS, sizeof(WS)));
        int got = 0;
        cublasLtMatmulHeuristicResult_t heur = {};
        checkCublasStatus(cublasLtMatmulAlgoGetHeuristic(lt, op, Ad, Bd, Cd, Dd, pref, 1, &heur, &got));
        if (got == 0) { printf("%6d %6d %6d | no algo\n", m, n, k); }
        else {
            auto matmul = [&](int) {
                checkCublasStatus(cublasLtMatmul(lt, op, &alpha, props.Adev, Ad, props.Bdev, Bd, &beta,
                                                 props.Cdev, Cd, props.Ddev, Dd, &heur.algo,
                                                 props.workspace, WS, stream));
            };
            SustainResult r = runSustained(matmul, m, n, k, stream);
            // bytes: fp4 A + fp4 B + fp4 D (0.5 byte), scales negligible (match python gbps formula)
            double gbps = ((double)m * k * 0.5 + (double)n * k * 0.5 + (double)m * n * 0.5) / (r.us / 1e6) / 1e9;
            out << m << "," << n << "," << k << "," << r.us << "," << r.tflops << "," << gbps << ","
                << r.sm_mhz << "," << r.power_w << "," << BACKEND << "\n";
            out.flush();
            printf("%6d %6d %6d | %9.2f %8.1f %8.1f | %5u %6u\n", m, n, k, r.us, r.tflops, gbps, r.sm_mhz, r.power_w);
        }

        cublasLtMatmulPreferenceDestroy(pref);
        cublasLtMatrixLayoutDestroy(Dd); cublasLtMatrixLayoutDestroy(Cd);
        cublasLtMatrixLayoutDestroy(Bd); cublasLtMatrixLayoutDestroy(Ad);
        cublasLtMatmulDescDestroy(op);
    }
    out.close();
    g_stop = true; th.join(); nvmlShutdown();
    printf("\nDone. %zu shapes -> %s\n", shapes.size(), outPath.c_str());
    return 0;
}
