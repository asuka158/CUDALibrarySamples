// Reproducer: cuBLASLt has no NVFP4 x NVFP4 -> FP8 path (see README.md).
//
// Sweeps cublasLtMatmulAlgoGetHeuristic for A,B = NVFP4 (CUDA_R_4F_E2M1, VEC16 UE4M3 block scales)
// with D = FP8 e4m3 across every C type / D-scale / D-out-scale / amax combination, and prints a
// control (NVFP4 -> bf16, which IS supported). Every fp8-output combo returns status 15
// (CUBLAS_STATUS_NOT_SUPPORTED) on cuBLASLt 13.4 / GB200 (sm_100). Re-run on future cuBLAS to recheck.
//
// Build (from this dir):
//   export PATH=/usr/local/cuda/bin:$PATH
//   nvcc -O3 -std=c++17 -arch=sm_100a check_unsupported.cu -I../../Common -lcublasLt -o check_unsupported
//   ./check_unsupported
#include <cublasLt.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cstdio>
#include "helpers.h"

using TBfp8 = TestBench<__nv_fp4_e2m1, __nv_fp8_e4m3, float, __nv_fp8_e4m3, float, __nv_bfloat16>;
using TBbf  = TestBench<__nv_fp4_e2m1, __nv_bfloat16, float, __nv_fp8_e4m3, float, __nv_bfloat16>;
// helpers.h lacks a fillData() specialization for nvfp4->fp8; provide a compile-safe one.
template <> inline void TBfp8::fillData() {
    for (size_t i = 0; i < Ahost.size(); i++) Ahost[i] = __nv_fp4x2_e2m1{float2{1.f, 2.f}};
    for (size_t i = 0; i < Bhost.size(); i++) Bhost[i] = __nv_fp4x2_e2m1{float2{1.f, 2.f}};
}

static const auto MA   = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
static const auto SC   = CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
static const auto V32  = CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
static const auto V128 = CUBLASLT_MATMUL_MATRIX_SCALE_VEC128_32F;
static const auto B128 = CUBLASLT_MATMUL_MATRIX_SCALE_BLK128x128_32F;
static const auto OUTV = CUBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F;
static const auto NONE = (cublasLtMatmulMatrixScale_t)(-1);

template <typename TBT>
static int probe(cudaDataType Dtype, cudaDataType Ctype,
                 cublasLtMatmulMatrixScale_t dScaleMode, cublasLtMatmulMatrixScale_t dOutMode, bool amax) {
    int m = 256, n = 512, k = 512; size_t WS = 32ULL * 1024 * 1024; float alpha = 1, beta = 0;
    auto ctorDOut = (dOutMode == NONE) ? MA : dOutMode;
    auto ctorDS   = (dScaleMode == NONE) ? SC : dScaleMode;
    TBT props(CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, alpha, beta, WS, 1, MA, MA, SC, ctorDS, ctorDOut);
    props.copyDataToDevice(); props.streamSynchronize();
    cublasLtMatmulDesc_t op = nullptr; cublasLtMatrixLayout_t Ad = nullptr, Bd = nullptr, Cd = nullptr, Dd = nullptr;
    cublasLtMatmulPreference_t pref = nullptr; cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
    cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta));
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb));
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &props.AScaleMode, sizeof(MA));
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &props.BScaleMode, sizeof(MA));
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &props.AscaleDev, sizeof(void*));
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &props.BscaleDev, sizeof(void*));
    if (dScaleMode != NONE) {
        cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_D_SCALE_MODE, &dScaleMode, sizeof(dScaleMode));
        cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_D_SCALE_POINTER, &props.DscaleDev, sizeof(void*));
    }
    if (dOutMode != NONE) {
        cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_D_OUT_SCALE_MODE, &dOutMode, sizeof(dOutMode));
        cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_D_OUT_SCALE_POINTER, &props.DOutscaleDev, sizeof(void*));
    }
    if (amax) cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_AMAX_D_POINTER, &props.DamaxDev, sizeof(void*));
    cublasLtMatrixLayoutCreate(&Ad, CUDA_R_4F_E2M1, k, m, props.lda);
    cublasLtMatrixLayoutCreate(&Bd, CUDA_R_4F_E2M1, k, n, props.ldb);
    cublasLtMatrixLayoutCreate(&Cd, Ctype, m, n, props.ldc);
    cublasLtMatrixLayoutCreate(&Dd, Dtype, m, n, props.ldd);
    cublasLtMatmulPreferenceCreate(&pref);
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &WS, sizeof(WS));
    cublasLtMatmulHeuristicResult_t heur[4] = {}; int got = 0;
    cublasLtMatmulAlgoGetHeuristic(props.ltHandle, op, Ad, Bd, Cd, Dd, pref, 4, heur, &got);
    cublasLtMatmulPreferenceDestroy(pref); cublasLtMatrixLayoutDestroy(Dd); cublasLtMatrixLayoutDestroy(Cd);
    cublasLtMatrixLayoutDestroy(Bd); cublasLtMatrixLayoutDestroy(Ad); cublasLtMatmulDescDestroy(op);
    return got;
}

int main() {
    printf("cuBLASLt version %zu\n", cublasLtGetVersion());
    printf("CONTROL  nvfp4 -> bf16 : got=%d (expect >=1)\n",
           probe<TBbf>(CUDA_R_16BF, CUDA_R_16BF, NONE, NONE, false));

    cudaDataType Cs[4] = {CUDA_R_16BF, CUDA_R_32F, CUDA_R_16F, CUDA_R_8F_E4M3};
    cublasLtMatmulMatrixScale_t dS[3] = {NONE, SC, V128};
    cublasLtMatmulMatrixScale_t dO[7] = {NONE, SC, MA, V32, V128, B128, OUTV};
    int tried = 0, ok = 0;
    for (auto C : Cs) for (auto s : dS) for (auto o : dO) for (int a = 0; a < 2; a++) {
        tried++; ok += probe<TBfp8>(CUDA_R_8F_E4M3, C, s, o, a) > 0 ? 1 : 0;
    }
    printf("SWEEP    nvfp4 -> fp8_e4m3 : %d/%d configurations supported\n", ok, tried);
    printf(ok == 0 ? "=> NVFP4 x NVFP4 -> FP8 is NOT supported by this cuBLASLt.\n"
                   : "=> Some fp8 config now works -- update the benchmark!\n");
    return 0;
}
