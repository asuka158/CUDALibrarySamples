/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cublasLt.h>

#include "helpers.h"
#include "sample_cublasLt_LtMxfp4Matmul.h"

/// Sample wrapper executing mxfp4 matmul with cublasLtMatmul, with per-block UE8M0 scaling for A and B, and
/// the workspace to support split-K algorithms. Output D is bf16, so no output (D) scaling is needed.
///
/// pointer mode for alpha and beta is always host, to change it configure the appropriate matmul descriptor
/// attribute. matmul is not using cublas handle's configuration of math mode, here tensor ops are implicitly
/// allowed; to change this configure the appropriate attribute in the preference handle.
void LtMxfp4Matmul(cublasLtHandle_t ltHandle,
                   cublasOperation_t transa,
                   cublasOperation_t transb,
                   int m,
                   int n,
                   int k,
                   const float *alpha,           /* host pointer */
                   const __nv_fp8_e8m0 *a_scale, /* device pointer */
                   const typename StorageType<__nv_fp4_e2m1>::type *A,
                   int lda,
                   const __nv_fp8_e8m0 *b_scale, /* device pointer */
                   const typename StorageType<__nv_fp4_e2m1>::type *B,
                   int ldb,
                   const float *beta, /* host pointer */
                   __nv_bfloat16 *C,
                   int ldc,
                   __nv_bfloat16 *D,
                   int ldd,
                   void *workspace,
                   size_t workspaceSize,
                   cublasLtMatmulMatrixScale_t AScaleMode,
                   cublasLtMatmulMatrixScale_t BScaleMode) {
    cublasLtMatmulDesc_t operationDesc = NULL;
    cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL, Ddesc = NULL;
    cublasLtMatmulPreference_t preference = NULL;

    int returnedResults = 0;
    cublasLtMatmulHeuristicResult_t heuristicResult = {};

    // create operation descriptor; see cublasLtMatmulDescAttributes_t for details about defaults; here we just need
    // to set the transforms for A and B
    checkCublasStatus(cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    checkCublasStatus(
        cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    checkCublasStatus(
        cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

    // set block scaling mode for A and B (mxfp4 uses VEC32_UE8M0, i.e. one UE8M0 scale per 32 fp4 elements)
    checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &AScaleMode,
                                                     sizeof(AScaleMode)));
    checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &BScaleMode,
                                                     sizeof(BScaleMode)));

    // set scaling factor pointers (device pointers to the UE8M0 block-scale tensors)
    checkCublasStatus(
        cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &a_scale, sizeof(a_scale)));
    checkCublasStatus(
        cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &b_scale, sizeof(b_scale)));

    // create matrix descriptors. A and B are FP4 (E2M1); C and D are bf16.
    // table of supported type combinations can be found in the documentation:
    // https://docs.nvidia.com/cuda/cublas/index.html#cublasltmatmul
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_4F_E2M1, transa == CUBLAS_OP_N ? m : k,
                                                 transa == CUBLAS_OP_N ? k : m, lda));
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_4F_E2M1, transb == CUBLAS_OP_N ? k : n,
                                                 transb == CUBLAS_OP_N ? n : k, ldb));
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_16BF, m, n, ldc));
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Ddesc, CUDA_R_16BF, m, n, ldd));

    // create preference handle; here we could use extra attributes to disable tensor ops or to make sure algo
    // selected will work with badly aligned A, B, C; here for simplicity we just assume A,B,C are always well
    // aligned (e.g. directly come from cudaMalloc)
    checkCublasStatus(cublasLtMatmulPreferenceCreate(&preference));
    checkCublasStatus(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                           &workspaceSize, sizeof(workspaceSize)));

    // we just need the best available heuristic to try and run matmul. There is no guarantee this will work, e.g. if
    // A is badly aligned, you can request more (e.g. 32) algos and try to run them one by one until something works
    checkCublasStatus(cublasLtMatmulAlgoGetHeuristic(ltHandle, operationDesc, Adesc, Bdesc, Cdesc, Ddesc, preference, 1,
                                                     &heuristicResult, &returnedResults));

    if (returnedResults == 0) {
        checkCublasStatus(CUBLAS_STATUS_NOT_SUPPORTED);
    }

    checkCublasStatus(cublasLtMatmul(ltHandle, operationDesc, alpha, A, Adesc, B, Bdesc, beta, C, Cdesc, D, Ddesc,
                                     &heuristicResult.algo, workspace, workspaceSize, 0));

    // descriptors are no longer needed as all GPU work was already enqueued
    if (preference) checkCublasStatus(cublasLtMatmulPreferenceDestroy(preference));
    if (Ddesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Ddesc));
    if (Cdesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Cdesc));
    if (Bdesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Bdesc));
    if (Adesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Adesc));
    if (operationDesc) checkCublasStatus(cublasLtMatmulDescDestroy(operationDesc));
}
