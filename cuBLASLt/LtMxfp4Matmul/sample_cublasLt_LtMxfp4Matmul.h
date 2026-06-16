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

#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cublasLt.h>

#include "helpers.h"

/// Sample wrapper executing mxfp4 matmul with cublasLtMatmul, and the workspace to support split-K algorithms.
///
/// MXFP4 (OCP microscaling FP4) = FP4 (E2M1) data + UE8M0 block scales with a block size of 32 elements
/// (CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0). This differs from NVFP4, which uses UE4M3 scales with a block
/// size of 16. Here both operands A and B are mxfp4 and the output D is bf16 (no output quantization / D scale),
/// which makes it directly comparable to a typical inference GEMM and to DeepGEMM's mxfp4 x mxfp4 kernel.
///
/// pointer mode for alpha and beta is always host; to change it configure the appropriate matmul descriptor
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
                   cublasLtMatmulMatrixScale_t BScaleMode);

// fillData specialization for the mxfp4 x mxfp4 -> bf16 type combination used by this sample.
// It must be visible in every translation unit that instantiates TestBench<...> (i.e. main.cpp),
// so it lives in this shared header (after helpers.h defines the primary TestBench template).
// A and B are packed FP4 (two e2m1 values per byte); C and D are bf16.
template <>
inline void
TestBench<__nv_fp4_e2m1, __nv_bfloat16, float, __nv_fp8_e8m0, float, __nv_bfloat16>::fillData() {
    for (size_t i = 0; i < Ahost.size(); i++) Ahost[i] = __nv_fp4x2_e2m1{float2{float(i % 5), float(i % 5) + 1}};
    for (size_t i = 0; i < Bhost.size(); i++) Bhost[i] = __nv_fp4x2_e2m1{float2{float(i % 5), float(i % 5) + 1}};
    for (size_t i = 0; i < Chost.size(); i++) Chost[i] = __nv_bfloat16(i % 5);
    for (size_t i = 0; i < biasHost.size(); i++) biasHost[i] = __nv_bfloat16(float(i % 5) + 1);
}
