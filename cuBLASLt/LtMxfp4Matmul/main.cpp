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

#include "sample_cublasLt_LtMxfp4Matmul.h"
#include "helpers.h"

int main() {
    // mxfp4 x mxfp4 -> bf16 matmul.
    //   InTypeAB    = __nv_fp4_e2m1   (packed FP4, two values per byte)
    //   OutType     = __nv_bfloat16   (D output, no output quantization)
    //   ComputeType = float
    //   ScaleType   = __nv_fp8_e8m0   (UE8M0 block scales, one per 32 fp4 elements)
    //   DScaleType  = float           (unused: bf16 D needs no scale)
    //   InTypeC     = __nv_bfloat16
    // A and B use CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0 (the OCP MXFP4 microscaling layout). C and D are bf16,
    // so their scale modes are SCALAR_32F (effectively no scaling).
    TestBench<__nv_fp4_e2m1, __nv_bfloat16, float, __nv_fp8_e8m0, float, __nv_bfloat16> props(
        CUBLAS_OP_T, CUBLAS_OP_N, 64, 128, 256, 2.0f, 1.0f, 32ULL * 1024 * 1024, 1,
        CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0, CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0,
        CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F, CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F,
        CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F);

    props.run([&props] {
        LtMxfp4Matmul(props.ltHandle, props.transa, props.transb, props.m, props.n, props.k, &props.alpha,
                      props.AscaleDev, props.Adev, props.lda, props.BscaleDev, props.Bdev, props.ldb, &props.beta,
                      props.Cdev, props.ldc, props.Ddev, props.ldd, props.workspace, props.workspaceSize,
                      props.AScaleMode, props.BScaleMode);
    });

    printf("mxfp4 x mxfp4 -> bf16 matmul (m=%d n=%d k=%d) completed successfully.\n", props.m, props.n, props.k);

    return 0;
}
