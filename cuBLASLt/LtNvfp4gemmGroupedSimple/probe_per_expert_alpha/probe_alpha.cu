// Probe: can cuBLASLt grouped NVFP4xNVFP4 -> bf16 gemm apply a PER-EXPERT (per-group)
// global scale? We need this for MoE w4a4: each expert has its own weight global scale
// (and per-token activation global scale). The basic sample uses ONE shared host alpha.
//
// Method: 2 groups, all A/B fp4 values = 1.0, all block scales = 1.0, K=32.
//   => raw (A^T@B) = K = 32 for every output element.
//   Apply per-group scale: group0 -> 2.0, group1 -> 0.5.
//   Expect D[group0] == 64.0 everywhere, D[group1] == 16.0 everywhere.
// We sweep several alpha mechanisms and report which one cuBLASLt accepts AND computes right.
//
// build:
//   export PATH=/usr/local/cuda/bin:$PATH
//   nvcc -O3 -std=c++17 -arch=sm_100a probe_alpha.cu -I../../Common -lcublasLt -lcudart -o probe_alpha
//   CUDA_VISIBLE_DEVICES=0 ./probe_alpha

#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cublasLt.h>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <string>

#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){printf("CUDA error %s @ %d: %s\n",#x,__LINE__,cudaGetErrorString(e));exit(1);} } while(0)
// non-aborting cublas check -> returns status
static const char* lts(cublasStatus_t s){
    switch(s){case CUBLAS_STATUS_SUCCESS:return "SUCCESS";case CUBLAS_STATUS_NOT_SUPPORTED:return "NOT_SUPPORTED";
    case CUBLAS_STATUS_INVALID_VALUE:return "INVALID_VALUE";case CUBLAS_STATUS_NOT_INITIALIZED:return "NOT_INITIALIZED";
    default:return "OTHER";}
}
#define CB(x) do { cublasStatus_t s=(x); if(s!=CUBLAS_STATUS_SUCCESS){printf("cublas %s @ %d = %s\n",#x,__LINE__,lts(s));} } while(0)

using fp4x2 = __nv_fp4x2_e2m1;
using e4m3  = __nv_fp8_e4m3;
using bf16  = __nv_bfloat16;

static size_t roundoff(size_t v, size_t m){ return (v + m - 1)/m*m; }
// VEC16_UE4M3 swizzled scale tensor size (matches Common/helpers.h getScaleTensorSize)
static size_t scaleSizeVEC16(int inner, int outer){
    const size_t V=16, BR=4*V /*=64*/, BC=32*4 /*=128*/;
    return roundoff(inner, BR)/V * roundoff(outer, BC);
}

int G = 2;                      // groups (experts)
int Marr[2] = {64, 32};         // per-group M (rows), multiple of 32 (fp4 16B align)
int N = 32, K = 32;             // fixed N,K, multiples of 32
float groupScale[2] = {2.0f, 0.5f};

// device buffers (per group)
e4m3   *Asc[2], *Bsc[2];        // block scales (all 1.0)
fp4x2  *Ad[2], *Bd[2];          // packed fp4 (all 1.0)
bf16   *Cd[2], *Dd[2];          // output
// device pointer arrays
void  **Ap, **Bp, **Cp, **Dp, **AscP, **BscP;
int    *mDev,*nDev,*kDev,*ldaDev,*ldbDev,*ldcDev,*lddDev;

__global__ void fill_fp4x2(fp4x2* p, size_t n){ size_t i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) p[i]=fp4x2(float2{1.0f,1.0f}); }
__global__ void fill_e4m3(e4m3* p, size_t n){ size_t i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) p[i]=e4m3(1.0f); }
__global__ void fill_e4m3v(e4m3* p, size_t n, float v){ size_t i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) p[i]=e4m3(v); }
__global__ void fill_bf16(bf16* p, size_t n, float v){ size_t i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) p[i]=__float2bfloat16(v); }

static void setupBuffers(){
    std::vector<void*> ap(G),bp(G),cp(G),dp(G),ascp(G),bscp(G);
    std::vector<int> mh(G),nh(G),kh(G),ldah(G),ldbh(G),ldch(G),lddh(G);
    for(int g=0; g<G; ++g){
        int M=Marr[g];
        size_t aF = (size_t)K*M, bF=(size_t)K*N;             // fp4 counts (transa=T: A is k x m; transb=N: B is k x n)
        CK(cudaMalloc(&Ad[g], aF/2*sizeof(fp4x2)));
        CK(cudaMalloc(&Bd[g], bF/2*sizeof(fp4x2)));
        CK(cudaMalloc(&Cd[g], (size_t)M*N*sizeof(bf16)));
        CK(cudaMalloc(&Dd[g], (size_t)M*N*sizeof(bf16)));
        size_t asz=scaleSizeVEC16(K,M), bsz=scaleSizeVEC16(K,N);   // A: inner=k outer=m ; B: inner=k outer=n
        CK(cudaMalloc(&Asc[g], asz*sizeof(e4m3)));
        CK(cudaMalloc(&Bsc[g], bsz*sizeof(e4m3)));
        fill_fp4x2<<<(aF/2+255)/256,256>>>(Ad[g], aF/2);
        fill_fp4x2<<<(bF/2+255)/256,256>>>(Bd[g], bF/2);
        fill_e4m3<<<(asz+255)/256,256>>>(Asc[g], asz);
        fill_e4m3<<<(bsz+255)/256,256>>>(Bsc[g], bsz);
        fill_bf16<<<((size_t)M*N+255)/256,256>>>(Cd[g],(size_t)M*N,0.0f);
        fill_bf16<<<((size_t)M*N+255)/256,256>>>(Dd[g],(size_t)M*N,0.0f);
        ap[g]=Ad[g]; bp[g]=Bd[g]; cp[g]=Cd[g]; dp[g]=Dd[g]; ascp[g]=Asc[g]; bscp[g]=Bsc[g];
        mh[g]=M; nh[g]=N; kh[g]=K; ldah[g]=K; ldbh[g]=K; ldch[g]=M; lddh[g]=M;
    }
    CK(cudaDeviceSynchronize());
    auto mkPtr=[&](void***d, std::vector<void*>&h){ CK(cudaMalloc(d, G*sizeof(void*))); CK(cudaMemcpy(*d,h.data(),G*sizeof(void*),cudaMemcpyHostToDevice)); };
    mkPtr(&Ap,ap); mkPtr(&Bp,bp); mkPtr(&Cp,cp); mkPtr(&Dp,dp); mkPtr(&AscP,ascp); mkPtr(&BscP,bscp);
    auto mkInt=[&](int**d, std::vector<int>&h){ CK(cudaMalloc(d, G*sizeof(int))); CK(cudaMemcpy(*d,h.data(),G*sizeof(int),cudaMemcpyHostToDevice)); };
    mkInt(&mDev,mh); mkInt(&nDev,nh); mkInt(&kDev,kh); mkInt(&ldaDev,ldah); mkInt(&ldbDev,ldbh); mkInt(&ldcDev,ldch); mkInt(&lddDev,lddh);
}

// returns true if matmul ran (heuristic found algo & matmul SUCCESS)
static bool runGrouped(cublasLtHandle_t h, void* ws, size_t wsSize,
                       cublasLtPointerMode_t pmode, const void* alphaArg, const void* betaArg,
                       int64_t alphaBatchStride, bool setPmode){
    cublasLtMatmulDesc_t op=nullptr; cublasLtMatrixLayout_t Ad_=nullptr,Bd_=nullptr,Cd_=nullptr,Dd_=nullptr; cublasLtMatmulPreference_t pref=nullptr;
    cublasOperation_t TA=CUBLAS_OP_T, TB=CUBLAS_OP_N;
    cublasLtMatmulMatrixScale_t sm=CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
    CB(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    CB(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &TA, sizeof(TA)));
    CB(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &TB, sizeof(TB)));
    CB(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &sm, sizeof(sm)));
    CB(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &sm, sizeof(sm)));
    CB(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &AscP, sizeof(AscP)));
    CB(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &BscP, sizeof(BscP)));
    if(setPmode){
        int pm=(int)pmode;
        CB(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_POINTER_MODE, &pm, sizeof(pm)));
        if(alphaBatchStride>=0)
            CB(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_ALPHA_VECTOR_BATCH_STRIDE, &alphaBatchStride, sizeof(alphaBatchStride)));
    }
    CB(cublasLtGroupedMatrixLayoutCreate(&Ad_, CUDA_R_4F_E2M1, G, kDev, mDev, ldaDev)); // transa=T: rows=k, cols=m
    CB(cublasLtGroupedMatrixLayoutCreate(&Bd_, CUDA_R_4F_E2M1, G, kDev, nDev, ldbDev)); // transb=N: rows=k, cols=n
    CB(cublasLtGroupedMatrixLayoutCreate(&Cd_, CUDA_R_16BF, G, mDev, nDev, ldcDev));
    CB(cublasLtGroupedMatrixLayoutCreate(&Dd_, CUDA_R_16BF, G, mDev, nDev, lddDev));
    CB(cublasLtMatmulPreferenceCreate(&pref));
    int64_t avgM=48, avgN=N, avgK=K;
    CB(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_GROUPED_DESC_D_AVERAGE_ROWS, &avgM, sizeof(avgM)));
    CB(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_GROUPED_DESC_D_AVERAGE_COLS, &avgN, sizeof(avgN)));
    CB(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_GROUPED_AVERAGE_REDUCTION_DIM, &avgK, sizeof(avgK)));
    CB(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsSize, sizeof(wsSize)));
    int got=0; cublasLtMatmulHeuristicResult_t hr{};
    cublasStatus_t hs = cublasLtMatmulAlgoGetHeuristic(h, op, Ad_, Bd_, Cd_, Dd_, pref, 1, &hr, &got);
    bool ok=false;
    if(hs==CUBLAS_STATUS_SUCCESS && got>0){
        cublasStatus_t ms = cublasLtMatmul(h, op, alphaArg, Ap, Ad_, Bp, Bd_, betaArg, Cp, Cd_, (void*)Dp, Dd_, &hr.algo, ws, wsSize, 0);
        if(ms==CUBLAS_STATUS_SUCCESS){ ok=true; }
        else printf("    matmul status = %s\n", lts(ms));
    } else {
        printf("    heuristic status = %s, got=%d\n", lts(hs), got);
    }
    if(pref)cublasLtMatmulPreferenceDestroy(pref);
    if(Dd_)cublasLtMatrixLayoutDestroy(Dd_); if(Cd_)cublasLtMatrixLayoutDestroy(Cd_);
    if(Bd_)cublasLtMatrixLayoutDestroy(Bd_); if(Ad_)cublasLtMatrixLayoutDestroy(Ad_);
    if(op)cublasLtMatmulDescDestroy(op);
    return ok;
}

static void checkResult(const char* label){
    CK(cudaDeviceSynchronize());
    for(int g=0; g<G; ++g){
        int M=Marr[g]; std::vector<bf16> hb((size_t)M*N);
        CK(cudaMemcpy(hb.data(), Dd[g], (size_t)M*N*sizeof(bf16), cudaMemcpyDeviceToHost));
        float want = groupScale[g]*K;
        float mn=1e30f,mx=-1e30f; double sum=0;
        for(auto v: hb){ float f=__bfloat162float(v); mn=f<mn?f:mn; mx=f>mx?f:mx; sum+=f; }
        printf("    [%s] group%d: want=%.1f  got min=%.3f max=%.3f mean=%.3f  -> %s\n",
               label, g, want, mn, mx, sum/hb.size(), (mn>want-0.5f&&mx<want+0.5f)?"PASS":"FAIL");
    }
}
static void zeroD(){ for(int g=0;g<G;++g) fill_bf16<<<((size_t)Marr[g]*N+255)/256,256>>>(Dd[g],(size_t)Marr[g]*N,0.0f); CK(cudaDeviceSynchronize()); }

int main(){
    cublasLtHandle_t h; CB(cublasLtCreate(&h));
    void* ws; size_t wsSize=32ull*1024*1024; CK(cudaMalloc(&ws, wsSize));
    setupBuffers();
    printf("Setup: %d groups, M=[%d,%d], N=%d, K=%d, groupScale=[%.1f,%.1f]; raw(A^T@B)=K=%d\n",
           G, Marr[0],Marr[1], N, K, groupScale[0],groupScale[1], K);
    printf("Expect per-group: group0=%.1f, group1=%.1f if per-expert scaling works.\n\n", groupScale[0]*K, groupScale[1]*K);

    float beta0=0.0f;

    // ---- baseline: single shared HOST alpha = 1.0 (sanity: both groups -> K=32) ----
    printf("[1] single HOST alpha=1.0 (baseline; both groups should be 32):\n");
    { float a=1.0f; zeroD(); float sv0=groupScale[0],sv1=groupScale[1]; groupScale[0]=1;groupScale[1]=1;
      if(runGrouped(h,ws,wsSize,CUBLASLT_POINTER_MODE_HOST,&a,&beta0,-1,false)) checkResult("HOST a=1");
      groupScale[0]=sv0;groupScale[1]=sv1; }

    // ---- [2] per-row DEVICE_VECTOR alpha, ONE vector of length sum(M)=96 ----
    // rows of group0 (first 64) = 2.0, group1 (next 32) = 0.5. Assumes concatenated row indexing.
    printf("\n[2] DEVICE_VECTOR alpha, single buffer length sum(M)=%d (per-output-row):\n", Marr[0]+Marr[1]);
    {
        int total=Marr[0]+Marr[1]; std::vector<float> ah(total);
        for(int i=0;i<Marr[0];++i) ah[i]=groupScale[0];
        for(int i=0;i<Marr[1];++i) ah[Marr[0]+i]=groupScale[1];
        float* aDev; CK(cudaMalloc(&aDev,total*sizeof(float))); CK(cudaMemcpy(aDev,ah.data(),total*sizeof(float),cudaMemcpyHostToDevice));
        zeroD();
        if(runGrouped(h,ws,wsSize,CUBLASLT_POINTER_MODE_DEVICE_VECTOR,aDev,&beta0,0,true)) checkResult("DEVVEC sumM");
        cudaFree(aDev);
    }

    // ---- [3] DEVICE_VECTOR alpha as array of G pointers (one scalar per group) ----
    printf("\n[3] DEVICE_VECTOR alpha, array of %d device pointers (one scalar per group):\n", G);
    {
        std::vector<void*> ph(G);
        for(int g=0;g<G;++g){ float* s; CK(cudaMalloc(&s,sizeof(float))); CK(cudaMemcpy(s,&groupScale[g],sizeof(float),cudaMemcpyHostToDevice)); ph[g]=s; }
        void** pdev; CK(cudaMalloc(&pdev,G*sizeof(void*))); CK(cudaMemcpy(pdev,ph.data(),G*sizeof(void*),cudaMemcpyHostToDevice));
        zeroD();
        if(runGrouped(h,ws,wsSize,CUBLASLT_POINTER_MODE_DEVICE_VECTOR,pdev,&beta0,0,true)) checkResult("DEVVEC ptrarr");
        cudaFree(pdev);
    }

    // ---- [4] ALPHA_DEVICE_VECTOR_BETA_ZERO, single buffer length sum(M) ----
    printf("\n[4] ALPHA_DEVICE_VECTOR_BETA_ZERO, single buffer length sum(M):\n");
    {
        int total=Marr[0]+Marr[1]; std::vector<float> ah(total);
        for(int i=0;i<Marr[0];++i) ah[i]=groupScale[0];
        for(int i=0;i<Marr[1];++i) ah[Marr[0]+i]=groupScale[1];
        float* aDev; CK(cudaMalloc(&aDev,total*sizeof(float))); CK(cudaMemcpy(aDev,ah.data(),total*sizeof(float),cudaMemcpyHostToDevice));
        zeroD();
        if(runGrouped(h,ws,wsSize,CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_ZERO,aDev,nullptr,0,true)) checkResult("ADVBZ sumM");
        cudaFree(aDev);
    }

    // ---- [5] THE FALLBACK: fold per-expert global into the (per-group) A block scales ----
    // alpha=1 (shared host). Set group0 A-blockscale=2.0, group1 A-blockscale=0.5, B stays 1.0.
    // Expect group0 -> 2.0*K=64, group1 -> 0.5*K=16. Proves per-expert scaling via block scales.
    printf("\n[5] per-expert scale folded into per-group A block scales (alpha=1 shared):\n");
    {
        size_t a0=scaleSizeVEC16(K,Marr[0]), a1=scaleSizeVEC16(K,Marr[1]);
        fill_e4m3v<<<(a0+255)/256,256>>>(Asc[0], a0, 2.0f);
        fill_e4m3v<<<(a1+255)/256,256>>>(Asc[1], a1, 0.5f);
        CK(cudaDeviceSynchronize());
        float a=1.0f; zeroD();
        if(runGrouped(h,ws,wsSize,CUBLASLT_POINTER_MODE_HOST,&a,&beta0,-1,false)) checkResult("blockscale");
        // restore
        fill_e4m3<<<(a0+255)/256,256>>>(Asc[0], a0);
        fill_e4m3<<<(a1+255)/256,256>>>(Asc[1], a1);
        CK(cudaDeviceSynchronize());
    }

    cudaFree(ws); cublasLtDestroy(h);
    return 0;
}
