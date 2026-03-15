#include "my_gemm.h"

#include <thread>
#include <vector>

using namespace std;

int g_my_threads = 1;

void my_set_num_threads(int t) {
    if (t < 1) t = 1;
    g_my_threads = t;
}

int my_get_num_threads() {
    return g_my_threads;
}

int is_trans(enum CBLAS_TRANSPOSE t) {
    return (t == CblasTrans || t == CblasConjTrans);
}

int is_conj(enum CBLAS_TRANSPOSE t) {
    return (t == CblasConjTrans);
}

bool check_params(enum CBLAS_ORDER Order, enum CBLAS_TRANSPOSE TransA, enum CBLAS_TRANSPOSE TransB,
                  int M, int N, int K, int lda, int ldb, int ldc) {
    if (Order != CblasRowMajor) return false;

    if (!(TransA == CblasNoTrans || TransA == CblasTrans || TransA == CblasConjTrans)) return false;
    if (!(TransB == CblasNoTrans || TransB == CblasTrans || TransB == CblasConjTrans)) return false;

    if (M < 0 || N < 0 || K < 0) return false;
    if (M == 0 || N == 0) return true;

    if (K > 0) {
        if (!is_trans(TransA)) {
            if (lda != K) return false;
        } else {
            if (lda != M) return false;
        }

        if (!is_trans(TransB)) {
            if (ldb != N) return false;
        } else {
            if (ldb != K) return false;
        }
    }

    if (ldc != N) return false;
    return true;
}

float getA_f(const float* A, int lda, int i, int p, enum CBLAS_TRANSPOSE TransA) {
    if (!is_trans(TransA)) return A[i * lda + p];
    return A[p * lda + i];
}

float getB_f(const float* B, int ldb, int p, int j, enum CBLAS_TRANSPOSE TransB) {
    if (!is_trans(TransB)) return B[p * ldb + j];
    return B[j * ldb + p];
}

double getA_d(const double* A, int lda, int i, int p, enum CBLAS_TRANSPOSE TransA) {
    if (!is_trans(TransA)) return A[i * lda + p];
    return A[p * lda + i];
}

double getB_d(const double* B, int ldb, int p, int j, enum CBLAS_TRANSPOSE TransB) {
    if (!is_trans(TransB)) return B[p * ldb + j];
    return B[j * ldb + p];
}

openblas_complex_float cf_make(float re, float im) {
    openblas_complex_float z;
    z.real = re;
    z.imag = im;
    return z;
}

openblas_complex_double cd_make(double re, double im) {
    openblas_complex_double z;
    z.real = re;
    z.imag = im;
    return z;
}

openblas_complex_float cf_add(openblas_complex_float a, openblas_complex_float b) {
    return cf_make(a.real + b.real, a.imag + b.imag);
}

openblas_complex_double cd_add(openblas_complex_double a, openblas_complex_double b) {
    return cd_make(a.real + b.real, a.imag + b.imag);
}

openblas_complex_float cf_mul(openblas_complex_float a, openblas_complex_float b) {
    return cf_make(a.real * b.real - a.imag * b.imag,
                   a.real * b.imag + a.imag * b.real);
}

openblas_complex_double cd_mul(openblas_complex_double a, openblas_complex_double b) {
    return cd_make(a.real * b.real - a.imag * b.imag,
                   a.real * b.imag + a.imag * b.real);
}

openblas_complex_float cf_conj(openblas_complex_float a) {
    return cf_make(a.real, -a.imag);
}

openblas_complex_double cd_conj(openblas_complex_double a) {
    return cd_make(a.real, -a.imag);
}

openblas_complex_float getA_c(const openblas_complex_float* A, int lda, int i, int p, enum CBLAS_TRANSPOSE TransA) {
    if (!is_trans(TransA)) return A[i * lda + p];
    openblas_complex_float v = A[p * lda + i];
    if (is_conj(TransA)) v = cf_conj(v);
    return v;
}

openblas_complex_float getB_c(const openblas_complex_float* B, int ldb, int p, int j, enum CBLAS_TRANSPOSE TransB) {
    if (!is_trans(TransB)) return B[p * ldb + j];
    openblas_complex_float v = B[j * ldb + p];
    if (is_conj(TransB)) v = cf_conj(v);
    return v;
}

openblas_complex_double getA_z(const openblas_complex_double* A, int lda, int i, int p, enum CBLAS_TRANSPOSE TransA) {
    if (!is_trans(TransA)) return A[i * lda + p];
    openblas_complex_double v = A[p * lda + i];
    if (is_conj(TransA)) v = cd_conj(v);
    return v;
}

openblas_complex_double getB_z(const openblas_complex_double* B, int ldb, int p, int j, enum CBLAS_TRANSPOSE TransB) {
    if (!is_trans(TransB)) return B[p * ldb + j];
    openblas_complex_double v = B[j * ldb + p];
    if (is_conj(TransB)) v = cd_conj(v);
    return v;
}

void sgemm_range(int i0, int i1,
                 enum CBLAS_TRANSPOSE TransA, enum CBLAS_TRANSPOSE TransB,
                 int M, int N, int K, float alpha, const float* A, int lda,
                 const float* B, int ldb, float beta, float* C, int ldc) {
    for (int i = i0; i < i1; i++) {
        for (int j = 0; j < N; j++) C[i * ldc + j] = beta * C[i * ldc + j];
        if (K == 0) continue;
        for (int p = 0; p < K; p++) {
            float a = alpha * getA_f(A, lda, i, p, TransA);
            for (int j = 0; j < N; j++) {
                C[i * ldc + j] += a * getB_f(B, ldb, p, j, TransB);
            }
        }
    }
}

void dgemm_range(int i0, int i1,
                 enum CBLAS_TRANSPOSE TransA, enum CBLAS_TRANSPOSE TransB,
                 int M, int N, int K, double alpha, const double* A, int lda,
                 const double* B, int ldb, double beta, double* C, int ldc) {
    for (int i = i0; i < i1; i++) {
        for (int j = 0; j < N; j++) C[i * ldc + j] = beta * C[i * ldc + j];
        if (K == 0) continue;
        for (int p = 0; p < K; p++) {
            double a = alpha * getA_d(A, lda, i, p, TransA);
            for (int j = 0; j < N; j++) {
                C[i * ldc + j] += a * getB_d(B, ldb, p, j, TransB);
            }
        }
    }
}

void cgemm_range(int i0, int i1,
                 enum CBLAS_TRANSPOSE TransA, enum CBLAS_TRANSPOSE TransB,
                 int M, int N, int K, openblas_complex_float alpha, const openblas_complex_float* A, int lda,
                 const openblas_complex_float* B, int ldb, openblas_complex_float beta, openblas_complex_float* C, int ldc) {
    for (int i = i0; i < i1; i++) {
        for (int j = 0; j < N; j++) C[i * ldc + j] = cf_mul(beta, C[i * ldc + j]);
        if (K == 0) continue;
        for (int p = 0; p < K; p++) {
            openblas_complex_float av = getA_c(A, lda, i, p, TransA);
            openblas_complex_float aav = cf_mul(alpha, av);
            for (int j = 0; j < N; j++) {
                openblas_complex_float bv = getB_c(B, ldb, p, j, TransB);
                C[i * ldc + j] = cf_add(C[i * ldc + j], cf_mul(aav, bv));
            }
        }
    }
}

void zgemm_range(int i0, int i1,
                 enum CBLAS_TRANSPOSE TransA, enum CBLAS_TRANSPOSE TransB,
                 int M, int N, int K, openblas_complex_double alpha, const openblas_complex_double* A, int lda,
                 const openblas_complex_double* B, int ldb, openblas_complex_double beta, openblas_complex_double* C, int ldc) {
    for (int i = i0; i < i1; i++) {
        for (int j = 0; j < N; j++) C[i * ldc + j] = cd_mul(beta, C[i * ldc + j]);
        if (K == 0) continue;
        for (int p = 0; p < K; p++) {
            openblas_complex_double av = getA_z(A, lda, i, p, TransA);
            openblas_complex_double aav = cd_mul(alpha, av);
            for (int j = 0; j < N; j++) {
                openblas_complex_double bv = getB_z(B, ldb, p, j, TransB);
                C[i * ldc + j] = cd_add(C[i * ldc + j], cd_mul(aav, bv));
            }
        }
    }
}

void run_threads(int M, int threads, vector<thread>& th, vector<int>& cuts) {
    if (threads < 1) threads = 1;
    if (threads > M) threads = M;

    cuts.clear();
    cuts.reserve(threads + 1);
    cuts.push_back(0);

    int base = M / threads;
    int rem = M % threads;
    int cur = 0;

    for (int i = 0; i < threads; i++) {
        cur += base + (i < rem ? 1 : 0);
        cuts.push_back(cur);
    }
}

bool my_sgemm(enum CBLAS_ORDER Order, enum CBLAS_TRANSPOSE TransA, enum CBLAS_TRANSPOSE TransB,
              int M, int N, int K, float alpha, const float *A, int lda,
              const float *B, int ldb, float beta, float *C, int ldc) {
    if (!check_params(Order, TransA, TransB, M, N, K, lda, ldb, ldc)) return false;
    if (M == 0 || N == 0) return true;

    int t = g_my_threads;
    if (t < 1) t = 1;
    if (t > M) t = M;

    if (t == 1) {
        sgemm_range(0, M, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        return true;
    }

    vector<thread> th;
    vector<int> cuts;
    run_threads(M, t, th, cuts);

    for (int i = 0; i < t; i++) {
        int i0 = cuts[i];
        int i1 = cuts[i + 1];
        th.emplace_back(sgemm_range, i0, i1, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    }

    for (auto& x : th) x.join();
    return true;
}

bool my_dgemm(enum CBLAS_ORDER Order, enum CBLAS_TRANSPOSE TransA, enum CBLAS_TRANSPOSE TransB,
              int M, int N, int K, double alpha, const double *A, int lda,
              const double *B, int ldb, double beta, double *C, int ldc) {
    if (!check_params(Order, TransA, TransB, M, N, K, lda, ldb, ldc)) return false;
    if (M == 0 || N == 0) return true;

    int t = g_my_threads;
    if (t < 1) t = 1;
    if (t > M) t = M;

    if (t == 1) {
        dgemm_range(0, M, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        return true;
    }

    vector<thread> th;
    vector<int> cuts;
    run_threads(M, t, th, cuts);

    for (int i = 0; i < t; i++) {
        int i0 = cuts[i];
        int i1 = cuts[i + 1];
        th.emplace_back(dgemm_range, i0, i1, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    }

    for (auto& x : th) x.join();
    return true;
}

bool my_cgemm(enum CBLAS_ORDER Order, enum CBLAS_TRANSPOSE TransA, enum CBLAS_TRANSPOSE TransB,
              int M, int N, int K, const void *alpha, const void *A, int lda,
              const void *B, int ldb, const void *beta, void *C, int ldc) {
    if (!check_params(Order, TransA, TransB, M, N, K, lda, ldb, ldc)) return false;
    if (M == 0 || N == 0) return true;

    openblas_complex_float a = *(const openblas_complex_float*)alpha;
    openblas_complex_float b = *(const openblas_complex_float*)beta;

    const openblas_complex_float* Ap = (const openblas_complex_float*)A;
    const openblas_complex_float* Bp = (const openblas_complex_float*)B;
    openblas_complex_float* Cp = (openblas_complex_float*)C;

    int t = g_my_threads;
    if (t < 1) t = 1;
    if (t > M) t = M;

    if (t == 1) {
        cgemm_range(0, M, TransA, TransB, M, N, K, a, Ap, lda, Bp, ldb, b, Cp, ldc);
        return true;
    }

    vector<thread> th;
    vector<int> cuts;
    run_threads(M, t, th, cuts);

    for (int i = 0; i < t; i++) {
        int i0 = cuts[i];
        int i1 = cuts[i + 1];
        th.emplace_back(cgemm_range, i0, i1, TransA, TransB, M, N, K, a, Ap, lda, Bp, ldb, b, Cp, ldc);
    }

    for (auto& x : th) x.join();
    return true;
}

bool my_zgemm(enum CBLAS_ORDER Order, enum CBLAS_TRANSPOSE TransA, enum CBLAS_TRANSPOSE TransB,
              int M, int N, int K, const void *alpha, const void *A, int lda,
              const void *B, int ldb, const void *beta, void *C, int ldc) {
    if (!check_params(Order, TransA, TransB, M, N, K, lda, ldb, ldc)) return false;
    if (M == 0 || N == 0) return true;

    openblas_complex_double a = *(const openblas_complex_double*)alpha;
    openblas_complex_double b = *(const openblas_complex_double*)beta;

    const openblas_complex_double* Ap = (const openblas_complex_double*)A;
    const openblas_complex_double* Bp = (const openblas_complex_double*)B;
    openblas_complex_double* Cp = (openblas_complex_double*)C;

    int t = g_my_threads;
    if (t < 1) t = 1;
    if (t > M) t = M;

    if (t == 1) {
        zgemm_range(0, M, TransA, TransB, M, N, K, a, Ap, lda, Bp, ldb, b, Cp, ldc);
        return true;
    }

    vector<thread> th;
    vector<int> cuts;
    run_threads(M, t, th, cuts);

    for (int i = 0; i < t; i++) {
        int i0 = cuts[i];
        int i1 = cuts[i + 1];
        th.emplace_back(zgemm_range, i0, i1, TransA, TransB, M, N, K, a, Ap, lda, Bp, ldb, b, Cp, ldc);
    }

    for (auto& x : th) x.join();
    return true;
}