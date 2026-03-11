#include "my_gemm.h"

using namespace std;

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

bool my_sgemm(enum CBLAS_ORDER Order, enum CBLAS_TRANSPOSE TransA, enum CBLAS_TRANSPOSE TransB,
              int M, int N, int K, float alpha, const float *A, int lda,
              const float *B, int ldb, float beta, float *C, int ldc) {
    if (!check_params(Order, TransA, TransB, M, N, K, lda, ldb, ldc)) return false;
    if (M == 0 || N == 0) return true;

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C[i * ldc + j] = beta * C[i * ldc + j];
        }
    }

    if (K == 0) return true;

    for (int i = 0; i < M; i++) {
        for (int p = 0; p < K; p++) {
            float a = alpha * getA_f(A, lda, i, p, TransA);
            for (int j = 0; j < N; j++) {
                C[i * ldc + j] += a * getB_f(B, ldb, p, j, TransB);
            }
        }
    }

    return true;
}

bool my_dgemm(enum CBLAS_ORDER Order, enum CBLAS_TRANSPOSE TransA, enum CBLAS_TRANSPOSE TransB,
              int M, int N, int K, double alpha, const double *A, int lda,
              const double *B, int ldb, double beta, double *C, int ldc) {
    if (!check_params(Order, TransA, TransB, M, N, K, lda, ldb, ldc)) return false;
    if (M == 0 || N == 0) return true;

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C[i * ldc + j] = beta * C[i * ldc + j];
        }
    }

    if (K == 0) return true;

    for (int i = 0; i < M; i++) {
        for (int p = 0; p < K; p++) {
            double a = alpha * getA_d(A, lda, i, p, TransA);
            for (int j = 0; j < N; j++) {
                C[i * ldc + j] += a * getB_d(B, ldb, p, j, TransB);
            }
        }
    }

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

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            Cp[i * ldc + j] = cf_mul(b, Cp[i * ldc + j]);
        }
    }

    if (K == 0) return true;

    for (int i = 0; i < M; i++) {
        for (int p = 0; p < K; p++) {
            openblas_complex_float av = getA_c(Ap, lda, i, p, TransA);
            openblas_complex_float aav = cf_mul(a, av);
            for (int j = 0; j < N; j++) {
                openblas_complex_float bv = getB_c(Bp, ldb, p, j, TransB);
                openblas_complex_float add = cf_mul(aav, bv);
                Cp[i * ldc + j] = cf_add(Cp[i * ldc + j], add);
            }
        }
    }

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

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            Cp[i * ldc + j] = cd_mul(b, Cp[i * ldc + j]);
        }
    }

    if (K == 0) return true;

    for (int i = 0; i < M; i++) {
        for (int p = 0; p < K; p++) {
            openblas_complex_double av = getA_z(Ap, lda, i, p, TransA);
            openblas_complex_double aav = cd_mul(a, av);
            for (int j = 0; j < N; j++) {
                openblas_complex_double bv = getB_z(Bp, ldb, p, j, TransB);
                openblas_complex_double add = cd_mul(aav, bv);
                Cp[i * ldc + j] = cd_add(Cp[i * ldc + j], add);
            }
        }
    }

    return true;
}