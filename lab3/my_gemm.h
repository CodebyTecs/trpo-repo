#pragma once

#ifndef FORCE_OPENBLAS_COMPLEX_STRUCT
#define FORCE_OPENBLAS_COMPLEX_STRUCT
#endif

#include <cblas.h>

void my_set_num_threads(int t);
int my_get_num_threads();

bool my_sgemm(enum CBLAS_ORDER Order, enum CBLAS_TRANSPOSE TransA, enum CBLAS_TRANSPOSE TransB,
              int M, int N, int K, float alpha, const float *A, int lda,
              const float *B, int ldb, float beta, float *C, int ldc);

bool my_dgemm(enum CBLAS_ORDER Order, enum CBLAS_TRANSPOSE TransA, enum CBLAS_TRANSPOSE TransB,
              int M, int N, int K, double alpha, const double *A, int lda,
              const double *B, int ldb, double beta, double *C, int ldc);

bool my_cgemm(enum CBLAS_ORDER Order, enum CBLAS_TRANSPOSE TransA, enum CBLAS_TRANSPOSE TransB,
              int M, int N, int K, const void *alpha, const void *A, int lda,
              const void *B, int ldb, const void *beta, void *C, int ldc);

bool my_zgemm(enum CBLAS_ORDER Order, enum CBLAS_TRANSPOSE TransA, enum CBLAS_TRANSPOSE TransB,
              int M, int N, int K, const void *alpha, const void *A, int lda,
              const void *B, int ldb, const void *beta, void *C, int ldc);
