#include <cblas.h>
#include <cstdio>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <cstdlib>

#include "my_gemm.h"

using namespace std;

extern "C" void openblas_set_num_threads(int num_threads);
extern "C" int openblas_get_num_threads(void);
extern "C" char* openblas_get_config(void);

float rand_f(mt19937& rng) {
    uniform_real_distribution<float> d(-1.0f, 1.0f);
    return d(rng);
}

double rand_d(mt19937& rng) {
    uniform_real_distribution<double> d(-1.0, 1.0);
    return d(rng);
}

openblas_complex_float rand_cf(mt19937& rng) {
    openblas_complex_float z;
    z.real = rand_f(rng);
    z.imag = rand_f(rng);
    return z;
}

openblas_complex_double rand_cd(mt19937& rng) {
    openblas_complex_double z;
    z.real = rand_d(rng);
    z.imag = rand_d(rng);
    return z;
}

void fill_float_vec(vector<float>& v, mt19937& rng) { for (auto& x : v) x = rand_f(rng); }
void fill_double_vec(vector<double>& v, mt19937& rng) { for (auto& x : v) x = rand_d(rng); }
void fill_cfloat_vec(vector<openblas_complex_float>& v, mt19937& rng) { for (auto& x : v) x = rand_cf(rng); }
void fill_cdouble_vec(vector<openblas_complex_double>& v, mt19937& rng) { for (auto& x : v) x = rand_cd(rng); }

void apply_threads(int t) {
    char buf[32];
    snprintf(buf, sizeof(buf), "%d", t);
    setenv("OMP_DYNAMIC", "FALSE", 1);
    setenv("OMP_NUM_THREADS", buf, 1);
    setenv("OPENBLAS_NUM_THREADS", buf, 1);
    setenv("GOTO_NUM_THREADS", buf, 1);
    openblas_set_num_threads(t);
    my_set_num_threads(t);
}

double bench_my_s_ms(int N, int inner, const vector<float>& A, const vector<float>& B, vector<float>& C) {
    int M = N, K = N, lda = K, ldb = N, ldc = N;
    auto t0 = chrono::high_resolution_clock::now();
    for (int it = 0; it < inner; it++) {
        float alpha = 1.0f;
        float beta = (float)(it & 1);
        bool ok = my_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                           M, N, K, alpha, A.data(), lda, B.data(), ldb, beta, C.data(), ldc);
        if (!ok) return -1.0;
    }
    auto t1 = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> ms = t1 - t0;
    return ms.count();
}

double bench_ob_s_ms(int N, int inner, const vector<float>& A, const vector<float>& B, vector<float>& C) {
    int M = N, K = N, lda = K, ldb = N, ldc = N;
    auto t0 = chrono::high_resolution_clock::now();
    for (int it = 0; it < inner; it++) {
        float alpha = 1.0f;
        float beta = (float)(it & 1);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    M, N, K, alpha, A.data(), lda, B.data(), ldb, beta, C.data(), ldc);
    }
    auto t1 = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> ms = t1 - t0;
    return ms.count();
}

double bench_my_d_ms(int N, int inner, const vector<double>& A, const vector<double>& B, vector<double>& C) {
    int M = N, K = N, lda = K, ldb = N, ldc = N;
    auto t0 = chrono::high_resolution_clock::now();
    for (int it = 0; it < inner; it++) {
        double alpha = 1.0;
        double beta = (double)(it & 1);
        bool ok = my_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                           M, N, K, alpha, A.data(), lda, B.data(), ldb, beta, C.data(), ldc);
        if (!ok) return -1.0;
    }
    auto t1 = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> ms = t1 - t0;
    return ms.count();
}

double bench_ob_d_ms(int N, int inner, const vector<double>& A, const vector<double>& B, vector<double>& C) {
    int M = N, K = N, lda = K, ldb = N, ldc = N;
    auto t0 = chrono::high_resolution_clock::now();
    for (int it = 0; it < inner; it++) {
        double alpha = 1.0;
        double beta = (double)(it & 1);
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    M, N, K, alpha, A.data(), lda, B.data(), ldb, beta, C.data(), ldc);
    }
    auto t1 = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> ms = t1 - t0;
    return ms.count();
}

double bench_my_c_ms(int N, int inner, const vector<openblas_complex_float>& A, const vector<openblas_complex_float>& B, vector<openblas_complex_float>& C) {
    int M = N, K = N, lda = K, ldb = N, ldc = N;
    auto t0 = chrono::high_resolution_clock::now();
    for (int it = 0; it < inner; it++) {
        openblas_complex_float alpha; alpha.real = 1.0f; alpha.imag = 0.0f;
        openblas_complex_float beta; beta.real = (float)(it & 1); beta.imag = 0.0f;
        bool ok = my_cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                           M, N, K, &alpha, A.data(), lda, B.data(), ldb, &beta, C.data(), ldc);
        if (!ok) return -1.0;
    }
    auto t1 = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> ms = t1 - t0;
    return ms.count();
}

double bench_ob_c_ms(int N, int inner, const vector<openblas_complex_float>& A, const vector<openblas_complex_float>& B, vector<openblas_complex_float>& C) {
    int M = N, K = N, lda = K, ldb = N, ldc = N;
    auto t0 = chrono::high_resolution_clock::now();
    for (int it = 0; it < inner; it++) {
        openblas_complex_float alpha; alpha.real = 1.0f; alpha.imag = 0.0f;
        openblas_complex_float beta; beta.real = (float)(it & 1); beta.imag = 0.0f;
        cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    M, N, K, &alpha, A.data(), lda, B.data(), ldb, &beta, C.data(), ldc);
    }
    auto t1 = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> ms = t1 - t0;
    return ms.count();
}

double bench_my_z_ms(int N, int inner, const vector<openblas_complex_double>& A, const vector<openblas_complex_double>& B, vector<openblas_complex_double>& C) {
    int M = N, K = N, lda = K, ldb = N, ldc = N;
    auto t0 = chrono::high_resolution_clock::now();
    for (int it = 0; it < inner; it++) {
        openblas_complex_double alpha; alpha.real = 1.0; alpha.imag = 0.0;
        openblas_complex_double beta; beta.real = (double)(it & 1); beta.imag = 0.0;
        bool ok = my_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                           M, N, K, &alpha, A.data(), lda, B.data(), ldb, &beta, C.data(), ldc);
        if (!ok) return -1.0;
    }
    auto t1 = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> ms = t1 - t0;
    return ms.count();
}

double bench_ob_z_ms(int N, int inner, const vector<openblas_complex_double>& A, const vector<openblas_complex_double>& B, vector<openblas_complex_double>& C) {
    int M = N, K = N, lda = K, ldb = N, ldc = N;
    auto t0 = chrono::high_resolution_clock::now();
    for (int it = 0; it < inner; it++) {
        openblas_complex_double alpha; alpha.real = 1.0; alpha.imag = 0.0;
        openblas_complex_double beta; beta.real = (double)(it & 1); beta.imag = 0.0;
        cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    M, N, K, &alpha, A.data(), lda, B.data(), ldb, &beta, C.data(), ldc);
    }
    auto t1 = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> ms = t1 - t0;
    return ms.count();
}

double geo_mean(const vector<double>& v) {
    double s = 0.0;
    for (double x : v) s += log(x);
    return exp(s / (double)v.size());
}

int pick_inner_my_s_1t_60s(int N, const vector<float>& A, const vector<float>& B, vector<float>& C, double want_run_ms) {
    apply_threads(1);
    bench_my_s_ms(N, 2, A, B, C);

    double one = bench_my_s_ms(N, 1, A, B, C);
    if (one <= 0.0) return 1;

    int inner = (int)ceil(want_run_ms / one);
    if (inner < 1) inner = 1;

    while (true) {
        double check = bench_my_s_ms(N, inner, A, B, C);
        if (check >= want_run_ms) return inner;
        int add = (int)ceil((want_run_ms - check) / one);
        if (add < 1) add = 1;
        inner += add;
    }
}

int pick_inner_my_d_1t_60s(int N, const vector<double>& A, const vector<double>& B, vector<double>& C, double want_run_ms) {
    apply_threads(1);
    bench_my_d_ms(N, 2, A, B, C);

    double one = bench_my_d_ms(N, 1, A, B, C);
    if (one <= 0.0) return 1;

    int inner = (int)ceil(want_run_ms / one);
    if (inner < 1) inner = 1;

    while (true) {
        double check = bench_my_d_ms(N, inner, A, B, C);
        if (check >= want_run_ms) return inner;
        int add = (int)ceil((want_run_ms - check) / one);
        if (add < 1) add = 1;
        inner += add;
    }
}

int pick_inner_my_c_1t_60s(int N, const vector<openblas_complex_float>& A, const vector<openblas_complex_float>& B, vector<openblas_complex_float>& C, double want_run_ms) {
    apply_threads(1);
    bench_my_c_ms(N, 2, A, B, C);

    double one = bench_my_c_ms(N, 1, A, B, C);
    if (one <= 0.0) return 1;

    int inner = (int)ceil(want_run_ms / one);
    if (inner < 1) inner = 1;

    while (true) {
        double check = bench_my_c_ms(N, inner, A, B, C);
        if (check >= want_run_ms) return inner;
        int add = (int)ceil((want_run_ms - check) / one);
        if (add < 1) add = 1;
        inner += add;
    }
}

int pick_inner_my_z_1t_60s(int N, const vector<openblas_complex_double>& A, const vector<openblas_complex_double>& B, vector<openblas_complex_double>& C, double want_run_ms) {
    apply_threads(1);
    bench_my_z_ms(N, 2, A, B, C);

    double one = bench_my_z_ms(N, 1, A, B, C);
    if (one <= 0.0) return 1;

    int inner = (int)ceil(want_run_ms / one);
    if (inner < 1) inner = 1;

    while (true) {
        double check = bench_my_z_ms(N, inner, A, B, C);
        if (check >= want_run_ms) return inner;
        int add = (int)ceil((want_run_ms - check) / one);
        if (add < 1) add = 1;
        inner += add;
    }
}

void run_table_float(int N, int threads, int inner, const vector<float>& A, const vector<float>& B, vector<float>& C) {
    apply_threads(threads);

    printf("\n[float] threads=%d N=%d inner=%d\n", threads, N, inner);
    puts("run  my_ms      openblas_ms  perf_%");
    puts("-------------------------------------");

    vector<double> pcts;
    pcts.reserve(10);

    for (int run = 1; run <= 10; run++) {
        double my_ms = bench_my_s_ms(N, inner, A, B, C);
        double ob_ms = bench_ob_s_ms(N, inner, A, B, C);
        double pct = (ob_ms / my_ms) * 100.0;
        pcts.push_back(pct);
        printf("%2d   %9.3f  %11.3f  %7.2f%%\n", run, my_ms, ob_ms, pct);
    }

    puts("-------------------------------------");
    printf("geo_mean: %.2f%%\n", geo_mean(pcts));
}

void run_table_double(int N, int threads, int inner, const vector<double>& A, const vector<double>& B, vector<double>& C) {
    apply_threads(threads);

    printf("\n[double] threads=%d N=%d inner=%d\n", threads, N, inner);
    puts("run  my_ms      openblas_ms  perf_%");
    puts("-------------------------------------");

    vector<double> pcts;
    pcts.reserve(10);

    for (int run = 1; run <= 10; run++) {
        double my_ms = bench_my_d_ms(N, inner, A, B, C);
        double ob_ms = bench_ob_d_ms(N, inner, A, B, C);
        double pct = (ob_ms / my_ms) * 100.0;
        pcts.push_back(pct);
        printf("%2d   %9.3f  %11.3f  %7.2f%%\n", run, my_ms, ob_ms, pct);
    }

    puts("-------------------------------------");
    printf("geo_mean: %.2f%%\n", geo_mean(pcts));
}

void run_table_cfloat(int N, int threads, int inner, const vector<openblas_complex_float>& A, const vector<openblas_complex_float>& B, vector<openblas_complex_float>& C) {
    apply_threads(threads);

    printf("\n[cgemm] threads=%d N=%d inner=%d\n", threads, N, inner);
    puts("run  my_ms      openblas_ms  perf_%");
    puts("-------------------------------------");

    vector<double> pcts;
    pcts.reserve(10);

    for (int run = 1; run <= 10; run++) {
        double my_ms = bench_my_c_ms(N, inner, A, B, C);
        double ob_ms = bench_ob_c_ms(N, inner, A, B, C);
        double pct = (ob_ms / my_ms) * 100.0;
        pcts.push_back(pct);
        printf("%2d   %9.3f  %11.3f  %7.2f%%\n", run, my_ms, ob_ms, pct);
    }

    puts("-------------------------------------");
    printf("geo_mean: %.2f%%\n", geo_mean(pcts));
}

void run_table_cdouble(int N, int threads, int inner, const vector<openblas_complex_double>& A, const vector<openblas_complex_double>& B, vector<openblas_complex_double>& C) {
    apply_threads(threads);

    printf("\n[zgemm] threads=%d N=%d inner=%d\n", threads, N, inner);
    puts("run  my_ms      openblas_ms  perf_%");
    puts("-------------------------------------");

    vector<double> pcts;
    pcts.reserve(10);

    for (int run = 1; run <= 10; run++) {
        double my_ms = bench_my_z_ms(N, inner, A, B, C);
        double ob_ms = bench_ob_z_ms(N, inner, A, B, C);
        double pct = (ob_ms / my_ms) * 100.0;
        pcts.push_back(pct);
        printf("%2d   %9.3f  %11.3f  %7.2f%%\n", run, my_ms, ob_ms, pct);
    }

    puts("-------------------------------------");
    printf("geo_mean: %.2f%%\n", geo_mean(pcts));
}

int main() {
    int N = 1024;
    double want_run_ms_1thread = 60000.0;

    int threads_list[5] = {1, 2, 4, 8, 16};

    int M = N, K = N;
    int sizeA = M * K, sizeB = K * N, sizeC = M * N;

    mt19937 rng(42);

    vector<float> Af(sizeA), Bf(sizeB), Cf(sizeC);
    vector<double> Ad(sizeA), Bd(sizeB), Cd(sizeC);
    vector<openblas_complex_float> Ac(sizeA), Bc(sizeB), Cc(sizeC);
    vector<openblas_complex_double> Az(sizeA), Bz(sizeB), Cz(sizeC);

    fill_float_vec(Af, rng); fill_float_vec(Bf, rng); fill_float_vec(Cf, rng);
    fill_double_vec(Ad, rng); fill_double_vec(Bd, rng); fill_double_vec(Cd, rng);
    fill_cfloat_vec(Ac, rng); fill_cfloat_vec(Bc, rng); fill_cfloat_vec(Cc, rng);
    fill_cdouble_vec(Az, rng); fill_cdouble_vec(Bz, rng); fill_cdouble_vec(Cz, rng);

    int inner_s = pick_inner_my_s_1t_60s(N, Af, Bf, Cf, want_run_ms_1thread);
    int inner_d = pick_inner_my_d_1t_60s(N, Ad, Bd, Cd, want_run_ms_1thread);
    int inner_c = pick_inner_my_c_1t_60s(N, Ac, Bc, Cc, want_run_ms_1thread);
    int inner_z = pick_inner_my_z_1t_60s(N, Az, Bz, Cz, want_run_ms_1thread);

    for (int i = 0; i < 5; i++) {
        int t = threads_list[i];
        run_table_float(N, t, inner_s, Af, Bf, Cf);
        run_table_double(N, t, inner_d, Ad, Bd, Cd);
        run_table_cfloat(N, t, inner_c, Ac, Bc, Cc);
        run_table_cdouble(N, t, inner_z, Az, Bz, Cz);
    }

    return 0;
}