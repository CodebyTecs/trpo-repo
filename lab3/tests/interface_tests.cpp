#include "my_gemm.h"

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

using namespace std;

bool almost_equal(double a, double b, double eps = 1e-5) {
    return fabs(a - b) <= eps;
}

bool check_float_vector(const vector<float>& actual, const vector<float>& expected, const string& name) {
    if (actual.size() != expected.size()) {
        cerr << name << ": size mismatch" << endl;
        return false;
    }

    for (size_t i = 0; i < actual.size(); ++i) {
        if (!almost_equal(actual[i], expected[i])) {
            cerr << name << ": mismatch at " << i << ", expected " << expected[i]
                 << ", got " << actual[i] << endl;
            return false;
        }
    }

    return true;
}

bool check_double_vector(const vector<double>& actual, const vector<double>& expected, const string& name) {
    if (actual.size() != expected.size()) {
        cerr << name << ": size mismatch" << endl;
        return false;
    }

    for (size_t i = 0; i < actual.size(); ++i) {
        if (!almost_equal(actual[i], expected[i], 1e-9)) {
            cerr << name << ": mismatch at " << i << ", expected " << expected[i]
                 << ", got " << actual[i] << endl;
            return false;
        }
    }

    return true;
}

bool check_cfloat_vector(const vector<openblas_complex_float>& actual,
                         const vector<openblas_complex_float>& expected,
                         const string& name) {
    if (actual.size() != expected.size()) {
        cerr << name << ": size mismatch" << endl;
        return false;
    }

    for (size_t i = 0; i < actual.size(); ++i) {
        if (!almost_equal(actual[i].real, expected[i].real) ||
            !almost_equal(actual[i].imag, expected[i].imag)) {
            cerr << name << ": mismatch at " << i << endl;
            return false;
        }
    }

    return true;
}

bool check_cdouble_vector(const vector<openblas_complex_double>& actual,
                          const vector<openblas_complex_double>& expected,
                          const string& name) {
    if (actual.size() != expected.size()) {
        cerr << name << ": size mismatch" << endl;
        return false;
    }

    for (size_t i = 0; i < actual.size(); ++i) {
        if (!almost_equal(actual[i].real, expected[i].real, 1e-9) ||
            !almost_equal(actual[i].imag, expected[i].imag, 1e-9)) {
            cerr << name << ": mismatch at " << i << endl;
            return false;
        }
    }

    return true;
}

bool test_sgemm_basic() {
    vector<float> A = {1, 2, 3,
                       4, 5, 6};
    vector<float> B = {7, 8,
                       9, 10,
                       11, 12};
    vector<float> C(4, 0.0f);

    bool ok = my_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                       2, 2, 3, 1.0f, A.data(), 3, B.data(), 2, 0.0f, C.data(), 2);

    return ok && check_float_vector(C, {58, 64, 139, 154}, "sgemm basic");
}

bool test_dgemm_alpha_beta() {
    vector<double> A = {1, 2,
                        3, 4};
    vector<double> B = {5, 6,
                        7, 8};
    vector<double> C = {1, 2,
                        3, 4};

    bool ok = my_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                       2, 2, 2, 0.5, A.data(), 2, B.data(), 2, 2.0, C.data(), 2);

    return ok && check_double_vector(C, {11.5, 15.0, 27.5, 33.0}, "dgemm alpha beta");
}

bool test_sgemm_transpose() {
    vector<float> A = {1, 4,
                       2, 5,
                       3, 6};
    vector<float> B = {7, 8,
                       9, 10,
                       11, 12};
    vector<float> C(4, 0.0f);

    bool ok = my_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                       2, 2, 3, 1.0f, A.data(), 2, B.data(), 2, 0.0f, C.data(), 2);

    return ok && check_float_vector(C, {58, 64, 139, 154}, "sgemm transpose");
}

bool test_cgemm_basic() {
    vector<openblas_complex_float> A = {{1.0f, 1.0f}, {2.0f, 0.0f}};
    vector<openblas_complex_float> B = {{3.0f, -1.0f}, {4.0f, 2.0f}};
    vector<openblas_complex_float> C = {{0.0f, 0.0f}};
    openblas_complex_float alpha = {1.0f, 0.0f};
    openblas_complex_float beta = {0.0f, 0.0f};

    bool ok = my_cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                       1, 1, 2, &alpha, A.data(), 2, B.data(), 1, &beta, C.data(), 1);

    return ok && check_cfloat_vector(C, {{12.0f, 6.0f}}, "cgemm basic");
}

bool test_zgemm_basic() {
    vector<openblas_complex_double> A = {{1.0, -1.0}, {2.0, 1.0}};
    vector<openblas_complex_double> B = {{3.0, 2.0}, {-1.0, 4.0}};
    vector<openblas_complex_double> C = {{0.0, 0.0}};
    openblas_complex_double alpha = {1.0, 0.0};
    openblas_complex_double beta = {0.0, 0.0};

    bool ok = my_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                       1, 1, 2, &alpha, A.data(), 2, B.data(), 1, &beta, C.data(), 1);

    return ok && check_cdouble_vector(C, {{-1.0, 6.0}}, "zgemm basic");
}

bool test_invalid_params() {
    vector<float> A = {1, 2, 3, 4};
    vector<float> B = {1, 2, 3, 4};
    vector<float> C(4, 0.0f);

    bool ok = my_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                       2, 2, 2, 1.0f, A.data(), 2, B.data(), 2, 0.0f, C.data(), 2);

    return !ok;
}

int main() {
    struct TestCase {
        const char* name;
        bool (*run)();
    };

    vector<TestCase> tests = {
        {"sgemm basic", test_sgemm_basic},
        {"dgemm alpha beta", test_dgemm_alpha_beta},
        {"sgemm transpose", test_sgemm_transpose},
        {"cgemm basic", test_cgemm_basic},
        {"zgemm basic", test_zgemm_basic},
        {"invalid params", test_invalid_params},
    };

    for (const TestCase& test : tests) {
        if (!test.run()) {
            cerr << "[FAIL] " << test.name << endl;
            return 1;
        }
        cout << "[PASS] " << test.name << endl;
    }

    cout << "All interface tests passed" << endl;
    return 0;
}
