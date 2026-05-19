#include "my_gemm.h"

#include <chrono>
#include <iostream>
#include <random>
#include <vector>

using namespace std;

int main() {
    const int n = 128;
    const int repeat = 3;

    vector<float> A(n * n);
    vector<float> B(n * n);
    vector<float> C(n * n, 0.0f);

    mt19937 rng(42);
    uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (float& x : A) x = dist(rng);
    for (float& x : B) x = dist(rng);

    my_set_num_threads(2);

    auto start = chrono::high_resolution_clock::now();
    for (int i = 0; i < repeat; ++i) {
        bool ok = my_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                           n, n, n, 1.0f, A.data(), n, B.data(), n, 0.0f, C.data(), n);
        if (!ok) {
            cerr << "performance test: my_sgemm returned false" << endl;
            return 1;
        }
    }
    auto finish = chrono::high_resolution_clock::now();

    chrono::duration<double, milli> elapsed = finish - start;
    double checksum = 0.0;
    for (float x : C) checksum += x;

    cout << "performance test: N=" << n
         << ", repeat=" << repeat
         << ", threads=" << my_get_num_threads()
         << ", elapsed_ms=" << elapsed.count()
         << ", checksum=" << checksum << endl;

    if (elapsed.count() <= 0.0) {
        cerr << "performance test: elapsed time is invalid" << endl;
        return 1;
    }

    return 0;
}
