#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <x86intrin.h>   // For AVX2 intrinsics (GCC/Clang)
#include <cstdint>

// ============================================================
// 1. Complex number type
// ============================================================
struct Complex {
    double re;
    double im;

    Complex() : re(0.0), im(0.0) {}
    Complex(double r, double i) : re(r), im(i) {}

    static Complex zero() { return Complex(0.0, 0.0); }
    static Complex one()  { return Complex(1.0, 0.0); }
};

// ============================================================
// 2. Initialize state |0...0>
// ============================================================
std::vector<Complex> initialize_state(size_t size) {
    std::vector<Complex> state(size, Complex::zero());
    state[0] = Complex::one();
    return state;
}

// ============================================================
// 3. Scalar (classical) heavy phase-shift gate
// ============================================================
void apply_heavy_phase_gate_scalar(std::vector<Complex>& state, double angle) {
    const size_t n = state.size();
    const double cos_a = std::cos(angle);
    const double sin_a = std::sin(angle);

    size_t step = 1;
    while (step < n) {
        size_t block = step << 1;
        for (size_t base = 0; base < n; base += block) {
            size_t end = base + step;
            for (size_t i = base; i < end; ++i) {
                size_t j = i + step;
                const Complex& a = state[i];
                const Complex& b = state[j];

                double new_ai_re = a.re * cos_a - a.im * sin_a + b.re * sin_a + b.im * cos_a;
                double new_ai_im = a.re * sin_a + a.im * cos_a - b.re * cos_a + b.im * sin_a;
                double new_bj_re = a.re * cos_a + a.im * sin_a - b.re * sin_a + b.im * cos_a;
                double new_bj_im = -a.re * sin_a + a.im * cos_a + b.re * cos_a + b.im * sin_a;

                state[i] = Complex(new_ai_re, new_ai_im);
                state[j] = Complex(new_bj_re, new_bj_im);
            }
        }
        step <<= 1;
    }
}

// ============================================================
// 4. AVX2-optimized heavy phase-shift gate
// ============================================================
void apply_heavy_phase_gate_avx2(std::vector<Complex>& state, double angle) {
    const size_t n = state.size();
    const double cos_a = std::cos(angle);
    const double sin_a = std::sin(angle);

    const __m256d v_cos = _mm256_set1_pd(cos_a);
    const __m256d v_sin = _mm256_set1_pd(sin_a);
    const __m256d v_neg_sin = _mm256_set1_pd(-sin_a);

    size_t step = 1;
    while (step < n) {
        size_t block = step << 1;
        for (size_t base = 0; base < n; base += block) {
            size_t end = base + step;
            size_t i = base;

            if (step == 1) {
                // Layer 0: adjacent pairs
                while (i < end) {
                    const Complex& a = state[i];
                    const Complex& b = state[i + 1];

                    double new_ai_re = a.re * cos_a - a.im * sin_a + b.re * sin_a + b.im * cos_a;
                    double new_ai_im = a.re * sin_a + a.im * cos_a - b.re * cos_a + b.im * sin_a;
                    double new_bj_re = a.re * cos_a + a.im * sin_a - b.re * sin_a + b.im * cos_a;
                    double new_bj_im = -a.re * sin_a + a.im * cos_a + b.re * cos_a + b.im * sin_a;

                    state[i]     = Complex(new_ai_re, new_ai_im);
                    state[i + 1] = Complex(new_bj_re, new_bj_im);
                    i += 2;
                }
            } else {
                // AVX2 processing: two pairs at a time
                while (i < end) {
                    size_t i_next = i + 1;
                    if (i_next < end && (i_next + step) < n) {
                        // Load two complex numbers from each pair
                        const double* ptr_a = reinterpret_cast<const double*>(&state[i]);
                        const double* ptr_b = reinterpret_cast<const double*>(&state[i + step]);

                        __m256d r_a = _mm256_loadu_pd(ptr_a); // [re_i, im_i, re_{i+1}, im_{i+1}]
                        __m256d r_b = _mm256_loadu_pd(ptr_b); // [re_{i+s}, im_{i+s}, re_{i+1+s}, im_{i+1+s}]

                        // Permute for complex multiplication
                        __m256d r_a_perm = _mm256_permute_pd(r_a, 0b0101); // [im_i, re_i, im_{i+1}, re_{i+1}]
                        __m256d r_b_perm = _mm256_permute_pd(r_b, 0b0101); // [im_{i+s}, re_{i+s}, im_{i+1+s}, re_{i+1+s}]

                        // Compute terms
                        __m256d term1 = _mm256_mul_pd(r_a, v_cos);
                        __m256d term2 = _mm256_mul_pd(r_a_perm, v_sin);
                        __m256d term3 = _mm256_mul_pd(r_b, v_sin);
                        __m256d term4 = _mm256_mul_pd(r_b_perm, v_cos);

                        // Result for a
                        __m256d res_a = _mm256_add_pd(
                            _mm256_sub_pd(term1, term2),
                            _mm256_add_pd(term3, term4)
                        );

                        // Result for b
                        __m256d term5 = _mm256_mul_pd(r_a, v_cos);
                        __m256d term6 = _mm256_mul_pd(r_a_perm, v_sin);
                        __m256d term7 = _mm256_mul_pd(r_b, v_neg_sin);
                        __m256d term8 = _mm256_mul_pd(r_b_perm, v_cos);
                        __m256d res_b = _mm256_add_pd(
                            _mm256_add_pd(term5, term6),
                            _mm256_add_pd(term7, term8)
                        );

                        // Store results
                        _mm256_storeu_pd(reinterpret_cast<double*>(&state[i]),             res_a);
                        _mm256_storeu_pd(reinterpret_cast<double*>(&state[i + step]),      res_b);

                        i += 2;
                    } else {
                        // Remaining pair (scalar)
                        const Complex& a = state[i];
                        const Complex& b = state[i + step];

                        double new_ai_re = a.re * cos_a - a.im * sin_a + b.re * sin_a + b.im * cos_a;
                        double new_ai_im = a.re * sin_a + a.im * cos_a - b.re * cos_a + b.im * sin_a;
                        double new_bj_re = a.re * cos_a + a.im * sin_a - b.re * sin_a + b.im * cos_a;
                        double new_bj_im = -a.re * sin_a + a.im * cos_a + b.re * cos_a + b.im * sin_a;

                        state[i] = Complex(new_ai_re, new_ai_im);
                        state[i + step] = Complex(new_bj_re, new_bj_im);
                        ++i;
                    }
                }
            }
        }
        step <<= 1;
    }
}

// ============================================================
// 5. Benchmark utility
// ============================================================
template <typename Func>
double benchmark(const std::string& name, int iterations, Func&& func) {
    // Warm-up (3 times)
    for (int i = 0; i < 3; ++i) func();

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        func();
    }
    auto end = std::chrono::high_resolution_clock::now();

    auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    double avg_ns = static_cast<double>(duration_ns) / iterations;

    std::cout << std::setw(15) << name << " : "
              << std::setw(12) << static_cast<long long>(avg_ns) << " ns avg  ("
              << iterations << " iterations)" << std::endl;
    return avg_ns;
}

// ============================================================
// 6. Main entry point
// ============================================================
int main() {
    const int n_qubits = 10;
    const size_t size = 1ULL << n_qubits;  // 2^10
    const int iterations = 500;
    const double test_angle = 0.7853981633974483; // 45 degrees

    std::cout << "========================================================" << std::endl;
    std::cout << "HEAVY Pseudo-Qubit Phase-Shift Gate Simulation" << std::endl;
    std::cout << "Number of qubits    : " << n_qubits << std::endl;
    std::cout << "State vector size   : " << size << std::endl;
    std::cout << "Memory per state    : " << (size * 16) / (1024 * 1024) << " MB" << std::endl;
    std::cout << "========================================================\n" << std::endl;

    // Initialize state
    std::vector<Complex> state = initialize_state(size);

    // ---------- Scalar benchmark ----------
    double avg_scalar = benchmark("Classic (Scalar)", iterations, [&]() {
        std::vector<Complex> clone = state; // Copy
        apply_heavy_phase_gate_scalar(clone, test_angle);
        return clone; // Prevent compiler optimization
    });

    // ---------- AVX2 benchmark ----------
    bool has_avx2 = __builtin_cpu_supports("avx2");
    if (has_avx2) {
        std::cout << "\nAVX2 detected! Running HEAVY optimized version...\n" << std::endl;
        double avg_avx2 = benchmark("Quantum (AVX2)", iterations, [&]() {
            std::vector<Complex> clone = state;
            apply_heavy_phase_gate_avx2(clone, test_angle);
            return clone;
        });

        // ---------- Summary ----------
        std::cout << "\n--- Heavy Gate Summary ---" << std::endl;
        std::cout << "Classic (Scalar)  : " << static_cast<long long>(avg_scalar) << " ns" << std::endl;
        std::cout << "Quantum (AVX2)    : " << static_cast<long long>(avg_avx2) << " ns" << std::endl;
        double ratio = avg_scalar / avg_avx2;
        std::cout << "Speed ratio (Scalar / AVX2) : " << std::fixed << std::setprecision(2) << ratio << "x" << std::endl;
        std::cout << "AVX2 is " << std::setprecision(2) << ratio << "x faster on heavy math!" << std::endl;
    } else {
        std::cout << "\nAVX2 not supported on this CPU." << std::endl;
    }

    return 0;
}