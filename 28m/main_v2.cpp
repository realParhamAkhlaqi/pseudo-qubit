#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <cstdlib>
#include <x86intrin.h>   // For AVX2 intrinsics (GCC/Clang)
#include <cstdint>

// ============================================================
// 1. Complex number type with 32-byte alignment for AVX2
// ============================================================
struct Complex {
    double re;
    double im;

    Complex() : re(0.0), im(0.0) {}
    Complex(double r, double i) : re(r), im(i) {}

    static Complex zero() { return Complex(0.0, 0.0); }
    static Complex one()  { return Complex(1.0, 0.0); }
} __attribute__((aligned(32)));  // 32-byte alignment for aligned loads

// ============================================================
// 2. Initialize state |0...0>
// ============================================================
std::vector<Complex> initialize_state(size_t size) {
    std::vector<Complex> state(size, Complex::zero());
    state[0] = Complex::one();
    return state;
}

// ============================================================
// 3. Helper function for 2x2 matrix multiplication (scalar fallback)
// ============================================================
inline void apply_2x2_scalar(Complex& a, Complex& b, double cos_a, double sin_a) {
    double a_re = a.re, a_im = a.im;
    double b_re = b.re, b_im = b.im;

    double new_ai_re = a_re * cos_a - a_im * sin_a + b_re * sin_a + b_im * cos_a;
    double new_ai_im = a_re * sin_a + a_im * cos_a - b_re * cos_a + b_im * sin_a;
    double new_bj_re = a_re * cos_a + a_im * sin_a - b_re * sin_a + b_im * cos_a;
    double new_bj_im = -a_re * sin_a + a_im * cos_a + b_re * cos_a + b_im * sin_a;

    a = Complex(new_ai_re, new_ai_im);
    b = Complex(new_bj_re, new_bj_im);
}

// ============================================================
// 4. Scalar (classical) implementation
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
                apply_2x2_scalar(state[i], state[i + step], cos_a, sin_a);
            }
        }
        step <<= 1;
    }
}

// ============================================================
// 5. AVX2-optimized implementation (no step==1 special case)
// ============================================================
void apply_heavy_phase_gate_avx2(std::vector<Complex>& state, double angle) {
    const size_t n = state.size();
    const double cos_a = std::cos(angle);
    const double sin_a = std::sin(angle);

    // SIMD constants
    const __m256d v_cos = _mm256_set1_pd(cos_a);
    const __m256d v_sin = _mm256_set1_pd(sin_a);
    const __m256d v_neg_sin = _mm256_set1_pd(-sin_a);

    size_t step = 1;
    while (step < n) {
        size_t block = step << 1;
        for (size_t base = 0; base < n; base += block) {
            size_t end = base + step;
            size_t i = base;

            // Process two pairs at a time (4 complex numbers = 8 doubles)
            while (i + 1 < end && (i + 1 + step) < n) {
                // Load 4 complex numbers (8 doubles)
                const double* ptr_a = reinterpret_cast<const double*>(&state[i]);
                const double* ptr_b = reinterpret_cast<const double*>(&state[i + step]);

                // Aligned loads (requires 32-byte alignment)
                __m256d r_a = _mm256_load_pd(ptr_a);   // [re_i, im_i, re_{i+1}, im_{i+1}]
                __m256d r_b = _mm256_load_pd(ptr_b);   // [re_{i+s}, im_{i+s}, re_{i+1+s}, im_{i+1+s}]

                // Permute real and imaginary components
                __m256d r_a_perm = _mm256_permute_pd(r_a, 0b0101); // [im_i, re_i, im_{i+1}, re_{i+1}]
                __m256d r_b_perm = _mm256_permute_pd(r_b, 0b0101); // [im_{i+s}, re_{i+s}, im_{i+1+s}, re_{i+1+s}]

                // Compute res_a = a*cos - a_perm*sin + b*sin + b_perm*cos
                __m256d term1 = _mm256_mul_pd(r_a, v_cos);
                __m256d term2 = _mm256_mul_pd(r_a_perm, v_sin);
                __m256d term3 = _mm256_mul_pd(r_b, v_sin);
                __m256d term4 = _mm256_mul_pd(r_b_perm, v_cos);

                __m256d res_a = _mm256_add_pd(
                    _mm256_sub_pd(term1, term2),
                    _mm256_add_pd(term3, term4)
                );

                // Compute res_b = a*cos + a_perm*sin - b*sin + b_perm*cos
                __m256d term5 = _mm256_mul_pd(r_a, v_cos);
                __m256d term6 = _mm256_mul_pd(r_a_perm, v_sin);
                __m256d term7 = _mm256_mul_pd(r_b, v_neg_sin);
                __m256d term8 = _mm256_mul_pd(r_b_perm, v_cos);

                __m256d res_b = _mm256_add_pd(
                    _mm256_add_pd(term5, term6),
                    _mm256_add_pd(term7, term8)
                );

                // Store results
                _mm256_store_pd(reinterpret_cast<double*>(&state[i]),         res_a);
                _mm256_store_pd(reinterpret_cast<double*>(&state[i + step]), res_b);

                i += 2;
            }

            // Handle remaining pair with scalar fallback
            while (i < end) {
                apply_2x2_scalar(state[i], state[i + step], cos_a, sin_a);
                ++i;
            }
        }
        step <<= 1;
    }
}

// ============================================================
// 6. Benchmark utility with proper warm-up
// ============================================================
template <typename Func>
double benchmark(const std::string& name, int iterations, std::vector<Complex>& state, Func&& func) {
    // Warm-up (3 iterations)
    for (int i = 0; i < 3; ++i) {
        std::vector<Complex> clone = state;
        func(clone, 0.7853981633974483);
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        std::vector<Complex> clone = state;
        func(clone, 0.7853981633974483);
        // Prevent compiler optimization
        asm volatile("" : : "r"(clone.data()) : "memory");
    }
    auto end = std::chrono::high_resolution_clock::now();

    auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    double avg_ns = static_cast<double>(duration_ns) / iterations;

    std::cout << std::setw(18) << name << " : "
              << std::setw(12) << static_cast<long long>(avg_ns) << " ns avg  ("
              << iterations << " iterations)" << std::endl;
    return avg_ns;
}

// ============================================================
// 7. Main entry point
// ============================================================
int main() {
    const int n_qubits = 10;
    const size_t size = 1ULL << n_qubits;
    const int iterations = 500;
    const double test_angle = 0.7853981633974483;  // 45 degrees

    std::cout << "========================================================" << std::endl;
    std::cout << "Pseudo-Qubit Phase-Shift Gate Simulation (AVX2)" << std::endl;
    std::cout << "Number of qubits    : " << n_qubits << std::endl;
    std::cout << "State vector size   : " << size << std::endl;
    std::cout << "Memory per state    : " << (size * sizeof(Complex)) / (1024 * 1024) << " MB" << std::endl;
    std::cout << "========================================================\n" << std::endl;

    // Initialize state
    std::vector<Complex> state = initialize_state(size);

    // Check AVX2 support
    bool has_avx2 = __builtin_cpu_supports("avx2");
    std::cout << "AVX2 support: " << (has_avx2 ? "Yes" : "No") << std::endl;

    // Scalar benchmark
    double avg_scalar = benchmark("Classic (Scalar)", iterations, state, 
        [](std::vector<Complex>& s, double angle) {
            apply_heavy_phase_gate_scalar(s, angle);
        }
    );

    if (has_avx2) {
        // AVX2 benchmark
        double avg_avx2 = benchmark("Quantum (AVX2)", iterations, state,
            [](std::vector<Complex>& s, double angle) {
                apply_heavy_phase_gate_avx2(s, angle);
            }
        );

        std::cout << "\n--- Summary ---" << std::endl;
        std::cout << "Scalar  : " << static_cast<long long>(avg_scalar) << " ns" << std::endl;
        std::cout << "AVX2    : " << static_cast<long long>(avg_avx2) << " ns" << std::endl;
        double ratio = avg_scalar / avg_avx2;
        std::cout << "Speedup : " << std::fixed << std::setprecision(2) << ratio << "x" << std::endl;
        std::cout << "AVX2 is " << std::setprecision(2) << ratio << "x faster than scalar!" << std::endl;
    }

    return 0;
}
