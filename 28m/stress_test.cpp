#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <cstdlib>
#include <fstream>
#include <x86intrin.h>   // For AVX2 intrinsics
#include <cstdint>

struct Complex {
    double re;
    double im;
    Complex() : re(0.0), im(0.0) {}
    Complex(double r, double i) : re(r), im(i) {}
    static Complex zero() { return Complex(0.0, 0.0); }
    static Complex one()  { return Complex(1.0, 0.0); }
};

std::vector<Complex> initialize_state(size_t size) {
    std::vector<Complex> state(size, Complex::zero());
    state[0] = Complex::one();
    return state;
}

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

            while (i + 1 < end) {
                const double* ptr_a = reinterpret_cast<const double*>(&state[i]);
                const double* ptr_b = reinterpret_cast<const double*>(&state[i + step]);

                __m256d r_a = _mm256_loadu_pd(ptr_a);   
                __m256d r_b = _mm256_loadu_pd(ptr_b);   

                __m256d r_a_perm = _mm256_permute_pd(r_a, 0b0101); 
                __m256d r_b_perm = _mm256_permute_pd(r_b, 0b0101); 

                __m256d term1 = _mm256_mul_pd(r_a, v_cos);
                __m256d term2 = _mm256_mul_pd(r_a_perm, v_sin);
                __m256d term3 = _mm256_mul_pd(r_b, v_sin);
                __m256d term4 = _mm256_mul_pd(r_b_perm, v_cos);
                __m256d res_a = _mm256_add_pd(_mm256_sub_pd(term1, term2), _mm256_add_pd(term3, term4));

                __m256d term5 = _mm256_mul_pd(r_a, v_cos);
                __m256d term6 = _mm256_mul_pd(r_a_perm, v_sin);
                __m256d term7 = _mm256_mul_pd(r_b, v_neg_sin);
                __m256d term8 = _mm256_mul_pd(r_b_perm, v_cos);
                __m256d res_b = _mm256_add_pd(_mm256_add_pd(term5, term6), _mm256_add_pd(term7, term8));

                _mm256_storeu_pd(reinterpret_cast<double*>(&state[i]),        res_a);
                _mm256_storeu_pd(reinterpret_cast<double*>(&state[i + step]), res_b);

                i += 2;
            }

            while (i < end) {
                apply_2x2_scalar(state[i], state[i + step], cos_a, sin_a);
                ++i;
            }
        }
        step <<= 1;
    }
}

int main() {
    // 28,000,000 Total Runs
    const int total_runs = 28000000;
    const int n_qubits = 12; 
    const size_t size = 1ULL << n_qubits;
    const double dummy_angle = 0.7853981633974483;

    std::ofstream outfile("marathon_28m_report.txt");
    if (!outfile.is_open()) {
        std::cerr << "Error creating output file!\n";
        return 1;
    }

    std::cout << "========================================================\n";
    std::cout << "🚀 INITIALIZING MARATHON 28,000,000x STRESS CORRIDOR...\n";
    std::cout << "Target Hardware: AVX2 Vectors vs Classical Scalar\n";
    std::cout << "Exporting to: marathon_28m_report.txt\n";
    std::cout << "========================================================\n";

    outfile << "========================================================\n";
    outfile << "       MARATHON 28,000,000 RUNS PERFORMANCE LOG         \n";
    outfile << "========================================================\n";
    outfile << "Run_Index\tScalar_ns\tAVX2_ns\tSpeedup\n";

    std::vector<Complex> origin_state = initialize_state(size);

    double total_scalar_time = 0, total_avx2_time = 0;
    double max_speedup = 0, min_speedup = 999999.0;
    int best_run = 0, worst_run = 0;
    long long successful_accelerations = 0;

    for (int run = 1; run <= total_runs; ++run) {
        // Console Progress Indicator (Triggers every 1%)
        if (run % 280000 == 0) {
            std::cout << "Progress: " << (run / 280000) << "% Processed [" << run << "/" << total_runs << "]\n";
        }

        // 1. Classical Routine Execution
        std::vector<Complex> state_scalar = origin_state;
        auto t1 = std::chrono::high_resolution_clock::now();
        apply_heavy_phase_gate_scalar(state_scalar, dummy_angle);
        asm volatile("" : : "r"(state_scalar.data()) : "memory");
        auto t2 = std::chrono::high_resolution_clock::now();
        double time_scalar = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();

        // 2. Hardware-Accelerated AVX2 Routine Execution
        std::vector<Complex> state_avx2 = origin_state;
        auto t3 = std::chrono::high_resolution_clock::now();
        apply_heavy_phase_gate_avx2(state_avx2, dummy_angle);
        asm volatile("" : : "r"(state_avx2.data()) : "memory");
        auto t4 = std::chrono::high_resolution_clock::now();
        double time_avx2 = std::chrono::duration_cast<std::chrono::nanoseconds>(t4 - t3).count();

        double speedup = time_scalar / time_avx2;

        total_scalar_time += time_scalar;
        total_avx2_time += time_avx2;

        if (speedup > 1.0) successful_accelerations++;

        if (speedup > max_speedup) {
            max_speedup = speedup;
            best_run = run;
        }
        if (speedup < min_speedup) {
            min_speedup = speedup;
            worst_run = run;
        }

        // Smart IO Throttling: log samples to save SSD lifetime and prevent overhead
        if (run <= 5000 || run % 10000 == 0 || run == total_runs) {
            outfile << run << "\t" << static_cast<long long>(time_scalar) << "\t" 
                    << static_cast<long long>(time_avx2) << "\t" << std::fixed << std::setprecision(2) << speedup << "x\n";
        }
    }

    // Mathematical Evaluations
    double avg_scalar = total_scalar_time / total_runs;
    double avg_avx2 = total_avx2_time / total_runs;
    double avg_speedup = avg_scalar / avg_avx2;
    double acceleration_consistency = (double)successful_accelerations / total_runs * 100.0;

    // Output Final Clean English Academic Summary
    outfile << "\n========================================================\n";
    outfile << "           FINAL 28-MILLION RUN ARCHIVE REPORT          \n";
    outfile << "========================================================\n";
    outfile << "Total Computations Benchmarked  : " << total_runs << "\n";
    outfile << "Global Average Scalar Execution : " << static_cast<long long>(avg_scalar) << " ns\n";
    outfile << "Global Average AVX2 Execution   : " << static_cast<long long>(avg_avx2) << " ns\n";
    outfile << "Hardware Efficiency Consistency : " << std::fixed << std::setprecision(2) << acceleration_consistency << "%\n";
    outfile << "--------------------------------------------------------\n";
    outfile << ">> NET SPEEDUP EFFICIENCY: The pseudo-qubit SIMD routine ran " 
            << std::fixed << std::setprecision(2) << avg_speedup << "x faster than standard scalar bit execution.\n";
    outfile << "--------------------------------------------------------\n";
    outfile << ">> MAXIMUM HARDWARE SPIKE (Cache-Hit Optimal Execution):\n";
    outfile << "   Identified at Run [" << best_run << "] yielding a " << max_speedup << "x speed improvement.\n\n";
    outfile << ">> MINIMUM THROTTLE DETECTED (OS Interrupt / Context Switch Interference):\n";
    outfile << "   Identified at Run [" << worst_run << "] throttling down to " << min_speedup << "x\n";
    outfile << "========================================================\n";

    outfile.close();
    std::cout << "\n🌟 [SUCCESS] 28 Million Runs Completed! Results saved in 'marathon_28m_report.txt'\n";
    return 0;
}