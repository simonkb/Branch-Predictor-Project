#include <cstdint>
#include <cstdio>

// TinyMLP: realistic MLP inference with magnitude-based sparsity pruning.
//
// Models quantised INT8 inference with per-layer bias terms and
// threshold-based activation pruning — the same optimisation used by
// real sparse-inference engines (e.g. DeepSparse, TVM sparse schedules).
//
// Branch behaviour:
//   - Counted loops (for i/j/k) — highly predictable by loop predictors.
//   - Threshold sparsity skips (|x| < TAU) — data-dependent, hard to predict.
//   - ReLU clamp — weakly biased conditional.
//   - Argmax comparison — data-dependent.
//
// Architecture:
//   Input(256) -[FC1+bias]-> ReLU+clamp -[FC2+bias, sparsity skip]-> argmax
//
// Deterministic pseudo-random init (no libc rand()) so results are
// reproducible across runs.

static const int SPARSITY_THRESHOLD = 2; // |activation| < TAU  → skip multiply

static inline int8_t relu_clamp(int32_t x) {
    if (x <= 0) return 0;
    if (x >= 127) return 127;
    return (int8_t)x;
}

static inline int32_t abs32(int32_t x) { return x < 0 ? -x : x; }

int main() {
    constexpr int IN  = 256;
    constexpr int HID = 256;
    constexpr int OUT = 10;

    static int8_t  x[IN];
    static int8_t  w1[HID][IN];
    static int32_t b1[HID];
    static int8_t  w2[OUT][HID];
    static int32_t b2[OUT];
    static int8_t  h[HID];
    static int32_t y[OUT];

    // Deterministic LCG
    uint32_t seed = 1u;
    auto rnd = [&seed]() -> uint32_t {
        seed = seed * 1103515245u + 12345u;
        return (seed >> 16) & 0x7FFFu;
    };

    // Initialize input with controlled sparsity:
    // ~50% near-zero values to trigger the threshold-skip branch often.
    for (int i = 0; i < IN; i++) {
        int v = (int)(rnd() % 31) - 15; // [-15..15]
        if ((rnd() & 1u) == 0u) v = (int)(rnd() % 3) - 1; // near-zero: {-1,0,1}
        x[i] = (int8_t)v;
    }

    // Initialize weights (small range) and biases
    for (int o = 0; o < HID; o++) {
        for (int i = 0; i < IN; i++)
            w1[o][i] = (int8_t)((int)(rnd() % 7) - 3); // [-3..3]
        b1[o] = (int32_t)((int)(rnd() % 21) - 10);      // [-10..10]
    }

    for (int o = 0; o < OUT; o++) {
        for (int i = 0; i < HID; i++)
            w2[o][i] = (int8_t)((int)(rnd() % 7) - 3);
        b2[o] = (int32_t)((int)(rnd() % 21) - 10);
    }

    // Run multiple inferences to stabilise branch predictor stats
    constexpr int ITERS = 200;
    int64_t checksum = 0;
    int64_t skipped  = 0;

    for (int it = 0; it < ITERS; it++) {
        // ---- FC1 with threshold sparsity skip ----
        for (int o = 0; o < HID; o++) {
            int32_t acc = b1[o];
            for (int i = 0; i < IN; i++) {
                int8_t xi = x[i];
                if (abs32(xi) < SPARSITY_THRESHOLD) { // magnitude pruning
                    skipped++;
                    continue;
                }
                acc += (int32_t)w1[o][i] * (int32_t)xi;
            }
            h[o] = relu_clamp(acc);
            checksum += h[o];
        }

        // ---- FC2 with sparsity skip on hidden activations ----
        // After ReLU many h[i] are zero → skip those multiplies.
        for (int o = 0; o < OUT; o++) {
            int32_t acc = b2[o];
            for (int i = 0; i < HID; i++) {
                int8_t hi = h[i];
                if (abs32(hi) < SPARSITY_THRESHOLD) { // ReLU-zero + near-zero skip
                    skipped++;
                    continue;
                }
                acc += (int32_t)w2[o][i] * (int32_t)hi;
            }
            y[o] = acc;
            checksum += acc;
        }

        // ---- Argmax (branchy comparisons) ----
        int best = 0;
        for (int o = 1; o < OUT; o++) {
            if (y[o] > y[best]) best = o;
        }
        checksum += best;

        // Slightly mutate input each iter to vary branch outcomes
        int idx = (int)(rnd() % IN);
        int v = (int)(rnd() % 31) - 15;
        if ((rnd() & 3u) == 0u) v = (int)(rnd() % 3) - 1; // keep some near-zero
        x[idx] = (int8_t)v;
    }

    printf("tinymlp checksum: %lld\n", (long long)checksum);
    printf("tinymlp skipped_multiplies: %lld\n", (long long)skipped);
    return 0;
}