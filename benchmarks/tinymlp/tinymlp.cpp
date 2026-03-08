#include <cstdint>
#include <cstdio>

// TinyMLP: real MLP inference (dense layers) + input-dependent sparsity skip.
// This is meant to create meaningful branch behavior:
//   if (x == 0) continue;   // data-dependent branch (varies with inputs)
//
// Architecture:
//   Input(256) -> FC(256) -> ReLU+clamp -> FC(10) -> argmax
//
// Deterministic pseudo-random init (no libc rand()) so results are reproducible.

static inline int8_t relu_clamp(int32_t x) {
    if (x <= 0) return 0;
    if (x >= 127) return 127;
    return (int8_t)x;
}

int main() {
    constexpr int IN  = 256;
    constexpr int HID = 256;
    constexpr int OUT = 10;

    static int8_t  x[IN];
    static int8_t  w1[HID][IN];
    static int8_t  w2[OUT][HID];
    static int8_t  h[HID];
    static int32_t y[OUT];

    // Deterministic RNG
    uint32_t seed = 1u;
    auto rnd = [&seed]() -> uint32_t {
        seed = seed * 1103515245u + 12345u;
        return (seed >> 16) & 0x7FFFu;
    };

    // Initialize input with controlled sparsity:
    // ~50% zeros to trigger the (x==0) skip branch often.
    for (int i = 0; i < IN; i++) {
        int v = (int)(rnd() % 31) - 15; // [-15..15]
        if ((rnd() & 1u) == 0u) v = 0;  // 50% zeros
        x[i] = (int8_t)v;
    }

    // Initialize weights (small range)
    for (int o = 0; o < HID; o++)
        for (int i = 0; i < IN; i++)
            w1[o][i] = (int8_t)((int)(rnd() % 7) - 3); // [-3..3]

    for (int o = 0; o < OUT; o++)
        for (int i = 0; i < HID; i++)
            w2[o][i] = (int8_t)((int)(rnd() % 7) - 3);

    // Run multiple inferences to stabilize stats
    constexpr int ITERS = 200;
    int64_t checksum = 0;
    int64_t skipped = 0;

    for (int it = 0; it < ITERS; it++) {
        // FC1 with sparsity skip
        for (int o = 0; o < HID; o++) {
            int32_t acc = 0;
            for (int i = 0; i < IN; i++) {
                int8_t xi = x[i];
                if (xi == 0) { // data-dependent branch
                    skipped++;
                    continue;
                }
                acc += (int32_t)w1[o][i] * (int32_t)xi;
            }
            h[o] = relu_clamp(acc);
            checksum += h[o];
        }

        // FC2 (small OUT) - no sparsity skip here (keep it simple)
        for (int o = 0; o < OUT; o++) {
            int32_t acc = 0;
            for (int i = 0; i < HID; i++)
                acc += (int32_t)w2[o][i] * (int32_t)h[i];
            y[o] = acc;
            checksum += acc;
        }

        // Argmax (branchy comparisons)
        int best = 0;
        for (int o = 1; o < OUT; o++) {
            if (y[o] > y[best]) best = o;
        }
        checksum += best;

        // Slightly mutate input each iter to vary branch outcomes but stay deterministic
        int idx = (int)(rnd() % IN);
        int v = (int)(rnd() % 31) - 15;
        if ((rnd() & 3u) == 0u) v = 0; // keep sparsity
        x[idx] = (int8_t)v;
    }

    printf("tinymlp checksum: %lld\n", (long long)checksum);
    printf("tinymlp skipped_multiplies: %lld\n", (long long)skipped);
    return 0;
}