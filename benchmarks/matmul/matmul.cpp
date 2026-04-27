#include <cstdint>
#include <cstdio>

static const int N    = 64;
static const int REPS = 20;

static int32_t A[N][N];
static int32_t B[N][N];
static int32_t C[N][N];

int main()
{
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            A[i][j] = (int32_t)(i + j);
            B[i][j] = (int32_t)(i * 2 + j);
        }

    int64_t checksum = 0;

    for (int rep = 0; rep < REPS; rep++) {
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) {
                int32_t acc = 0;
                for (int k = 0; k < N; k++)
                    acc += A[i][k] * B[k][j];
                C[i][j] = acc;
            }

        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                checksum += C[i][j];
    }

    printf("checksum: %lld\n", (long long)checksum);
    return 0;
}
