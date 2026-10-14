/* fxfft --- frozen deterministic fixed-point FFT reference, any supported size.
 *
 * This is the size-parameterized companion to fxfft256_ref.c. That file stays
 * exactly as it is: it is the frozen fxfft256 v1 artifact, it is what the
 * golden vectors were produced against, and it is what kotekan vendors. This
 * file exists because frame sizes of 2**15 and 2**16 put L_F = 2N/K at 512,
 * 1024 and 2048, and the deployed transform has to follow.
 *
 * Same specification, length as a parameter:
 *   FX_N-point forward DFT (e^{-2 pi i b m / FX_N}), radix-2 decimation in
 *   time, bit-reversed load of the zero-padded FX_N/2-sample input,
 *   natural-order output, round15 per butterfly, no scaling.
 *
 * Twiddles come from the single master table in fxfft_master_twiddle.h, which
 * is generated from pilot_proxy.fxfft.MASTER_TWIDDLE_Q15. Because
 * W_n[k] = W_M[k * M / n] holds exactly on the pre-rounding real value, the
 * entries a length-n transform reads are bit-identical to a table generated
 * at length n -- adding a size introduces no new rounding decisions. The
 * stride folds into the shift: stage s reads master[t << (FX_MASTER_LOG2 - s)],
 * which at FX_N = 256 selects exactly the entries the frozen table holds.
 *
 * Set the size at compile time, default 256:
 *   cc -O2 -DFX_N=1024 -o fxfft_ref fxfft_ref.c
 *
 * The per-size input contract tightens with stage count, because one butterfly
 * maps the working infinity-norm M to at most (1 + sqrt(2) + 2**-15) M + 1:
 *   FX_N   256   512  1024  2048
 *   |x| <= 2**20 2**18 2**17 2**16
 * Deployed row sums are bounded by 128 * K <= 2**14, inside all of them. The
 * 2**20 contract of the eight-stage transform is NOT reusable at nine stages;
 * FX_INPUT_ABS_MAX below is derived per size rather than inherited.
 *
 * Build (plain C, no CUDA needed):  cc -O2 -o fxfft_ref fxfft_ref.c
 * Harness protocol: ./fxfft_ref in.bin out.bin
 *   in.bin  = uint32 LE count n, then n * (FX_N/2) * 2 int32 LE (re, im)
 *   out.bin = n * FX_N * 2 int32 LE (re, im)
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "fxfft_master_twiddle.h"

#ifndef FX_N
#define FX_N 256
#endif

#define FX_NIN (FX_N / 2)
#define FX_SHIFT 15
#define FX_ROUND 16384

/* log2(FX_N), as a compile-time constant expression. */
#if FX_N == 128
#define FX_LOG2 7
#define FX_INPUT_ABS_MAX (1 << 21)
#elif FX_N == 256
#define FX_LOG2 8
#define FX_INPUT_ABS_MAX (1 << 20)
#elif FX_N == 512
#define FX_LOG2 9
#define FX_INPUT_ABS_MAX (1 << 18)
#elif FX_N == 1024
#define FX_LOG2 10
#define FX_INPUT_ABS_MAX (1 << 17)
#elif FX_N == 2048
#define FX_LOG2 11
#define FX_INPUT_ABS_MAX (1 << 16)
#else
#error "FX_N must be one of 128, 256, 512, 1024, 2048 (see fxfft.input_abs_max)"
#endif

#if FX_N > FX_MASTER_N
#error "FX_N exceeds the master twiddle table; raise MASTER_N and regenerate"
#endif

static int32_t round15(int64_t v) { return (int32_t)((v + FX_ROUND) >> FX_SHIFT); }

static unsigned bitrev(unsigned i)
{
    unsigned r = 0, k;
    for (k = 0; k < FX_LOG2; ++k)
        r |= ((i >> k) & 1u) << (FX_LOG2 - 1u - k);
    return r;
}

/* One stream: FX_N/2 (re,im) int32 in, FX_N (re,im) int32 out. */
void fxfft(const int32_t in[FX_NIN][2], int32_t out[FX_N][2])
{
    int i, s, j0, t;
    for (i = 0; i < FX_N; ++i) {
        const unsigned src = bitrev((unsigned)i);
        out[i][0] = (src < (unsigned)FX_NIN) ? in[src][0] : 0;
        out[i][1] = (src < (unsigned)FX_NIN) ? in[src][1] : 0;
    }
    for (s = 1; s <= FX_LOG2; ++s) {
        const int m = 1 << s, half = m >> 1;
        for (j0 = 0; j0 < FX_N; j0 += m) {
            for (t = 0; t < half; ++t) {
                const int32_t *w = FX_TW_MASTER[t << (FX_MASTER_LOG2 - s)];
                const int64_t br = out[j0 + t + half][0];
                const int64_t bi = out[j0 + t + half][1];
                const int32_t tr = round15(br * w[0] - bi * w[1]);
                const int32_t ti = round15(bi * w[0] + br * w[1]);
                const int32_t ar = out[j0 + t][0];
                const int32_t ai = out[j0 + t][1];
                out[j0 + t][0] = ar + tr;
                out[j0 + t][1] = ai + ti;
                out[j0 + t + half][0] = ar - tr;
                out[j0 + t + half][1] = ai - ti;
            }
        }
    }
}

/* Define FXFFT_REF_NO_MAIN to compile only the transform, e.g. when linking
 * this translation unit into another harness. */
#ifndef FXFFT_REF_NO_MAIN
int main(int argc, char **argv)
{
    FILE *fi, *fo;
    uint32_t n, v;
    int i;
    static int32_t in[FX_NIN][2];
    static int32_t out[FX_N][2];

    if (argc != 3) {
        fprintf(stderr, "usage: %s in.bin out.bin (FX_N=%d)\n", argv[0], FX_N);
        return 2;
    }
    /* toolchain self-checks: arithmetic shift + two's complement, matching
     * fxfft256_ref.c, then the rounding rule itself at its boundaries. */
    if ((((int64_t)-3) >> 1) != -2 || (int32_t)0x80000000u >= 0) {
        fprintf(stderr, "toolchain lacks arithmetic shift semantics\n");
        return 2;
    }
    if (round15(16383) != 0 || round15(16384) != 1 || round15(-16384) != 0
        || round15(-16385) != -1 || round15(-32768) != -1) {
        fprintf(stderr, "round15 disagrees with the frozen rounding rule\n");
        return 3;
    }
    fi = fopen(argv[1], "rb");
    fo = fopen(argv[2], "wb");
    if (!fi || !fo) {
        fprintf(stderr, "cannot open files\n");
        return 2;
    }
    if (fread(&n, sizeof n, 1, fi) != 1) {
        fprintf(stderr, "short read\n");
        return 2;
    }
    for (v = 0; v < n; ++v) {
        for (i = 0; i < FX_NIN; ++i)
            if (fread(in[i], sizeof(int32_t), 2, fi) != 2) {
                fprintf(stderr, "short read\n");
                return 2;
            }
        fxfft((const int32_t (*)[2])in, out);
        if (fwrite(out, sizeof(int32_t), (size_t)FX_N * 2, fo) != (size_t)FX_N * 2) {
            fprintf(stderr, "short write\n");
            return 2;
        }
    }
    fclose(fi);
    fclose(fo);
    return 0;
}
#endif /* FXFFT_REF_NO_MAIN */
