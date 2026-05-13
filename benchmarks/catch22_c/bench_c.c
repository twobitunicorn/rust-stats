/* bench_c.c — non-interactive timing harness for the canonical
 * catch22 C library. Mirrors examples/bench_catch22.rs on the Rust
 * side: same n schedule, same xorshift64 + Box-Muller normals, same
 * "median ms per full 22-feature panel" metric.
 *
 * Build alongside libcatch22.a in this directory:
 *   gcc -O3 bench_c.c libcatch22.a -lm -o bench_c
 * Run:
 *   ./bench_c
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "DN_HistogramMode_5.h"
#include "DN_HistogramMode_10.h"
#include "DN_Mean.h"
#include "DN_Spread_Std.h"
#include "CO_AutoCorr.h"
#include "DN_OutlierInclude.h"
#include "FC_LocalSimple.h"
#include "IN_AutoMutualInfoStats.h"
#include "MD_hrv.h"
#include "SB_BinaryStats.h"
#include "SB_MotifThree.h"
#include "SC_FluctAnal.h"
#include "SP_Summaries.h"
#include "SB_TransitionMatrix.h"
#include "PD_PeriodicityWang.h"
#include "stats.h"

/* xorshift64 + Box-Muller — matches examples/bench_catch22.rs RNG so
 * inputs are statistically identical to the Rust bench. */
static uint64_t rng_state;
static uint64_t next_u64(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return rng_state;
}
static double next_normal(void) {
    double u1 = (double)next_u64() / (double)UINT64_MAX;
    if (u1 < 1e-300) u1 = 1e-300;
    double u2 = (double)next_u64() / (double)UINT64_MAX;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

/* Full 22-feature catch22 panel on a *pre-zscored* series, matching
 * what main.c::run_features does between begin/end of each feature
 * timing. We time the whole panel as one unit and include the z-score
 * in the timed block — same convention as our Rust bench. */
static void run_panel(const double *raw, int n, double *zbuf) {
    /* 1. z-score (catch22 convention) */
    zscore_norm2((double *)raw, n, zbuf);

    /* 2. all 22 features, in main.c's order. Discard returned values;
     * we want the wall-clock cost, not the numeric output. */
    volatile double sink = 0.0;
    sink += DN_OutlierInclude_n_001_mdrmd(zbuf, n);
    sink += DN_OutlierInclude_p_001_mdrmd(zbuf, n);
    sink += DN_HistogramMode_5(zbuf, n);
    sink += DN_HistogramMode_10(zbuf, n);
    sink += CO_Embed2_Dist_tau_d_expfit_meandiff(zbuf, n);
    sink += CO_f1ecac(zbuf, n);
    sink += CO_FirstMin_ac(zbuf, n);
    sink += CO_HistogramAMI_even_2_5(zbuf, n);
    sink += CO_trev_1_num(zbuf, n);
    sink += FC_LocalSimple_mean1_tauresrat(zbuf, n);
    sink += FC_LocalSimple_mean3_stderr(zbuf, n);
    sink += IN_AutoMutualInfoStats_40_gaussian_fmmi(zbuf, n);
    sink += MD_hrv_classic_pnn40(zbuf, n);
    sink += SB_BinaryStats_diff_longstretch0(zbuf, n);
    sink += SB_BinaryStats_mean_longstretch1(zbuf, n);
    sink += SB_MotifThree_quantile_hh(zbuf, n);
    sink += SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1(zbuf, n);
    sink += SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1(zbuf, n);
    sink += SP_Summaries_welch_rect_area_5_1(zbuf, n);
    sink += SP_Summaries_welch_rect_centroid(zbuf, n);
    sink += SB_TransitionMatrix_3ac_sumdiagcov(zbuf, n);
    sink += PD_PeriodicityWang_th0_01(zbuf, n);
    /* Touch sink to prevent dead-code elimination. */
    if (sink == -INFINITY) puts("");
}

static int dcmp(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static double median_ms(const int iters, const double *raw, int n, double *zbuf) {
    double *samples = malloc(iters * sizeof *samples);
    /* Warm-up so first-iteration costs (page faults, BSS touches) don't
     * skew the median — matches what the Rust bench does. */
    run_panel(raw, n, zbuf);
    for (int i = 0; i < iters; ++i) {
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        run_panel(raw, n, zbuf);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double secs = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
        samples[i] = secs * 1000.0;
    }
    qsort(samples, iters, sizeof *samples, dcmp);
    double med = samples[iters / 2];
    free(samples);
    return med;
}

int main(void) {
    static const struct { int n, iters; } sched[] = {
        {200,    200},
        {1000,    50},
        {5000,    20},
        {20000,   10},
        {50000,    5},
        {100000,   3},
    };
    const int nsched = sizeof sched / sizeof sched[0];

    rng_state = 0xCAFEBABEull;
    /* one large buffer reused across n's */
    int max_n = 0;
    for (int i = 0; i < nsched; ++i) if (sched[i].n > max_n) max_n = sched[i].n;
    double *raw  = malloc(max_n * sizeof *raw);
    double *zbuf = malloc(max_n * sizeof *zbuf);

    printf("\ncanonical catch22 C kernel, full 22-feature panel\n\n");
    printf("%8s  %10s\n", "n", "ms/call");
    printf("------------------------\n");
    for (int i = 0; i < nsched; ++i) {
        int n = sched[i].n;
        rng_state = 0xCAFEBABEull;
        for (int j = 0; j < n; ++j) raw[j] = next_normal();
        double med = median_ms(sched[i].iters, raw, n, zbuf);
        printf("%8d  %10.3f\n", n, med);
        fflush(stdout);
    }
    free(raw);
    free(zbuf);
    return 0;
}
